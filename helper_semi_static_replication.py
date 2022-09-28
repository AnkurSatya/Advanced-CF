import torch
from torch import nn

class ANN(nn.Module):
    def __init__(self, no_of_assets, no_hidden_units):
        super(ANN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(no_of_assets, no_hidden_units), 
            nn.ReLU(),
            nn.Linear(no_hidden_units, 1)
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight, a=-0.01, b=0.01)
                # m.bias.data.fill_(0.01)

        self.model.apply(init_weights)

    

    def forward(self, x):
        output = self.model(x)
        return x


def train_one_epoch(model, optimizer, loss_fn, epoch_index, batch_size):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % batch_size == batch_size-1:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            # tb_x = epoch_index * len(training_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def train_model(model, X_train, Y_train, epochs, model_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    best_vloss = 1_000_000.0

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model, optimizer, loss_fn, epoch)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    device = "cpu"
    no_of_assets = 1
    no_hidden_units = 10
    model = ANN(no_of_assets, no_hidden_units).to(device)
    # print(model)
    # print(model.model[2].weight)
    # print(model.model[2].bias)

    # x = torch.rand(1, 1, device=device)
    # output = model(x)
    # print(output)



    ## Loading a saved model
    # model_path = ""
    # saved_model = ANN()
    # saved_model.load_state_dict(torch.load(model_path))