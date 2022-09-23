import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm


def build_tree(S, vol, T, N):
    dt = 1.0*T/N
    
    matrix = np.zeros((N+1, N+1))
    
    u = np.exp(vol * (dt)**0.5)
    d = np.exp(-vol * (dt)**0.5)
#     print("u: {}\nd: {}".format(u, d))
    
    for i in range(N+1):
        for j in range(i+1):
            matrix[i, j] = S*(u**j)*(d**(i-j))
            
    return matrix


def get_option_value_binomial_matrix(tree, T, r, K, vol, N, exercise_time, option_type="call"):
    dt = 1.0*T/N
    
    u = np.exp(vol * (dt)**0.5)
    d = np.exp(-vol * (dt)**0.5)

    # print(u, d)
    
    p = (np.exp(r*dt) - d)/(u-d)
#     print("u: {}, d: {}, p: {}".format(u,d, p))
    
    rows, cols = tree.shape

    # ToDo:May be replace with the following algorithm: 
    # a = (exercise_time[1] - exercise_time[0])*N
    # exercise_points = [N*exercise_time[0], N*exercise_time[0] + a, N*exercise_time[0] + 2*a, ...]
    # exercise_points = [int(N*val) for val in exercise_time]

    # exercise_points = np.arange(N/len(exercise_time), N, N/len(exercise_time))
    # exercise_points[-1] = N

    a = 1.0*N/365
    exercise_points = [int(pt) for pt in np.arange(30*a, N, 30*a)]

    print(exercise_points)
    
    # Adding payoff value for the last step of the tree
    for c in range(cols):
        S_T = tree[rows-1, c]
        if option_type == "call":
            tree[rows-1, c] = max(0, S_T - K)
        else:
            tree[rows-1, c] = max(0, K - S_T)

    # Adding payoff for the rest of the steps
    for i in range(rows-1-1, -1, -1):
        for j in range(i+1):
            f_u = tree[i+1, j+1]
            f_d = tree[i+1, j]

            # print("p, f_u, f_d, r, dt: ", p, f_u, f_d, r, dt)
            holding_value = (p*f_u + (1-p)*f_d)*np.exp(-r*dt) # Option value if option is kept until the expiry time
            # print("holding value: ", holding_value)

            # if is_european:
            #     f = holding_value
            # else:
            if i in exercise_points:
                if option_type == "call":
                    exercise_value = tree[i,j] - K # Option value if exercised(completed) at this time.
                else:
                    exercise_value = K - tree[i,j]
                f = max(exercise_value, holding_value)
            else:
                f = holding_value

            tree[i, j] = f
        
    return tree

# def get_option_value_scholes(S_t, r, K, vol, T, t=0, option_type="call"):
#     rv = norm()
#     tau = T-t
#     d1 = (np.log(S_t/K) + (r+0.5*vol**2)*tau)/(vol*(tau**0.5))
#     d2 = d1 - vol*(tau**0.5)

#     if option_type == "call":    
#         option_value = S_t * rv.cdf(d1) - (np.exp(-r*tau))*K*rv.cdf(d2)
#     else:
#         option_value = (np.exp(-r*tau))*K*rv.cdf(-d2) - S_t * rv.cdf(-d1) 
    
#     return option_value