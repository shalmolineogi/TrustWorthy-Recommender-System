import numpy as np
import pandas as pd

def foo(n_user, trust_matrix, rating_data, sim_matrix, user_mean, T):
    # global trust_matrix
    # global rating_data
    # global sim_matrix

    error = []

    for c_id in range(1, n_user + 1):
        c_rating = rating_data.loc[c_id].dropna()
        c_mean = user_mean(c_id)

        c_p = trust_matrix.loc[c_id]
        c_producers = c_p[c_p>=T]

        print(c_id)
        # print(c_producers)
        for p_id in c_producers.keys():
            producer_rating = rating_data.loc[int(p_id)].dropna()
            p_mean = user_mean(int(p_id))

            keys = set(c_rating.keys()).intersection(set(producer_rating.keys()))
            if len(keys)==0:
                continue
            
            producer_rating = producer_rating[list(keys)]
            c_p_rating = c_rating[list(keys)]

            c_pred = c_mean + (producer_rating - p_mean) * trust_matrix[p_id][c_id]

            error+=list((c_pred - c_p_rating))
        
        # print(c_id)

    print("mse: ", np.mean(np.square(np.array(error))))
    return error
    
    







