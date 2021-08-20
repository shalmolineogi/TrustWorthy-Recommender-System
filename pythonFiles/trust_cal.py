from pcc import pcc, similarity_calc
import pandas as pd
import numpy as np

def user_mean(u_id):
    return np.mean(rating_data.loc[u_id].dropna())

def consumer_i(c_id, i_id):
    global consumer_rating
    c_mean = np.mean(user_mean(c_id))
    print(c_mean)
    
    producers = rating_data[i_id].dropna()
    producer_means = np.array([ user_mean(user) for user in producers.keys()])
    producer_consumer_sim = np.array([ sim_matrix[c_id][user] for user in producers.keys()])
    
    producer_diff = np.array(producers.values()) - np.array(producer_means)
    
    sec_num = np.sum(producer_diff * producer_consumer_sim)
    sec_denom = np.sum(np.abs(producer_consumer_sim))
    
    consumer_rating[i_id][c_id] = c_mean + (sec_num/sec_denom)
    

if __name__ == "__main__":
    rating_data = pd.read_csv("UserMovieRatings.csv")
    rating_data.index = rating_data.UserID