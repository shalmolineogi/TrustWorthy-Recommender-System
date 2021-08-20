import pandas as pd
import numpy as np

def pcc(u, v):
    u=np.array(u)
    v=np.array(v)
    u_part = u - np.mean(u)
    v_part = v - np.mean(v)
    
    num = np.sum(u_part * v_part)
    
    din1 = np.sqrt(np.sum((np.square(u_part))))
    din2 = np.sqrt(np.sum((np.square(v_part))))
    
    if din1==0 or din2==0:
        return 1
    
    return num / (din1 * din2)

def pcc_2(u,v):
    u=np.array(u)
    v=np.array(v)
    
    num = 50 * np.sum(u*v)  - np.sum(u) * np.sum(v)

    din1 = np.sqrt( 50 * np.sum(u*u) - np.square(np.sum(u)))
    din2 = np.sqrt( 50 * np.sum(v*v) - np.square(np.sum(v)))
    
    if din1==0 or din2==0:
        return 1
    return num / (din1 * din2)

def pcc_3(u,v):
    u=np.array(u)
    v=np.array(v)
    
    u_mean, v_mean = np.mean(u), np.mean(v)
    
    num = np.sum(u*v)  - (50*u_mean* v_mean)

    din1 = np.sqrt(np.sum(u*u) - (50*u_mean*u_mean))
    din2 = np.sqrt(np.sum(v*v) - (50*v_mean*v_mean))
    
    if din1==0 or din2==0:
        return 1
    return num / (din1 * din2)

def similarity_calc(feature, metric=pcc):
    n_user,n_feature = feature.shape
    
    sim_matrix = np.eye(n_user)
    
    for i in range(n_user):
        u = feature.loc[i]
        for j in range(i+1, n_user):
            v = feature.loc[j]
            sim_matrix[i][j] = metric(u,v)
            sim_matrix[j][i] = sim_matrix[i][j]
            
    return sim_matrix

if __name__ == "__main__":
    user_feature = pd.read_csv("user_features.csv")
    del user_feature['Unnamed: 0']
    
    # u = user_feature.loc[0]
    # v = user_feature.loc[100]
    
    # print(pcc(u,v))
    # print(pcc_2(u,v))
    # print(pcc_3(u,v))
    sim_matrix = similarity_calc(user_feature, metric = pcc)
    pd.DataFrame(sim_matrix).to_csv("similarity.csv")