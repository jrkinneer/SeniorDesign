import numpy as np
import random
import scipy
from tqdm import tqdm

def kmeans_multiple(X, K, iters, R):
    orig_1 = X.shape[0]
    orig_2 = X.shape[1]

    m = X.shape[0] * X.shape[1]
    n = X.shape[2]
    X = X.reshape((m, n))
    ids = np.empty([m,1])
    
    means = np.empty([K, n])
    
    #find the range of values in m
    #min_m is of size n, holds mins of all measurements of n_i
    min_m = np.min(X, axis=0)
    
    max_m = np.max(X, axis=0)
    
    #find best random means to start
    best_mean = np.empty([K, n])
    current_mean = np.empty([K, n])
    
    start_ssd = 0
    current_ssd = 0
    for y in range(K):
        for z in range(n):
            current_mean[y][z] = random.randint(int(min_m[z]), int(max_m[z]))
    best_mean = current_mean
    
    #finds distances between all m points and current K means
    distances = scipy.spatial.distance.cdist(X, current_mean)
    for z in range(m):
        #find minimum distance of each point and attach it to the ids list
        closest_cluster_idx = np.argmin(distances[z])
        ids[z][0] = closest_cluster_idx
    #calculate ssd
    for i in range(m):
        cluster = int(ids[i][0])
        distance = X[i] - current_mean[cluster]
        start_ssd += distance**2
    
    #do the above R times and figure out lowest ssd and use that for iters loops
    for r in tqdm(range(R), leave=False):
        current_mean = np.empty([K, n])
        current_ssd = 0
        for y in range(K):
            for z in range(n):
                current_mean[y][z] = random.randint(int(min_m[z]), int(max_m[z]))
        
        #finds distances between all m points and current K means
        distances = scipy.spatial.distance.cdist(X, current_mean)
        for z in range(m):
            #find minimum distance of each point and attach it to the ids list
            closest_cluster_idx = np.argmin(distances[z])
            ids[z][0] = closest_cluster_idx
        #calculate ssd
        for i in range(m):
            cluster = int(ids[i][0])
            distance = X[i] - current_mean[cluster]
            current_ssd += distance**2
    
        if (np.sum(current_ssd) < np.sum(start_ssd)):
            best_mean = current_mean
            start_ssd = current_ssd
        
    means = best_mean
    #loop over in range iters
    for y in tqdm(range(iters), leave=False):
        
        #finds distances between all m points and current K means
        distances = scipy.spatial.distance.cdist(X, means)
        for z in range(m):
            #find minimum distance of each point and attach it to the ids list
            closest_cluster_idx = np.argmin(distances[z])
            ids[z][0] = closest_cluster_idx
          
        #calculate new average  
        sum = np.zeros([K,n])
        points_in_cluster = [0] * K
        for v in range(m):
            current_cluster = int(ids[v][0])
            points_in_cluster[current_cluster] += 1
            for a in range(n):
                sum[current_cluster][a] += X[v][a] 
            
        for i in range(K):
            for j in range(n):
                if (points_in_cluster[i] != 0):
                    means[i][j] = sum[i][j]/points_in_cluster[i]
           
    #recolor image to be a cluster
    new_img = np.zeros((m, n))
    
    for i in tqdm(range(ids.shape[0]), leave=False):
        cluster = int(ids[i][0])
        new_img[i] = means[cluster]
        
    new_img = new_img.reshape((orig_1, orig_2, 3)).astype('uint8')
    return new_img, means, ids

def kmeans_multiple_grayscale(X, K, iters, R):
    orig_1 = X.shape[0]
    orig_2 = X.shape[1]

    m = X.shape[0] * X.shape[1]
    n = 1
    X = X.reshape((m, n))
    ids = np.empty([m,1])
    
    means = np.empty([K, n])
    
    #find the range of values in m
    #min_m is of size n, holds mins of all measurements of n_i
    min_m = np.min(X, axis=0)
    
    max_m = np.max(X, axis=0)
    
    #find best random means to start
    best_mean = np.empty([K, n])
    current_mean = np.empty([K, n])
    
    start_ssd = 0
    current_ssd = 0
    for y in range(K):
        for z in range(n):
            current_mean[y][z] = random.randint(int(min_m[z]), int(max_m[z]))
    best_mean = current_mean
    
    #finds distances between all m points and current K means
    distances = scipy.spatial.distance.cdist(X, current_mean)
    for z in range(m):
        #find minimum distance of each point and attach it to the ids list
        closest_cluster_idx = np.argmin(distances[z])
        ids[z][0] = closest_cluster_idx
    #calculate ssd
    for i in range(m):
        cluster = int(ids[i][0])
        distance = X[i] - current_mean[cluster]
        start_ssd += distance**2
    
    #do the above R times and figure out lowest ssd and use that for iters loops
    for r in tqdm(range(R), leave=False):
        current_mean = np.empty([K, n])
        current_ssd = 0
        for y in range(K):
            for z in range(n):
                current_mean[y][z] = random.randint(int(min_m[z]), int(max_m[z]))
        
        #finds distances between all m points and current K means
        distances = scipy.spatial.distance.cdist(X, current_mean)
        for z in range(m):
            #find minimum distance of each point and attach it to the ids list
            closest_cluster_idx = np.argmin(distances[z])
            ids[z][0] = closest_cluster_idx
        #calculate ssd
        for i in range(m):
            cluster = int(ids[i][0])
            distance = X[i] - current_mean[cluster]
            current_ssd += distance**2
    
        if (np.sum(current_ssd) < np.sum(start_ssd)):
            best_mean = current_mean
            start_ssd = current_ssd
        
    means = best_mean
    #loop over in range iters
    for y in tqdm(range(iters), leave=False):
        
        #finds distances between all m points and current K means
        distances = scipy.spatial.distance.cdist(X, means)
        for z in range(m):
            #find minimum distance of each point and attach it to the ids list
            closest_cluster_idx = np.argmin(distances[z])
            ids[z][0] = closest_cluster_idx
          
        #calculate new average  
        sum = np.zeros([K,n])
        points_in_cluster = [0] * K
        for v in range(m):
            current_cluster = int(ids[v][0])
            points_in_cluster[current_cluster] += 1
            for a in range(n):
                sum[current_cluster][a] += X[v][a] 
            
        for i in range(K):
            for j in range(n):
                if (points_in_cluster[i] != 0):
                    means[i][j] = sum[i][j]/points_in_cluster[i]
           
    #recolor image to be a cluster
    new_img = np.zeros((m, n))
    
    for i in tqdm(range(ids.shape[0]), leave=False):
        cluster = int(ids[i][0])
        new_img[i] = means[cluster]
        
    new_img = new_img.reshape((orig_1, orig_2)).astype('uint8')
    return new_img, means.astype('uint8'), ids