from k_means import K_means
import numpy as np
import matplotlib.pyplot as plt

dataset = np.random.rand(100,2).tolist() # not limited to 2D dataset
K = 5 
k_mean = K_means(dataset,K)
n_iter = 100 # max number of iterations
i = 0
done = False
centroids = k_mean.begin() # initialize centroids
clusters = k_mean.find_dist(centroids) # initialize clusters

while not done:
    
    centroids = k_mean.find_new_centroids(clusters)
    old_clusters = clusters
    clusters = k_mean.find_dist(centroids)
    done = k_mean.cluster_change(old_clusters, clusters) or i>n_iter
    i += 1

# Assigning colors to each clusters and combining them
plot_dataset = np.ones((1,3))
for i in range(K):
    plot_clusters = np.column_stack((np.array(clusters[i]),i*np.ones((len(clusters[i]),1))))
    plot_dataset = np.concatenate((plot_dataset,plot_clusters))

plot_dataset = plot_dataset[1:]
plt.scatter(plot_dataset[:,0], plot_dataset[:,1], c=plot_dataset[:,2], s=50, cmap='viridis')
plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], c='black', s=200, alpha=0.5);   
