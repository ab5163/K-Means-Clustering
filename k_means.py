import random
import numpy as np

class K_means:

    def __init__(self, dataset, K):
        
        self.dataset = dataset
        self.K = K
   
    # Function for random initialization of centroids
    def begin(self):
        centroids = random.sample(self.dataset, self.K)
        return centroids
    
    # Function for clustering datapoints
    def find_dist(self, centroids):
        clusters = [[] for i in range(self.K)]
        dist = []
        for sample in self.dataset:
            dist = [np.sqrt(sum([(sample[i]-centroids[q][i])**2 for i in range(len(self.dataset[0]))])) for q in range(self.K)]
            k = np.argmin(np.array(dist))        
            clusters[k].append(sample)
            dist = []
        return clusters
   
    # Function to find new centroids
    def find_new_centroids(self, clusters):
        average = []
        for q in range(self.K):
            if clusters[q]:
                average.append([np.mean(np.array(clusters[q])[:,i]) for i in range(len(self.dataset[0]))])
        return average
    
    # Function to determine if there is a cluster assignments change
    def cluster_change(self,old_clusters, clusters):
        for i in range(self.K):
            if len(clusters[i]) != len(old_clusters[i]):
                return False
            else:
                for j in range(len(clusters[i])):
                    if clusters[i][j] != old_clusters[i][j]:
                        return False
        return True
