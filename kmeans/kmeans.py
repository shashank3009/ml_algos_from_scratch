import numpy as np
from random import sample
from sklearn.datasets import make_blobs

x, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

def calc_dist(x1, x2):    
    return np.sqrt(np.sum((x1-x2)**2))


def assign_to_clusters(x, clusters):
    
    grp = {i:[] for i in range(len(clusters))}
    
    for _xi in x:
        _d = 10000
        _idx = -1
        for _ki, _ci in enumerate(clusters):
            _dist = calc_dist(_xi, _ci)
            if _dist < _d:
                _idx = _ki
                _d = _dist
                
        grp[_idx].append(_xi)
        
    return grp
                
def calc_new_clusters(grp, clusters):
    
    old_clusters = clusters.copy()
    for key, val in grp.items():
        clusters[key] = np.array(val).mean(axis=0)
    return old_clusters, clusters


def check_clusters_thr(old_c, new_c, thr):
     return (abs(np.array(old_c - new_c)) < thr).all()
 
def group_clusters(x, k, iterations, thr):
    
    clusters = x[np.random.randint(0, len(x), k)]

    for _ in range(iterations):       
        
        grp = assign_to_clusters(x, clusters)
        
        clusters, new_clusters = calc_new_clusters(grp, clusters)
        
        if check_clusters_thr(clusters, new_clusters, thr):
            break
        
    return grp, clusters

grp, clusters  = group_clusters(x, 3, 100, 0.01)

print("groups")
print(grp)

print("\nclusters")
print(clusters)