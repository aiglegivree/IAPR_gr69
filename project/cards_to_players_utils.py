from sklearn.cluster import DBSCAN
import numpy as np

def clustering(card_points, eps=525, min_samples=1):
    card_points = np.asarray(card_points)
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(card_points)
    cluster_centers = []
    for label in np.unique(clusters.labels_):
        cluster_points = card_points[clusters.labels_ == label]
        centroid = np.mean(cluster_points, axis=0)
        cluster_centers.append(centroid)
    cluster_centers = np.array(cluster_centers)

    return cluster_centers, clusters.labels_

def cluster_2_player_mapping(cluster_centers, cluster_labels, W, H, buffer=200):
    cluster_centers = np.asarray(cluster_centers)
    cluster_labels = np.asarray(cluster_labels)
    unique_cluster_labels = np.unique(cluster_labels)
    player_positions = np.array([
        [W // 2, H - buffer],
        [W - buffer, H // 2],
        [W // 2, buffer],
        [buffer, H // 2]
        
    ])
    distances = np.linalg.norm(cluster_centers[:, np.newaxis] - player_positions, axis=2)
    closest_players = np.argmin(distances, axis=1) + 1
    map_cluster2player = dict(zip(unique_cluster_labels, closest_players))
    player_ids = [map_cluster2player.get(label, None) for label in cluster_labels]

    return np.array(player_ids)
