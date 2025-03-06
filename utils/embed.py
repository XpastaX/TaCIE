from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.manifold import TSNE
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
import torch


def uniformity_measure(points):
    tree = KDTree(points)
    distances, _ = tree.query(points, k=2)  # k=2 because the closest point is the point itself
    nearest_neighbor_distances = distances[:, 1]  # distances to the nearest neighbor
    variance = np.var(nearest_neighbor_distances)
    return np.round(variance, 4)


def get_emb(text: list, model_path='ckpt/sentence_transformer/all-MiniLM-L6-v2', d_reduction=None, seed=42, model=None,
            device='cpu'):
    if model is None:
        model = SentenceTransformer(model_path).to(device)
    embeddings = model.encode(text)
    if d_reduction is not None:
        embeddings = get_tsne(embeddings, d_reduction, seed)
    return embeddings


def get_tsne(embeddings, target_dimension, seed=42):
    tsne = TSNE(n_components=target_dimension, random_state=seed)
    embeddings = tsne.fit_transform(embeddings)
    return embeddings


def cluster(emb, num_clusters, seed=42):
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
    kmeans.fit(emb)
    return kmeans


def plot_elbow(emb, num_clusters: list, seed=42):
    inertias = []
    for c in tqdm(num_clusters):
        kmean = cluster(emb, c, seed)
        inertias.append(kmean.inertia_)
    matplotlib.use('TkAgg')
    plt.plot(num_clusters, inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()
    val = input("Enter best cluster number: ")
    return int(val)


def plot_list(y, x=None, size=None, save_path=None, xlabel=None, ylabel=None, show=False):
    if x is None:
        x = list(range(1, y + 1))
    matplotlib.use('TkAgg')
    if size is not None:
        plt.figure(figsize=size)
    plt.plot(x, y, marker='o')
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')


def find_nearest_sample_to_cluster_mean(embeddings, cluster_labels, cluster_centers):
    nearest_samples = []
    for _cluster, center in enumerate(cluster_centers):
        candidate_idx = [i for i, sample in enumerate(embeddings) if cluster_labels[i] == _cluster]
        candidate = [embeddings[i] for i in candidate_idx]
        # print(f"Cluster{_cluster} #Candidate:{len(candidate)}")
        candidate = np.vstack(candidate)
        distances = np.linalg.norm(candidate - center, axis=1)
        nearest_sample_index = np.argmin(distances)
        nearest_samples.append(candidate_idx[nearest_sample_index])
    # count={i:0 for i in nearest_samples}
    # print(len(count))
    return nearest_samples


def cal_diversity(embeddings: np.ndarray, cache_path='cache/s7'):
    num_embeddings = embeddings.shape[0]
    try:
        max_sim = torch.load(f'{cache_path}/max_sim_{num_embeddings}.torch')
    except:
        # Define a function to compute the maximum similarity for a given embedding
        def compute_max_similarity(i):
            sim = cosine_similarity(embeddings[i:i + 1], embeddings)[0]
            sim[i] = -2
            return max(sim)

        # Use joblib to parallelize the computation of max_sim
        max_sim = Parallel(n_jobs=10)(delayed(compute_max_similarity)(i) for i in tqdm(range(num_embeddings)))
        max_sim = np.array(max_sim)
        torch.save(max_sim, f'{cache_path}/max_sim_{num_embeddings}.torch')
    div = (1 - max_sim) / 2
    # diversity = []
    # for k in range(1, 6):
    #     d = np.power(sum(np.power(div, k)) / num_embeddings, 1 / k)
    #     diversity.append(d)
    #     print(k, d)
    return np.var(div)


def most_dissimilar_embeddings_bruteforce(embeddings, k, metric='cosine'):
    """
    Selects a subgroup of k embeddings that are most dissimilar to each other using brute force.

    Parameters:
    embeddings (np.ndarray): 2D array of embeddings (num_embeddings x embedding_dim), the first embedding is the anchor.
    k (int): Number of most dissimilar embeddings to select

    Returns:
    np.ndarray: Indices of the selected embeddings
    """
    num_embeddings = embeddings.shape[0]
    if k > num_embeddings:
        raise ValueError("k cannot be greater than the number of embeddings")

    if metric == 'distance':
        # Compute the distance matrix
        matrix = squareform(pdist(embeddings, 'euclidean'))
    elif metric == 'cosine':
        # Compute the cosine distance matrix
        matrix = 1 - cosine_similarity(embeddings)
    else:
        raise NotImplementedError("metric is not implemented")

    # Initialize the best score and the best combination
    max_dissimilarity = -1000000000000
    best_combination = None

    # Iterate over all combinations
    for combo in combinations(range(num_embeddings), k):
        # make sure the original seed is in the combination
        if 0 not in combo: continue
        # Calculate the sum of all pairwise distances in this combination
        combo_dissimilarity = 1
        for i in combo:
            for j in combo:
                if i < j:
                    combo_dissimilarity *= matrix[i, j]

        # If this combination has a higher dissimilarity, update the best found
        if combo_dissimilarity > max_dissimilarity:
            max_dissimilarity = combo_dissimilarity
            best_combination = combo

    return np.array(best_combination)


def find_best_diverse_group(data, metric='cosine'):
    emb = np.vstack([sample['emb'] for sample in data])
    combination_list = [[0]]
    for k in range(2, len(data) + 1):
        best_combination = most_dissimilar_embeddings_bruteforce(emb, k, metric=metric)
        combination_list.append(list(best_combination))
    return combination_list
