import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph

# Load the dataset
df = pd.read_csv(r'C:\Users\lenovo\Desktop\STAT 462 Data Mining\Summer Project\netflix_titles.csv', low_memory=False)

# Ensure cast column is not empty
df_cast = df[['title', 'cast']].dropna()
df_cast = df_cast[df_cast['cast'].str.strip() != '']

df_cast['cast'] = df_cast['cast'].astype(str)

# Compute TF-IDF for cast names
vectorizer = TfidfVectorizer(stop_words='english')
cast_tfidf = vectorizer.fit_transform(df_cast['cast'])

# Use Gaussian RBF Kernel for similarity
gamma = 1.0 / cast_tfidf.shape[1]  # Scale gamma based on feature size
affinity_matrix_rbf = rbf_kernel(cast_tfidf, gamma=gamma)

# Ensure the graph is fully connected using k-nearest neighbors
connectivity = kneighbors_graph(affinity_matrix_rbf, n_neighbors=5, mode='connectivity', include_self=True)
affinity_matrix_connected = (affinity_matrix_rbf + connectivity.toarray()) / 2

# Ensure the affinity matrix is symmetric
affinity_matrix_connected = (affinity_matrix_connected + affinity_matrix_connected.T) / 2

# Ensure diagonal values are zero
np.fill_diagonal(affinity_matrix_connected, 0)

# Compute normalized Laplacian
L_normalized = csgraph.laplacian(affinity_matrix_connected, normed=True)

# Compute eigenvalues
eigenvalues_fixed, _ = eigsh(L_normalized, k=10, which='SM')

# Plot Eigenvalues for Eigengap Heuristic
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(eigenvalues_fixed) + 1), eigenvalues_fixed, marker="o")
plt.xlabel("Index")
plt.ylabel("Eigenvalue")
plt.title("Eigenvalues of Normalized Laplacian")
plt.show()

# Determine the optimal number of clusters
best_k_fixed = 0
best_score_fixed = -1
silhouette_scores_fixed = []

for k in range(2, 11):  # Test k from 2 to 10
    clustering = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
    labels = clustering.fit_predict(affinity_matrix_connected)
    
    score = silhouette_score(affinity_matrix_connected, labels, metric='precomputed')
    silhouette_scores_fixed.append(score)

    if score > best_score_fixed:
        best_k_fixed = k
        best_score_fixed = score

# Plot silhouette scores
plt.figure(figsize=(6, 4))
plt.plot(range(2, 11), silhouette_scores_fixed, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. Number of Clusters")
plt.show()

print(f"Optimal number of clusters: {best_k_fixed}")
