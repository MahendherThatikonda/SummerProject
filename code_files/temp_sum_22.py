import pandas as pd 
import numpy as np  # For numerical operations and mathematical computations
from collections import defaultdict  # For creating dictionaries with default values
import matplotlib.pyplot as plt  # For creating visualizations and plots
import seaborn as sns  # For advanced data visualization, complementing matplotlib
import networkx as nx  # For creating and analyzing graphs and networks
from sklearn.manifold import TSNE  # For dimensionality reduction and visualization
import openpyxl  # For reading and writing Excel files
from scipy.sparse.csgraph import laplacian  # For generating graph Laplacian, useful in spectral clustering
from scipy.sparse import csr_matrix  # For working with sparse matrices
from sklearn.cluster import KMeans, AgglomerativeClustering  # For clustering data using the K-means algorithm
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import normalize  # For normalizing data to improve clustering results
import random  # For generating random numbers
import sys  # For interacting with the Python runtime environment
from wordcloud import WordCloud, STOPWORDS  # For generating word clouds and handling common stopwords

print(sys.executable)  # Prints the path of the Python interpreter being used

# Load the dataset
netflix_data = pd.read_csv(r'C:\Users\lenovo\Desktop\STAT 462 Data Mining\Summer Project\netflix_titles.csv')

# Ensure 'date_added' is parsed as a datetime object
netflix_data['date_added'] = pd.to_datetime(netflix_data['date_added'], errors='coerce')

# Filter the dataset for the year 2021
filtered_data = netflix_data[netflix_data['date_added'].dt.year == 2021]
print(len(filtered_data))

# Parse Actor Lists
productions = filtered_data['cast'].str.split(',').apply(lambda x: [actor.strip() for actor in x] if isinstance(x, list) else [])
print("The number of productions are", len(filtered_data))

# Construct Weighted Matrix
weights = defaultdict(int)
for production in productions:
    for i in range(len(production)):
        for j in range(i + 1, len(production)):
            actor1, actor2 = production[i], production[j]
            weights[(actor1, actor2)] += 1
            weights[(actor2, actor1)] += 1

actors = sorted(set(actor for production in productions for actor in production))
print("The number of actors are ", len(actors))

actor_index = {actor: idx for idx, actor in enumerate(actors)}

data_values, row_indices, col_indices = [], [], []
for (actor1, actor2), weight in weights.items():
    i, j = actor_index[actor1], actor_index[actor2]
    row_indices.append(i)
    col_indices.append(j)
    data_values.append(weight)

weighted_matrix_sparse = csr_matrix((data_values, (row_indices, col_indices)), shape=(len(actors), len(actors)))

# Compute Laplacian Matrix and Perform Clustering
laplacian_matrix, _ = laplacian(weighted_matrix_sparse, normed=True, return_diag=True)
eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix.toarray())
num_clusters = 2
selected_eigenvectors = eigenvectors[:, :num_clusters]
print("The eigen vectors are ", selected_eigenvectors)
normalized_eigenvectors = normalize(selected_eigenvectors, norm='l2', axis=1)

# Clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=25)
print("The KMeans is ", kmeans)
clusters = kmeans.fit_predict(normalized_eigenvectors)
print("The clusters are ", clusters)

# Create a graph using networkx
G = nx.Graph()

# Add edges to the graph with weights
for (actor1, actor2), weight in weights.items():
    if weight > 0:  # Optional: Filter out edges with no weight
        G.add_edge(actor1, actor2, weight=weight)

# Generate 3D positions for nodes using a force-directed layout
pos_2d = nx.spring_layout(G, seed=42)  # Generate a 2D layout
pos_3d = {node: (x, y, np.random.rand()) for node, (x, y) in pos_2d.items()}  # Extend to 3D

# Extract node positions
x_vals = [pos_3d[node][0] for node in G.nodes()]
y_vals = [pos_3d[node][1] for node in G.nodes()]
z_vals = [pos_3d[node][2] for node in G.nodes()]

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_vals, y_vals, z_vals, c='blue', s=10, alpha=0.7, label='Actors')

# Draw edges in 3D
for edge in G.edges():
    x_edge = [pos_3d[edge[0]][0], pos_3d[edge[1]][0]]
    y_edge = [pos_3d[edge[0]][1], pos_3d[edge[1]][1]]
    z_edge = [pos_3d[edge[0]][2], pos_3d[edge[1]][2]]
    ax.plot(x_edge, y_edge, z_edge, c='gray', alpha=0.3)

# Set labels and title
ax.set_title("3D Similarity Graph of Actors (2021)", fontsize=14)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.legend()

# Show the plot
plt.show()
