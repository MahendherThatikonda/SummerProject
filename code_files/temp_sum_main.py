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
from sklearn.cluster import KMeans,AgglomerativeClustering  # For clustering data using the K-means algorithm
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import normalize  # For normalizing data to improve clustering results
import random  # For generating random numbers
import sys  # For interacting with the Python runtime environment
print(sys.executable)  # Prints the path of the Python interpreter being used
from wordcloud import WordCloud, STOPWORDS  # For generating word clouds and handling common stopwords

# Load the dataset
netflix_data = pd.read_csv(r'C:\Users\lenovo\Desktop\STAT 462 Data Mining\Summer Project\netflix_titles.csv')



# Ensure 'date_added' is parsed as a datetime object
netflix_data['date_added'] = pd.to_datetime(netflix_data['date_added'], errors='coerce')

# Filter the dataset for the year 2021
filtered_data = netflix_data[netflix_data['date_added'].dt.year == 2021]#filtered_data = netflix_data[netflix_data['date_added'].str.contains('2021', na=False)]
print(len(filtered_data))
# Parse Actor Lists
productions = filtered_data['cast'].str.split(',').apply(lambda x: [actor.strip() for actor in x] if isinstance(x, list) else [])
print("The number of productions are",len(filtered_data))
# Construct Weighted Matrix
weights = defaultdict(int)
for production in productions:
    for i in range(len(production)):
        for j in range(i + 1, len(production)):
            actor1, actor2 = production[i], production[j]
            weights[(actor1, actor2)] += 1
            weights[(actor2, actor1)] += 1

actors = sorted(set(actor for production in productions for actor in production))
print("The number of actors are ",len(actors))

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
print("The eigen vectors are ",selected_eigenvectors)
normalized_eigenvectors = normalize(selected_eigen+vectors, norm='l2', axis=1)

# Clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=25)
print("The KMEans is ",kmeans)
clusters = kmeans.fit_predict(normalized_eigenvectors)
print("The clusters are ", clusters)
# Map Clusters Back to Filtered Data
# Ensure 'filtered_data' aligns with the clustering result
filtered_data = filtered_data.reset_index(drop=True)
filtered_data['clusters'] = clusters[:len(filtered_data)]  # Align lengths

# Form Cluster Groups
cluster_groups = filtered_data.groupby('clusters')
print("The c;uster groups are ",cluster_groups)
def summarize_group(group):
    return {
        'Countries': group['country'].value_counts().to_dict(),
        'Genres': group['listed_in'].value_counts().to_dict(),
#        'Years': group['release_year'].value_counts().to_dict(),
        'Ratings': group['rating'].value_counts().to_dict(),
 #       'Durations': group['duration'].value_counts().to_dict(),
 #       'Productions': len(group)
    }

cluster_summaries = cluster_groups.apply(summarize_group, include_groups=False)

# Save summaries to CSV
cluster_summary_df = pd.DataFrame(cluster_summaries.tolist(), index=cluster_summaries.index)
#cluster_summary_df.to_csv('cluster_summary.csv')
cluster_summary_df.to_csv(r'C:\Users\lenovo\Desktop\Cluster_2\cluster_summary.csv')

print(cluster_summary_df)
# Display the filtered data
print(filtered_data.head())


print(cluster_summary_df)
# Display the filtered data
print(filtered_data.head())


# Display the filtered data with cast and clusters
cast_clusters = filtered_data[['cast', 'clusters']]

# Split the cast into individual actors with their clusters
expanded_cast_clusters = (
    cast_clusters
    .assign(cast=cast_clusters['cast'].str.split(','))  # Split the cast into lists
    .explode('cast')  # Expand rows for each actor
    .dropna(subset=['cast'])  # Drop rows where cast is NaN
    .assign(cast=lambda df: df['cast'].str.strip())  # Strip whitespace
)

print(expanded_cast_clusters)

# Saving the expanded cast-cluster assignments to a CSV
expanded_cast_clusters.to_csv(r'C:\Users\lenovo\Desktop\Cluster_2\cast_clusters.csv', index=False)



#import matplotlib.pyplot as plt

# Combine all descriptions for each cluster
descriptions_by_cluster = (
    filtered_data.groupby('clusters')['description']
    .apply(lambda desc: ' '.join(desc.dropna()))  # Combine descriptions
)

# Find intersection words across all clusters
all_descriptions = ' '.join(descriptions_by_cluster)
all_words = set(all_descriptions.split())
cluster_words = [set(desc.split()) for desc in descriptions_by_cluster]
intersection_words = set.intersection(*cluster_words)

# Define additional stopwords
custom_stopwords = {'friends', 'family', 'life', 'love', 'new', 'story', 'people', 'world', 'find', 'man', 'woman','young','take'}
# Import stopwords
stopwords = set(STOPWORDS).union(custom_stopwords).union(intersection_words)

# Generate and display a word cloud
def generate_word_cloud(text, cluster_id):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords,
        collocations=False
    ).generate(text)
    
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Cluster {cluster_id}')
    plt.show()

# Generate word clouds for each cluster
for cluster_id, descriptions in descriptions_by_cluster.items():
    if descriptions.strip():  # Ensure there is text to process
        generate_word_cloud(descriptions, cluster_id)


#Histogram of clusters by frequency
 # Count the number of items in each cluster
cluster_counts = filtered_data['clusters'].value_counts()

# Plot the histogram
plt.figure(figsize=(12, 8))
plt.bar(cluster_counts.index, cluster_counts.values)
plt.xlabel('Cluster ID', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.title('Number of Productions in Each Cluster', fontsize=16)
plt.xticks(cluster_counts.index, fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Histogram of release years by clusters
for cluster_id, group in cluster_groups:
    plt.figure(figsize=(12, 8))
    group['release_year'].hist(bins=20, alpha=0.7, label=f'Cluster {cluster_id}')
    plt.xlabel('Release Year', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.title(f'Distribution of Release Years in Cluster {cluster_id}', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.show()
'''#Histogram of Genres by Cluster 
for cluster_id, group in cluster_groups:
    genre_counts = group['listed_in'].value_counts().head(10)
    plt.figure(figsize=(15, 10))
    genre_counts.plot(kind='bar', alpha=0.7)
    plt.xlabel('Genre')
    plt.ylabel('Frequency')
    plt.title(f'Genre Distribution in Cluster {cluster_id}')
    plt.xticks(rotation=45)
    plt.show()'''

for cluster_id, group in cluster_groups:
    genre_counts = group['listed_in'].value_counts().head(10)
    plt.figure(figsize=(14, 10))
    genre_counts.plot(kind='bar', alpha=0.7)
    plt.xlabel('Genre', fontsize=16)  # Increase label font size
    plt.ylabel('Frequency', fontsize=16)  # Increase label font size
    plt.title(f'Genre Distribution in Cluster {cluster_id}', fontsize=18)  # Increase title font size
    plt.xticks(rotation=45, fontsize=14)  # Rotate labels and increase font size
    plt.yticks(fontsize=14)  # Adjust Y-axis font size
    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.show()


# Histogram of clusters by Country
for cluster_id, group in cluster_groups:
    country_counts = group['country'].value_counts().head(10)
    plt.figure(figsize=(12, 8))
    country_counts.plot(kind='bar', alpha=0.7)
    plt.xlabel('Country', fontsize=14)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Country Distribution in Cluster {cluster_id}')
    plt.xticks(rotation=45)
    plt.show()

# Ratings distribution
for cluster_id, group in cluster_groups:
    plt.figure(figsize=(12, 8))
    group['rating'].value_counts().plot(kind='bar', alpha=0.7)
    plt.xlabel('Rating',fontsize=16)
    plt.ylabel('Frequency',fontsize=16)
    plt.title(f'Rating Distribution in Cluster {cluster_id}',fontsize=18)
    plt.xticks(rotation=45,fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

# Durations distribution
for cluster_id, group in cluster_groups:
    plt.figure(figsize=(12, 8))
    group['duration'].value_counts().head(10).plot(kind='bar', alpha=0.7)
    plt.xlabel('Duration', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.title(f'Duration Distribution in Cluster {cluster_id}', fontsize=18)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


# Combined Histogram
plt.figure(figsize=(12, 8))
sns.histplot(data=filtered_data, x='clusters', bins=num_clusters, kde=False, discrete=True)
plt.xlabel('Cluster ID', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.title('Overall Distribution of Clusters', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


'''THIS IS 3D

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize

# Assuming you already have 'normalized_eigenvectors' and 'clusters' from your clustering code
'''
# Apply PCA to reduce dimensions to 3
'''pca = PCA(n_components=2)
reduced_data = pca.fit_transform(normalized_eigenvectors)

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each cluster with a different color
for cluster_id in sorted(set(clusters)):
    cluster_points = reduced_data[clusters == cluster_id]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {cluster_id}', s=30)

# Add plot details
ax.set_title('3D Similarity Graph of Clusters')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.legend()
plt.show()'''


'''siilarity graph'''

'''
# Create a graph using networkx
G = nx.Graph()

# Add edges to the graph with weights
for (actor1, actor2), weight in weights.items():
    if weight > 0:  # Optional: Filter out edges with no weight
        G.add_edge(actor1, actor2, weight=weight)

# Plot the graph
plt.figure(figsize=(15, 15))
pos = nx.spring_layout(G, seed=42)  # Positions nodes using a spring layout
nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.7)
nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=8, font_color='black', alpha=0.9)

plt.title("Similarity Graph of Actors (2021)", fontsize=16)
plt.axis('off')
plt.show()
'''