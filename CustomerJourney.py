import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Generate Sample Customer Journey Dataset
def generate_sample_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'Page_Views': np.random.randint(1, 100, 200),
        'Time_Spent_Minutes': np.random.randint(5, 300, 200),
        'Clicks': np.random.randint(1, 50, 200),
        'Purchases': np.random.randint(0, 5, 200),
        'Bounce_Rate': np.random.uniform(10, 90, 200)
    })
    return data

# Preprocess Data
def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Dimensionality Reduction using PCA
def apply_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca

# Clustering using K-Means
def apply_clustering(data, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans

# Visualization Function
def plot_clusters(reduced_data, clusters):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette='tab10')
    plt.title('Customer Journey Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    st.pyplot(plt)

# Streamlit App
def main():
    st.title("Customer Journey Analysis Using Clustering")

    # Generate and Display Data
    st.write("### Sample Customer Journey Data")
    data = generate_sample_data()
    st.write(data.head())

    # Preprocess Data
    scaled_data = preprocess_data(data)

    # Apply PCA
    reduced_data, pca = apply_pca(scaled_data)
    st.write("### Explained Variance by PCA Components")
    st.bar_chart(pca.explained_variance_ratio_)

    # Apply Clustering
    clusters, kmeans = apply_clustering(reduced_data)
    data['Cluster'] = clusters

    # Show Clustered Data
    st.write("### Clustered Customer Data")
    st.write(data.head())

    # Visualize Clusters
    st.write("### Cluster Visualization")
    plot_clusters(reduced_data, clusters)

if __name__ == "__main__":
    main()
