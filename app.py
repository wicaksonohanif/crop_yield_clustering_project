import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from scipy.stats import yeojohnson
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load Data
def load_data():
    df = pd.read_csv('yield_df.csv')
    return df

# Data Preprocessing
def preprocess_data(df):
    df_clean = df.drop(columns=['Unnamed: 0', 'Area', 'Item', 'Year', 'hg/ha_yield', 'pesticides_tonnes'])
    df_clean['average_rain_fall_mm_per_year'] = np.sqrt(df_clean['average_rain_fall_mm_per_year'])
    df_clean['avg_temp'], _ = yeojohnson(df_clean['avg_temp'])
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_clean)
    return df, df_clean, df_scaled

# Clustering
def cluster_data(df_scaled, df):
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=20, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    return df, kmeans

# Sidebar Navigation
st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio("Go to", ["Home 🏠", "Data Overview 👁️", "Data Visualization 📈", "Modeling & Evaluation 📊"])

# Load and preprocess data
df = load_data()
df, df_clean, df_scaled = preprocess_data(df)
df, kmeans = cluster_data(df_scaled, df)

# Home Page
if page == "Home 🏠":
    st.markdown("""
        <h1 style='text-align: center;'>🌾 Crop Yield Clustering Based on Rainfall & Temperature ☔📈</h1>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.image("home_image.jpg", use_column_width=True)
    st.markdown("**Dashboard Description:**")
    st.write("This dashboard contains information about the harvest results in several areas that are influenced by several internal and external factors. The main purpose of this dashboard is to review the grouping (Clusters) of the harvest results of a crop based on average rainfall and temperature.")

# Data Overview
elif page == "Data Overview 👁️":
    st.markdown("<h1 style='text-align: center;'>Data Overview 👁️</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Dataset Source:** https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.subheader("Data Types")
    st.write(df.dtypes)
    st.subheader("Dataset Shape")
    st.write(f'Columns: {df.shape[1]}, Rows: {df.shape[0]}')
    st.subheader("Statistical Summary")
    st.write(df.describe(include='all'))


# Data Visualization
elif page == "Data Visualization 📈":
    st.markdown("<h1 style='text-align: center;'>Data Visualization 📈</h1>", unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Scatter Plot: Rainfall vs Yield")
    fig, ax = plt.subplots()
    sns.scatterplot(x='average_rain_fall_mm_per_year', y='hg/ha_yield', data=df, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Scatter Plot: Temperature vs Yield")
    fig, ax = plt.subplots()
    sns.scatterplot(x='avg_temp', y='hg/ha_yield', data=df, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Boxplot of Numerical Features")
    features = ['Year', 'hg/ha_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        if i < len(axes):
            sns.boxplot(x=df[feature], ax=axes[i])
            axes[i].set_title(feature)
    plt.tight_layout()
    st.pyplot(fig)

    features = ["Year", "hg/ha_yield", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
    st.subheader("Feature Distribution (Histogram)")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, feature in enumerate(features):
        row, col = divmod(i, 3)
        sns.histplot(df[feature], bins=30, kde=True, ax=axes[row, col])
        axes[row, col].set_title(f"{feature}")
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Geospatial Visualization")
    world = gpd.read_file('ne_110m_admin_0_countries.shp')
    selected_countries = df['Area'].unique()
    world_selected = world[world['ADMIN'].isin(selected_countries)]
    fig, ax = plt.subplots(figsize=(12, 8))
    world.plot(ax=ax, color='lightgrey', edgecolor='black')
    world_selected.plot(ax=ax, color='lightgreen', edgecolor='green')
    plt.title("Map of Countries in the Dataset")
    st.pyplot(fig)

    st.subheader("Feature Correlation Heatmap")
    df_heatmap = df.drop(columns=['Unnamed: 0'])
    fig, ax = plt.subplots()
    sns.heatmap(df_heatmap.select_dtypes(exclude='object').corr(), ax=ax, cmap='coolwarm', annot=True)
    st.pyplot(fig)

# Modeling & Evaluation
elif page == "Modeling & Evaluation 📊":
    st.markdown("<h1 style='text-align: center;'>Modeling & Evaluation 📊</h1>", unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Elbow Method for Cluster Selection")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=20, random_state=0)
        kmeans.fit(df_scaled)
        wcss.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_title('Elbow Method')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('WCSS')
    st.pyplot(fig)
    
    st.subheader("Clustering Results")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df_clean['average_rain_fall_mm_per_year'], y=df_clean['avg_temp'], hue=df['Cluster'], palette='Set2', legend='full', ax=ax)
    ax.set_title('Clustering Results')
    st.pyplot(fig)
    
    st.subheader("Crop Yield Distribution per Cluster")
    cluster_counts = df.groupby("Cluster")['Item'].value_counts().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_counts.plot(kind="bar", ax=ax)
    ax.set_title("Crop Yield per Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.legend(title="Crop")
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Cluster Statistics")
    for i in range(2):
        st.markdown(f"**Cluster {i}:**") 
        st.write(df[df['Cluster'] == i].describe())

