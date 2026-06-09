import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Page Configuration
st.set_page_config(page_title="t-SNE Visualization", layout="wide")

st.title("📊 t-SNE Visualization - Mall Customers Dataset")

# Load Dataset
df = pd.read_csv("data/Mall_Customers.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Select numerical columns
numeric_df = df.select_dtypes(include=["int64", "float64"])

# Remove CustomerID if present
if "CustomerID" in numeric_df.columns:
    numeric_df = numeric_df.drop(columns=["CustomerID"])

st.subheader("Features Used")
st.write(numeric_df.columns.tolist())

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

# t-SNE Parameters
st.subheader("t-SNE Parameters")

perplexity = st.slider(
    "Perplexity",
    min_value=5,
    max_value=50,
    value=30
)

# Apply t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

# Create DataFrame
tsne_df = pd.DataFrame(
    X_tsne,
    columns=["t-SNE Component 1", "t-SNE Component 2"]
)

# Show transformed data
st.subheader("t-SNE Transformed Data")
st.dataframe(tsne_df.head())

# Visualization
st.subheader("t-SNE Visualization")

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    tsne_df["t-SNE Component 1"],
    tsne_df["t-SNE Component 2"],
    alpha=0.7
)

ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")
ax.set_title("2D t-SNE Projection of Mall Customers")

st.pyplot(fig)

st.success("t-SNE Dimensionality Reduction Completed Successfully!")