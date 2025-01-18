import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import tree
import plotly.express as px

# Streamlit Title
st.title("Machine Learning Model Interactive Dashboard")

# Algorithm Selection
st.sidebar.header("Select Algorithm & Parameters")

algorithms = [
    "Linear Regression", "Logistic Regression", "Neural Network", "Linear SVM",
    "Non-linear SVM", "KMeans", "Naive Bayes", "Decision Tree", "PCA", "None"
]
algo_selected = st.sidebar.selectbox("Algorithm:", algorithms)

# Hyperparameter Controls (Displayed Dynamically)
if algo_selected in ["Linear Regression", "Logistic Regression", "Linear SVM"]:
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 1.0, 0.001, step=0.0001)
    reg_strength = st.sidebar.slider("Regularization Strength", 0.0, 1.0, 0.001, step=0.0001)
    reg_type = st.sidebar.selectbox("Regularization Type", ["L1", "L2"])
    dataset = st.sidebar.selectbox("Dataset", ["Linear", "Square Root"] if algo_selected == "Linear Regression"
                                    else ["Uniform", "XOR"])
    num_epochs = st.sidebar.slider("Epochs", 10, 1000, 300, step=10)

elif algo_selected == "Neural Network":
    activation = st.sidebar.selectbox("Activation Function", ["logistic", "tanh", "relu"])
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 1.0, 0.001, step=0.0001)
    reg_strength = st.sidebar.slider("Regularization Strength", 0.0, 1.0, 0.001, step=0.0001)
    l1_size = st.sidebar.selectbox("L1 Size", [1, 2, 4, 8], index=3)
    l2_size = st.sidebar.selectbox("L2 Size", [0, 1, 2, 4, 8], index=2)
    l3_size = st.sidebar.selectbox("L3 Size", [0, 1, 2, 4, 8], index=0)
    l4_size = st.sidebar.selectbox("L4 Size", [0, 1, 2, 4, 8], index=0)
    dataset = st.sidebar.selectbox("Dataset", ["Circular", "XOR"])
    num_epochs = st.sidebar.slider("Epochs", 10, 1000, 300, step=10)

elif algo_selected == "KMeans":
    dataset = st.sidebar.selectbox("Dataset", ["Uniform", "XOR", "With Outliers", "4 Cluster"])
    n_clusters = st.sidebar.slider("Number of Clusters", 1, 7, 3)
    num_epochs = st.sidebar.slider("Epochs", 10, 1000, 300, step=10)

elif algo_selected == "Naive Bayes":
    dataset = st.sidebar.selectbox("Dataset", ["Independent", "Dependent"])

elif algo_selected == "Decision Tree":
    dataset = st.sidebar.selectbox("Dataset", ["Uniform", "XOR"])
    criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])

elif algo_selected == "PCA":
    dataset = st.sidebar.selectbox("Dataset", ["Uniform", "XOR", "Circular"])

elif algo_selected == "Non-linear SVM":
    dataset = st.sidebar.selectbox("Dataset", ["Uniform", "XOR", "Circular"])
    svm_kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
    poly_degree = st.sidebar.slider("Polynomial Degree", 1, 5, 3)
    reg_strength = st.sidebar.slider("Regularization Strength", 0.0, 1.0, 0.001, step=0.0001)

# Function to Execute Selected Model
def run():
    st.write(f"### Running {algo_selected} with dataset '{dataset}'")

    # Display Selected Hyperparameters
    st.write("**Hyperparameters:**")
    if algo_selected in ["Linear Regression", "Logistic Regression", "Linear SVM"]:
        st.write(f"- Learning Rate: {learning_rate}")
        st.write(f"- Regularization Strength: {reg_strength}")
        st.write(f"- Regularization Type: {reg_type}")
        st.write(f"- Epochs: {num_epochs}")

    elif algo_selected == "Neural Network":
        st.write(f"- Activation: {activation}")
        st.write(f"- Learning Rate: {learning_rate}")
        st.write(f"- Regularization Strength: {reg_strength}")
        st.write(f"- Hidden Layer Sizes: {l1_size}, {l2_size}, {l3_size}, {l4_size}")
        st.write(f"- Epochs: {num_epochs}")

    elif algo_selected == "KMeans":
        st.write(f"- Number of Clusters: {n_clusters}")
        st.write(f"- Epochs: {num_epochs}")

    elif algo_selected == "Decision Tree":
        st.write(f"- Criterion: {criterion}")

    elif algo_selected == "Non-linear SVM":
        st.write(f"- Kernel: {svm_kernel}")
        st.write(f"- Polynomial Degree: {poly_degree}")
        st.write(f"- Regularization Strength: {reg_strength}")

    # Placeholder for Model Execution
    st.success("Model execution started. (Replace this with actual ML training code)")

# Run Model Button
if st.button("Run Model"):
    run()
