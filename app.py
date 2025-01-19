# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# from sklearn.metrics import mean_squared_error
# from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import SGDClassifier
# from sklearn.svm import LinearSVC, SVC
# from sklearn.cluster import KMeans
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.decomposition import PCA
# from sklearn import tree
# import plotly.express as px

# # Streamlit Title
# st.title("Machine Learning Model Visualizer")

# # Algorithm Selection
# st.sidebar.header("Select Algorithm & Parameters")

# algorithms = [
#     "Linear Regression", "Logistic Regression", "Neural Network", "Linear SVM",
#     "Non-linear SVM", "KMeans", "Naive Bayes", "Decision Tree", "PCA", "None"
# ]
# algo_selected = st.sidebar.selectbox("Algorithm:", algorithms)

# # Hyperparameter Controls (Displayed Dynamically)
# if algo_selected in ["Linear Regression", "Logistic Regression", "Linear SVM"]:
#     learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 1.0, 0.001, step=0.0001)
#     reg_strength = st.sidebar.slider("Regularization Strength", 0.0, 1.0, 0.001, step=0.0001)
#     reg_type = st.sidebar.selectbox("Regularization Type", ["L1", "L2"])
#     dataset = st.sidebar.selectbox("Dataset", ["Linear", "Square Root"] if algo_selected == "Linear Regression"
#                                     else ["Uniform", "XOR"])
#     num_epochs = st.sidebar.slider("Epochs", 10, 1000, 300, step=10)

# elif algo_selected == "Neural Network":
#     activation = st.sidebar.selectbox("Activation Function", ["logistic", "tanh", "relu"])
#     learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 1.0, 0.001, step=0.0001)
#     reg_strength = st.sidebar.slider("Regularization Strength", 0.0, 1.0, 0.001, step=0.0001)
#     l1_size = st.sidebar.selectbox("L1 Size", [1, 2, 4, 8], index=3)
#     l2_size = st.sidebar.selectbox("L2 Size", [0, 1, 2, 4, 8], index=2)
#     l3_size = st.sidebar.selectbox("L3 Size", [0, 1, 2, 4, 8], index=0)
#     l4_size = st.sidebar.selectbox("L4 Size", [0, 1, 2, 4, 8], index=0)
#     dataset = st.sidebar.selectbox("Dataset", ["Circular", "XOR"])
#     num_epochs = st.sidebar.slider("Epochs", 10, 1000, 300, step=10)

# elif algo_selected == "KMeans":
#     dataset = st.sidebar.selectbox("Dataset", ["Uniform", "XOR", "With Outliers", "4 Cluster"])
#     n_clusters = st.sidebar.slider("Number of Clusters", 1, 7, 3)
#     num_epochs = st.sidebar.slider("Epochs", 10, 1000, 300, step=10)

# elif algo_selected == "Naive Bayes":
#     dataset = st.sidebar.selectbox("Dataset", ["Independent", "Dependent"])

# elif algo_selected == "Decision Tree":
#     dataset = st.sidebar.selectbox("Dataset", ["Uniform", "XOR"])
#     criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])

# elif algo_selected == "PCA":
#     dataset = st.sidebar.selectbox("Dataset", ["Uniform", "XOR", "Circular"])

# elif algo_selected == "Non-linear SVM":
#     dataset = st.sidebar.selectbox("Dataset", ["Uniform", "XOR", "Circular"])
#     svm_kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
#     poly_degree = st.sidebar.slider("Polynomial Degree", 1, 5, 3)
#     reg_strength = st.sidebar.slider("Regularization Strength", 0.0, 1.0, 0.001, step=0.0001)

# # Function to Execute Selected Model
# def run():
#     st.write(f"### Running {algo_selected} with dataset '{dataset}'")

#     # Display Selected Hyperparameters
#     st.write("**Hyperparameters:**")
#     if algo_selected in ["Linear Regression", "Logistic Regression", "Linear SVM"]:
#         st.write(f"- Learning Rate: {learning_rate}")
#         st.write(f"- Regularization Strength: {reg_strength}")
#         st.write(f"- Regularization Type: {reg_type}")
#         st.write(f"- Epochs: {num_epochs}")

#     elif algo_selected == "Neural Network":
#         st.write(f"- Activation: {activation}")
#         st.write(f"- Learning Rate: {learning_rate}")
#         st.write(f"- Regularization Strength: {reg_strength}")
#         st.write(f"- Hidden Layer Sizes: {l1_size}, {l2_size}, {l3_size}, {l4_size}")
#         st.write(f"- Epochs: {num_epochs}")

#     elif algo_selected == "KMeans":
#         st.write(f"- Number of Clusters: {n_clusters}")
#         st.write(f"- Epochs: {num_epochs}")

#     elif algo_selected == "Decision Tree":
#         st.write(f"- Criterion: {criterion}")

#     elif algo_selected == "Non-linear SVM":
#         st.write(f"- Kernel: {svm_kernel}")
#         st.write(f"- Polynomial Degree: {poly_degree}")
#         st.write(f"- Regularization Strength: {reg_strength}")

#     # Placeholder for Model Execution
#     st.success("Model execution started.")

# # Run Model Button
# if st.button("Run Model"):
#     run()


import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import tree
import seaborn as sns

# Import original functions without modifications
def init_nn(activation='tanh', alpha=0.001, l1_size=8, l2_size=4, l3_size=0, l4_size=0, lr=0.001):
    h_layer_sizes = (l1_size,)
    if l2_size != 0 and l3_size == 0 and l4_size == 0:
        h_layer_sizes = (l1_size, l2_size)
    elif l2_size != 0 and l3_size != 0 and l4_size == 0:
        h_layer_sizes = (l1_size, l2_size, l3_size)
    elif l2_size != 0 and l3_size != 0 and l4_size != 0:
        h_layer_sizes = (l1_size, l2_size, l3_size, l4_size)
    clf = MLPClassifier(activation=activation, solver='adam', alpha=alpha, hidden_layer_sizes=h_layer_sizes,
                        learning_rate="constant", learning_rate_init=lr,
                        random_state=42, max_iter=1, warm_start=True)
    return clf

@st.cache_data
def load_dataset(dataset_name):
    path = f"datasets/{dataset_name}.csv"
    df = pd.read_csv(path)
    return df

# Streamlit UI Setup
st.title("Machine Learning Algorithm Visualizer")
st.sidebar.header("Select Algorithm")

algorithm = st.sidebar.selectbox("Choose an algorithm", [
    "Linear Regression", "Logistic Regression", "Neural Network",
    "Linear SVM", "Non-linear SVM", "KMeans", "Naive Bayes",
    "Decision Tree", "PCA"])

# Hyperparameters UI
if algorithm in ["Linear Regression", "Logistic Regression", "Neural Network", "Linear SVM"]:
    lr = st.sidebar.slider("Learning Rate", 0.0001, 1.0, 0.001, 0.0001)
    alpha = st.sidebar.slider("Regularization Strength", 0.0, 1.0, 0.001, 0.0001)
    reg_type = st.sidebar.selectbox("Regularization Type", ["L1", "L2", "None"])
    epochs = st.sidebar.slider("Epochs", 10, 1000, 300, 10)

if algorithm == "Neural Network":
    activation = st.sidebar.selectbox("Activation Function", ["logistic", "tanh", "relu"])
    l1_size = st.sidebar.selectbox("L1 Layer Size", [1, 2, 4, 8])
    l2_size = st.sidebar.selectbox("L2 Layer Size", [0, 1, 2, 4, 8])
    l3_size = st.sidebar.selectbox("L3 Layer Size", [0, 1, 2, 4, 8])
    l4_size = st.sidebar.selectbox("L4 Layer Size", [0, 1, 2, 4, 8])

if algorithm == "KMeans":
    n_clusters = st.sidebar.slider("Number of Clusters", 1, 7, 3, 1)

if algorithm == "Decision Tree":
    criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])

if algorithm == "Non-linear SVM":
    kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
    degree = st.sidebar.slider("Polynomial Degree", 1, 5, 3, 1)

# Dataset selection
if algorithm == "Linear Regression":
    dataset = st.sidebar.selectbox("Dataset", ["regression_linear_line", "regression_linear_square_root"])
elif algorithm == "Logistic Regression":
    dataset = st.sidebar.selectbox("Dataset", ["classification_linear_uniform", "classification_nonlinear_xor"])
elif algorithm == "KMeans":
    dataset = st.sidebar.selectbox("Dataset", ["clustering_4clusters", "classification_linear_uniform", "classification_nonlinear_xor", "classification_clustering_outliers"])
elif algorithm == "Naive Bayes":
    dataset = st.sidebar.selectbox("Dataset", ["classification_naive_bayes_bernoulli_independent", "classification_dependent_feature"])
else:
    dataset = st.sidebar.selectbox("Dataset", ["classification_circle", "classification_nonlinear_xor", "classification_linear_uniform"])

df = load_dataset(dataset)
# Run Model
if st.sidebar.button("Run Model"):
    st.write(f"### Running {algorithm} on {dataset} dataset")
    
    if algorithm == "Neural Network":
        clf = init_nn(activation=activation, alpha=alpha, l1_size=l1_size, l2_size=l2_size,
                      l3_size=l3_size, l4_size=l4_size, lr=lr)
        X = df[['x1', 'x2']].to_numpy()
        y = df['y'].to_numpy()
        clf.fit(X, y)
        st.success("Model Trained Successfully!")
        
    elif algorithm == "KMeans":
        X = df[['x1', 'x2']].to_numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X)
        fig, ax = plt.subplots()
        plt.scatter(df['x1'], df['x2'], c=df['cluster'], cmap='viridis')
        plt.title("KMeans Clustering")
        st.pyplot(fig)
        
    elif algorithm == "Decision Tree":
        X = df[['x1', 'x2']].to_numpy()
        y = df['y'].to_numpy()
        clf = DecisionTreeClassifier(criterion=criterion)
        clf.fit(X, y)
        fig, ax = plt.subplots(figsize=(8, 4))
        tree.plot_tree(clf, filled=True, ax=ax)
        st.pyplot(fig)
        st.success("Decision Tree Trained Successfully!")
        
    elif algorithm == "PCA":
        X = df[['x1', 'x2']].to_numpy()
        pca = PCA()
        pca.fit_transform(X)
        fig, ax = plt.subplots()
        plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
        plt.title("PCA Projection")
        st.pyplot(fig)
        st.success("PCA Analysis Completed!")
    
    elif algorithm == "Non-linear SVM":
        X = df[['x1', 'x2']].to_numpy()
        y = df['y'].to_numpy()
        clf = SVC(kernel=kernel, degree=degree)
        clf.fit(X, y)
        st.success("Non-linear SVM Model Trained Successfully!")
    
    else:
        st.warning("Algorithm not implemented yet!")

st.write("### Dataset Preview")
fig, ax = plt.subplots()
sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df.iloc[:, -1], palette='coolwarm', edgecolor='black')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
st.pyplot(fig)
