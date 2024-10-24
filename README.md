# Drug-Disease Link Prediction Using Graph Embeddings

 The pipeline demonstrates the process of drug repurposing by utilizing a biological knowledge graph that contains drugs, diseases, genes, and other biomedical entities.
 By analyzing a biological knowledge graph with nodes and edges representing relationships between drugs, diseases, and other entities, the goal is to:

Generate embeddings for the nodes using a Graph Neural Network (GNN).
Train a classifier to predict drug-disease links based on these embeddings.
Evaluate the model on a test set of known drug-disease pairs.

# Approach and Design

# Step 1: Data Preparation
Load Data: The data includes nodes (Nodes.csv), edges (Edges.csv), and ground truth drug-disease associations (Ground Truth.csv).
Graph Construction: Using the NetworkX library, a graph is built where nodes represent biomedical entities, and edges represent their known relationships.
Graph Transformation: The graph is converted into a format compatible with the PyTorch Geometric library to facilitate graph-based machine learning.

# Step 2: Embedding Generation
GraphSAGE Model: A GraphSAGE model is defined to generate node embeddings. It utilizes a series of graph convolutional layers combined with batch normalization, dropout, and Leaky ReLU activation.
Training: The GraphSAGE model is trained using positive and negative samples of drug-disease pairs. Negative samples are generated through negative sampling techniques.
Embedding Extraction: After training, node embeddings are extracted, saved, and used for the classification step.

# Step 3: Link Prediction Classifier
Feature Extraction: Drug-disease pairs are represented by combining their node embeddings.
SMOTE for Data Balancing: SMOTE (Synthetic Minority Over-sampling Technique) is applied to handle imbalanced classes in the dataset.
Classifier Training: An XGBoost classifier is trained using these features, optimized for memory usage and performance.

# Step 4: Evaluation
The trained classifier is evaluated on a test set, and metrics such as accuracy, precision, and recall are reported.

# Dependencies

The code requires the following Python libraries:

torch
torch-geometric
networkx
pandas
scikit-learn
imblearn (for SMOTE)
xgboost
numpy
gc (Garbage Collection)

To install the dependencies, run:
pip install torch torch-geometric networkx pandas scikit-learn imbalanced-learn xgboost
Dataset

# Place the following files in your working directory:

Nodes.csv: Contains node information (drugs, diseases, genes, etc.)
Edges.csv: Contains relationships between nodes (e.g., drug-treats-disease, drug-targets-gene)
Ground Truth.csv: Known drug-disease associations with labels (1 for positive, 0 for negative)
Ensure that the column names are appropriately formatted as shown in the provided script.

# How to Run the Code

Prepare the Data: Ensure the CSV files are placed in the correct path.
Run the Script: Execute the Python script to start the embedding generation, training, and evaluation process.

python drug_disease_link_prediction.py

View Results: The script will print training progress and final evaluation metrics to the console. Node embeddings will be saved to node_embeddings.csv.
Note

The script has been designed to run on both CPU and GPU. If you have a CUDA-enabled GPU, the code will automatically utilize it, improving performance.

# Evaluation

The model evaluation provides key performance metrics:
Accuracy: Measures the overall prediction correctness.
Classification Report: Includes precision, recall, and F1-score, offering a deeper insight into how well the model identifies drug-disease links.

