from torch_geometric.nn import SAGEConv
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
import random
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import BatchNorm
from torch_geometric.utils import from_networkx
import torch
from torch_geometric.data import Data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx

# Load the nodes, edges, and ground truth data from CSV files
# 'Nodes.csv' contains information about different biomedical entities (e.g., drugs, diseases, genes)
# 'Edges.csv' contains relationships between these entities (e.g., drug-targets-gene)
# 'Ground Truth.csv' contains known drug-disease associations with labels (1 for positive, 0 for negative)
nodes_df = pd.read_csv('Nodes.csv')  
edges_df = pd.read_csv('Edges.csv')  
ground_truth_df = pd.read_csv('Ground Truth.csv') 

# Clean up and standardise the column names in the edges dataframe
edges_df.columns = edges_df.columns.str.strip()
edges_df.rename(columns={'subject': 'source', 'object': 'target', 'predicate': 'relationship'}, inplace=True)

# Rename the columns in the ground truth data for consistency and clarity
ground_truth_df.rename(columns={'source': 'drug_id', 'target': 'disease_id', 'y': 'label'}, inplace=True)

# Display the first few rows of each dataframe to verify that the data is loaded correctly
print(nodes_df.head())
print(edges_df.head())
print(ground_truth_df.head())

# Initialise an empty graph using NetworkX to hold the nodes and edges
G = nx.Graph()

# Add nodes to the graph from the nodes dataframe, including their IDs, names, and categories
for _, row in nodes_df.iterrows():
    G.add_node(row['id'], name=row['name'], category=row['category'])

# Add edges to the graph from the edges dataframe, specifying the relationship type as an edge attribute
for _, row in edges_df.iterrows():
    G.add_edge(row['source'], row['target'], relationship=row['relationship'])

# Convert the NetworkX graph into a PyTorch Geometric data object for graph-based machine learning
data = from_networkx(G)

# Define the number of nodes and the feature dimension size
num_nodes = len(G.nodes)
num_features = 16  # Arbitrarily chosen feature dimension for node embeddings

# Generate random initial features for each node (to be refined during model training)
features = torch.rand((num_nodes, num_features), dtype=torch.float)

# Add the features to the data object as node attributes
data.x = features

# Define a GraphSAGE model for link prediction
# The model consists of multiple GraphSAGE layers with batch normalization, dropout, and activation functions
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)  # First GraphSAGE layer
        self.bn1 = BatchNorm(hidden_channels)  # Batch normalization after the first layer
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)  # Second GraphSAGE layer
        self.bn2 = BatchNorm(hidden_channels)  # Batch normalization after the second layer
        self.conv3 = SAGEConv(hidden_channels, out_channels)  # Third GraphSAGE layer for output
        self.dropout = nn.Dropout(p=0.3)  # Dropout layer to prevent overfitting (30% probability)
        self.activation = nn.LeakyReLU(negative_slope=0.1)  # Leaky ReLU activation function

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply the first GraphSAGE layer, followed by batch normalization, activation, and dropout
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Apply the second GraphSAGE layer, followed by batch normalization, activation, and dropout
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Apply the third GraphSAGE layer to generate the final node embeddings
        x = self.conv3(x, edge_index)
        return x

# Initialize the GraphSAGE model, optimizer, and loss function
model = GraphSAGE(in_channels=num_features, hidden_channels=64, out_channels=32)
optimizer = optim.Adam(model.parameters(), lr=0.005)  # Adam optimizer with a lower learning rate

# Calculate the ratio of negative to positive samples to address class imbalance
num_positive = len(ground_truth_df[ground_truth_df['label'] == 1])
num_negative = len(ground_truth_df[ground_truth_df['label'] == 0])
pos_weight_value = num_negative / num_positive  # Used for weighting the loss function

# Define the loss function with a class imbalance adjustment using the calculated positive weight
pos_weight = torch.tensor([pos_weight_value], device='cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Move the data and model to the appropriate device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = model.to(device)

# Create a mapping from node IDs to indices in the embeddings array for easy access
node_id_to_index = {node_id: i for i, node_id in enumerate(G.nodes)}

# Prepare positive samples (drug-disease pairs) for link prediction from the ground truth data
positive_pairs = []
labels = []

for _, row in ground_truth_df.iterrows():
    drug_id = row['drug_id']
    disease_id = row['disease_id']
    label = row['label']
    if label == 1:
        positive_pairs.append((node_id_to_index[drug_id], node_id_to_index[disease_id]))
        labels.append(1)

# Generate negative samples for training by performing negative sampling
edge_index = data.edge_index
num_negatives = len(positive_pairs)
neg_edge_index = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=num_negatives)

# Extract the negative pairs from the generated edges and add them to the training set
negative_pairs = []
for i in range(neg_edge_index.size(1)):
    drug_index, disease_index = neg_edge_index[0, i].item(), neg_edge_index[1, i].item()
    negative_pairs.append((drug_index, disease_index))
    labels.append(0)

# Combine positive and negative samples into one dataset
all_pairs = positive_pairs + negative_pairs
labels = torch.tensor(labels, dtype=torch.float, device=device)

# Convert the edge pairs and labels to tensors for the DataLoader
edge_pairs = torch.tensor(all_pairs, dtype=torch.long)
labels_tensor = labels

# Create a DataLoader for batching edge pairs and labels during training
dataset = TensorDataset(edge_pairs, labels_tensor)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

# Define a function for training the model using mini-batches for link prediction
def train_link_prediction():
    model.train()
    total_loss = 0
    for edge_pairs_batch, labels_batch in dataloader:
        optimizer.zero_grad()

        # Forward pass to obtain node embeddings from the model
        embeddings = model(data)

        # Prepare the embeddings for the edges in the current batch
        drug_indices_batch = edge_pairs_batch[:, 0].to(device)
        disease_indices_batch = edge_pairs_batch[:, 1].to(device)
        edge_embeddings_batch = torch.cat([embeddings[drug_indices_batch], embeddings[disease_indices_batch]], dim=1)

        # Make predictions by summing the embedding pairs
        predictions_batch = torch.sum(edge_embeddings_batch, dim=1).view(-1)

        # Calculate the loss using the true labels
        labels_batch = labels_batch.view(-1).to(device)
        loss = loss_fn(predictions_batch, labels_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Train the model for a specified number of epochs and print the loss every 10 epochs
for epoch in range(100):
    loss = train_link_prediction()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Obtain the node embeddings from the trained GraphSAGE model
with torch.no_grad():
    embeddings = model(data)

# Convert the embeddings to a NumPy array for further processing or analysis
embeddings = embeddings.cpu().numpy()

# Save the node embeddings to a CSV file for future use
embedding_df = pd.DataFrame(embeddings)
embedding_df.insert(0, 'node_id', list(G.nodes))  # Include node IDs as the first column
embedding_df.to_csv('node_embeddings.csv', index=False)

print("Embeddings saved to node_embeddings.csv")




