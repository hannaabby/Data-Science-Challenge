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

# Load the nodes, edges, and ground truth data
nodes_df = pd.read_csv('Nodes.csv')  
edges_df = pd.read_csv('Edges.csv')  
ground_truth_df = pd.read_csv('Ground Truth.csv') 

edges_df.columns = edges_df.columns.str.strip()
edges_df.rename(columns={'subject': 'source', 'object': 'target', 'predicate': 'relationship'}, inplace=True)
ground_truth_df.rename(columns={'source': 'drug_id', 'target': 'disease_id', 'y': 'label'}, inplace=True)
print(nodes_df.head())
print(edges_df.head())
print(ground_truth_df.head())

G = nx.Graph()

for _, row in nodes_df.iterrows():
    G.add_node(row['id'], name=row['name'], category=row['category'])
    
for _, row in edges_df.iterrows():
    G.add_edge(row['source'], row['target'], relationship=row['relationship'])

data = from_networkx(G)

num_nodes = len(G.nodes)
num_features = 16
features = torch.rand((num_nodes, num_features), dtype=torch.float)

# Add features to the data object
data.x = features

# Define GraphSAGE model for link prediction
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=0.3)  # Dropout with 30% probability
        self.activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply GraphSAGE layers with batch normalization, activation, and dropout
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        return x

# Initialize model, optimizer, and loss function
model = GraphSAGE(in_channels=num_features, hidden_channels=64, out_channels=32)
optimizer = optim.Adam(model.parameters(), lr=0.005)  # Lower learning rate to stabilize training

# Calculate the ratio of negative to positive samples
num_positive = len(ground_truth_df[ground_truth_df['label'] == 1])
num_negative = len(ground_truth_df[ground_truth_df['label'] == 0])
pos_weight_value = num_negative / num_positive  # This helps in balancing the class contributions

# Modify the loss function
pos_weight = torch.tensor([pos_weight_value], device='cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Move the data and model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = model.to(device)

# Create a mapping from node ID to index in the embeddings array
node_id_to_index = {node_id: i for i, node_id in enumerate(G.nodes)}

# Prepare positive samples for link prediction
positive_pairs = []
labels = []

# Add positive pairs from the ground truth
for _, row in ground_truth_df.iterrows():
    drug_id = row['drug_id']
    disease_id = row['disease_id']
    label = row['label']
    if label == 1:
        positive_pairs.append((node_id_to_index[drug_id], node_id_to_index[disease_id]))
        labels.append(1)

# Negative sampling for negative pairs
edge_index = data.edge_index
num_negatives = len(positive_pairs)
neg_edge_index = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=num_negatives)

# Extract negative pairs from generated edge index
negative_pairs = []
for i in range(neg_edge_index.size(1)):
    drug_index, disease_index = neg_edge_index[0, i].item(), neg_edge_index[1, i].item()
    negative_pairs.append((drug_index, disease_index))
    labels.append(0)

# Combine positive and negative pairs
all_pairs = positive_pairs + negative_pairs
labels = torch.tensor(labels, dtype=torch.float, device=device)

# Training loop for link prediction using mini-batches
edge_pairs = torch.tensor(all_pairs, dtype=torch.long)
labels_tensor = labels

# Prepare DataLoader for batching edge pairs and labels
dataset = TensorDataset(edge_pairs, labels_tensor)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

def train_link_prediction():
    model.train()
    total_loss = 0
    for edge_pairs_batch, labels_batch in dataloader:
        optimizer.zero_grad()

        # Forward pass to get node embeddings
        embeddings = model(data)

        # Prepare edge embeddings for link prediction (in batch)
        drug_indices_batch = edge_pairs_batch[:, 0].to(device)
        disease_indices_batch = edge_pairs_batch[:, 1].to(device)
        edge_embeddings_batch = torch.cat([embeddings[drug_indices_batch], embeddings[disease_indices_batch]], dim=1)

        # Make predictions
        predictions_batch = torch.sum(edge_embeddings_batch, dim=1).view(-1)

        # Calculate loss
        labels_batch = labels_batch.view(-1).to(device)
        loss = loss_fn(predictions_batch, labels_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Train the model using batched link prediction
for epoch in range(100):
    loss = train_link_prediction()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Get node embeddings from the trained GraphSAGE model
with torch.no_grad():
    embeddings = model(data)

# Convert embeddings to numpy for use in scikit-learn and further processing
embeddings = embeddings.cpu().numpy()
# Save embeddings to a CSV file
embedding_df = pd.DataFrame(embeddings_np)
embedding_df.insert(0, 'node_id', list(G.nodes))  # Insert node IDs as the first column
embedding_df.to_csv('node_embeddings.csv', index=False)

print("Embeddings saved to node_embeddings.csv")


