import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Create a mapping from each node ID to its index in the embeddings array.
# This is necessary for efficiently retrieving the embeddings of specific nodes (e.g., drugs and diseases)
node_id_to_index = {node_id: i for i, node_id in enumerate(G.nodes)}

# Define a function to retrieve the combined embeddings of a drug and disease pair.
# The function uses the node_id_to_index mapping to find the correct indices in the embeddings array.
def get_embedding_pair(drug_id, disease_id, embeddings, node_id_to_index):
    drug_index = node_id_to_index[drug_id]  # Get the index of the drug node
    disease_index = node_id_to_index[disease_id]  # Get the index of the disease node
    # Concatenate the embeddings of the drug and disease to form a feature vector
    return np.concatenate([embeddings[drug_index], embeddings[disease_index]])

# Prepare feature matrix X and label vector y
# X will contain the concatenated embeddings for each drug-disease pair, and y will have the corresponding labels
X = []
y = []

# Iterate through the ground truth dataset to create features and labels
for _, row in ground_truth_df.iterrows():
    drug_id = row['drug_id']
    disease_id = row['disease_id']
    label = row['label']
    # Use the get_embedding_pair function to extract features for each drug-disease pair
    embedding_pair = get_embedding_pair(drug_id, disease_id, embeddings_np, node_id_to_index)
    X.append(embedding_pair)  # Add the feature vector to X
    y.append(label)  # Add the corresponding label to y

# Convert the lists to NumPy arrays for compatibility with scikit-learn
X = np.array(X)
y = np.array(y)

# Subsample the data to a maximum of 10,000 samples to make processing faster
# This helps in applying SMOTE effectively without overwhelming memory resources
subsample_size = 10000
if len(X) > subsample_size:
    # Randomly select a subset of the data if the dataset is too large
    X_subsample, _, y_subsample, _ = train_test_split(X, y, train_size=subsample_size, random_state=42)
else:
    # Use the entire dataset if it is already within the limit
    X_subsample, y_subsample = X, y

# Apply SMOTE (Synthetic Minority Over-sampling Technique) to the subset to balance the classes
# SMOTE generates synthetic samples for the minority class to improve the classifier's ability to generalize
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_subsample, y_subsample)

# Free unused memory to avoid excessive memory consumption
gc.collect()

# Split the balanced data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Free unused memory again to keep resource usage efficient
gc.collect()

# Define an XGBoost classifier with optimized parameters for training on the prepared dataset
# XGBoost is chosen for its ability to handle imbalanced datasets and provide strong performance
xgb_clf = XGBClassifier(
    use_label_encoder=False,  # Disable use of the label encoder (for compatibility)
    eval_metric='logloss',  # Use log loss as the evaluation metric
    n_estimators=100,  # Reduce the number of boosting rounds to save memory and speed up training
    max_depth=5,       # Use shallow trees to prevent overfitting and manage memory usage
    learning_rate=0.05,  # Use a slower learning rate to improve stability during training
    subsample=0.8,     # Use 80% of the training data for each boosting round (to prevent overfitting)
    colsample_bytree=0.8,  # Use 80% of the features for each tree (to prevent overfitting)
    tree_method='hist',  # Use the histogram-based method to reduce memory usage and speed up training
    n_jobs=-1  # Utilize all available CPU cores for training to accelerate the process
)

# Train the XGBoost model on the training data
xgb_clf.fit(X_train, y_train)

# Free memory after the model has been trained
gc.collect()

# Evaluate the trained classifier on the test set to determine its performance
y_pred = xgb_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')

# Print a detailed classification report, including precision, recall, and F1-score
# This helps in understanding how well the model is performing for each class (e.g., true positives vs. false positives)
print(classification_report(y_test, y_pred))

