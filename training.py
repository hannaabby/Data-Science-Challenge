import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Create a mapping from node ID to index in the embeddings array
node_id_to_index = {node_id: i for i, node_id in enumerate(G.nodes)}

# Update the get_embedding_pair function to use the mapping
def get_embedding_pair(drug_id, disease_id, embeddings, node_id_to_index):
    drug_index = node_id_to_index[drug_id]
    disease_index = node_id_to_index[disease_id]
    return np.concatenate([embeddings[drug_index], embeddings[disease_index]])

# Prepare feature matrix X and labels y
X = []
y = []

for _, row in ground_truth_df.iterrows():
    drug_id = row['drug_id']
    disease_id = row['disease_id']
    label = row['label']
    embedding_pair = get_embedding_pair(drug_id, disease_id, embeddings_np, node_id_to_index)
    X.append(embedding_pair)
    y.append(label)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

subsample_size = 10000  # Use a subset of the data for SMOTE
if len(X) > subsample_size:
    X_subsample, _, y_subsample, _ = train_test_split(X, y, train_size=subsample_size, random_state=42)
else:
    X_subsample, y_subsample = X, y

# Apply SMOTE to the smaller subset
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_subsample, y_subsample)

# Free unused memory
gc.collect()

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Free unused memory again
gc.collect()

# Define and train an XGBoost classifier with optimized parameters
xgb_clf = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=100,  # Reduce the number of boosting rounds to save memory
    max_depth=5,       # Reduce tree depth to save memory
    learning_rate=0.05,  # Slower learning rate to make training more stable
    subsample=0.8,     # Fraction of training data per boosting round
    colsample_bytree=0.8,  # Fraction of features used per boosting round
    tree_method='hist',  # Use histogram-based method to reduce memory usage
    n_jobs=-1  # Use all CPU cores available
)

# Train the model
xgb_clf.fit(X_train, y_train)

# Free memory after training
gc.collect()

# Evaluate the classifier
y_pred = xgb_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))
