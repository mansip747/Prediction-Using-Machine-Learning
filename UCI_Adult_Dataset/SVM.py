import pandas as pd

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)
df.replace("?", pd.NA, inplace=True)
df = df.dropna()
categorical_columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
df = pd.get_dummies(df, columns=categorical_columns)

# print(df.head())

import pandas as pd

column_to_move = df.columns[6]  # Assuming 0-based indexing
df = df.assign(**{column_to_move: df.pop(column_to_move)})

df

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

y = df.iloc[:, -1]

columns_to_standardize = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
X_to_standardize = df[columns_to_standardize]

scaler = StandardScaler()
X_standardized = pd.DataFrame(scaler.fit_transform(X_to_standardize), columns=columns_to_standardize)

X = pd.concat([X_standardized, df.drop(columns=columns_to_standardize).reset_index(drop=True)], axis=1)

X = X.iloc[:, :-1]

y_new = []
for val in y.values:
  if val[0] == "<":
    y_new.append(0)
  else:
    y_new.append(1)

y = np.array(y_new)
X = X.values

# Apply PCA
pca = PCA(n_components=11, random_state=40)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, hinge_loss
from sklearn.metrics import log_loss

import numpy as np

# Define the RBF Sampler
rbf_sampler = RBFSampler(gamma=0.01, random_state=42)

# Apply kernel approximation to the training and testing data
X_train_transformed = rbf_sampler.fit_transform(X_train)
X_test_transformed = rbf_sampler.transform(X_test)

# Define the model
model = SGDClassifier(random_state=42, )

# Define the hyperparameter grid
param_grid = {
    'alpha': [0.0001, 0.001],
    'penalty': ['l1', 'l2'],
    'max_iter': [1000],
    'loss' : ['hinge'],
    'learning_rate' : ['constant'],
    'eta0' : [0.0001, 0.001, 0.01]
}

# Create GridSearchCV object with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_transformed, y_train)

# Get the best parameters and the corresponding model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

all_models = grid_search.cv_results_['params']

# Use cross_val_score to get cross-validation scores for each model
cross_val_scores = []
for model_params in all_models:
    model = SGDClassifier(**model_params)
    scores = cross_val_score(model, X_train_transformed, y_train, cv=5, scoring='accuracy')  # Fix the variable name
    cross_val_scores.append(scores)

plt.figure(figsize=(12, 8))
sns.boxplot(data=cross_val_scores)
plt.xticks(ticks=range(len(all_models)), labels=['Model' + str(i+1) for i in range(len(all_models))], rotation=45, ha='right')
plt.title('Cross-Validation Scores for Each Model')
plt.xlabel('Model Parameters')
plt.ylabel('Accuracy')
plt.show()

print('Cross validation scores : ', cross_val_scores)

# Simulate partial_fit with batch-wise training
batch_size = 32
num_samples = X_train_transformed.shape[0]
num_batches = int(np.ceil(num_samples / batch_size))

print("Best model : ", best_params)

# Training and validation losses
epochs = 60
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        X_batch = X_train_transformed[batch_start:batch_end]
        y_batch = y_train[batch_start:batch_end]

        # Train the model on the batch
        best_model.partial_fit(X_batch, y_batch, classes=np.unique(y_train))

    # Evaluate on training and validation sets
    train_predictions = best_model.predict(X_train_transformed)
    val_predictions = best_model.predict(X_test_transformed)

    train_loss = hinge_loss(y_train, best_model.decision_function(X_train_transformed))
    val_loss = hinge_loss(y_test, best_model.decision_function(X_test_transformed))

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracy = accuracy_score(y_train, train_predictions)
    val_accuracy = accuracy_score(y_test, val_predictions)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Training Hinge Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Hinge Loss')
plt.title('Training and Validation Hinge Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Hinge Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test_transformed)
y_prob = best_model.decision_function(X_test_transformed)

# Testing accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print("Testing Accuracy:", test_accuracy)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
