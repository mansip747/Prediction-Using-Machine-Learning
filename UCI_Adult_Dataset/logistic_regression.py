import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, roc_curve 
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

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

column_to_move = df.columns[6]  # Assuming 0-based indexing
df = df.assign(**{column_to_move: df.pop(column_to_move)})

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = SGDClassifier(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'alpha': [0.00001, 0.0001, 0.001, 0.01, .1],
    'penalty': ['l1', 'l2'],
    'max_iter': [1000],
    'loss' : ['log_loss'],
    'learning_rate' : ['constant'],
    'eta0' : [0.00001, 0.0001, 0.001, 0.01, .1]
}

# Create GridSearchCV object with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and the corresponding model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
   
all_models = grid_search.cv_results_['params']

# Use cross_val_score to get cross-validation scores for each model
cross_val_scores = []
for model_params in all_models:
    model = SGDClassifier(**model_params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cross_val_scores.append(scores)

plt.figure(figsize=(12, 8))
sns.boxplot(data=cross_val_scores)
plt.xticks(ticks=range(len(all_models)))    
    
    
# Simulate partial_fit with batch-wise training
batch_size = X_train.shape[0]
num_samples = X_train.shape[0]
num_batches = int(np.ceil(num_samples / batch_size))
print("Best model : ", best_params)

# Training and validation losses
epochs = [20, 80]
for each in epochs:
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # best_sgd_model = SGDClassifier(**best_params)
    best_sgd_model = SGDClassifier(alpha=best_params['alpha'], eta0=best_params['eta0'],
                                   learning_rate=best_params['learning_rate'],loss=best_params['loss'],
                                   max_iter=best_params['max_iter'],penalty=best_params['penalty'],random_state=42)

    # Learning rate schedule
    initial_learning_rate = 0.01
    learning_rate = initial_learning_rate

    # Early stopping parameters
    patience = 5
    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(each):
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            X_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]

            # Train the model on the batch
            best_sgd_model.partial_fit(X_batch, y_batch, classes=np.unique(y_train))
        # print(epoch)

        # Evaluate on training and validation sets
        train_predictions = best_sgd_model.predict(X_train)
        val_predictions = best_sgd_model.predict(X_test)    
        train_loss = log_loss(y_train, train_predictions)
        val_loss = log_loss(y_test, val_predictions)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracy = accuracy_score(y_train, train_predictions)
        val_accuracy = accuracy_score(y_test, val_predictions)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    # Plotting
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Log Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Log Loss')
    plt.title('Training and Validation Log Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Evaluate the best model on the test set
    y_pred = best_sgd_model.predict(X_test)
    y_prob = best_sgd_model.decision_function(X_test)

    # Testing accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Testing Accuracy:", test_accuracy)

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(confusion_mat)


    # ROC curve and AUC score
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
