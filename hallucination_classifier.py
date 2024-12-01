import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import joblib

# Load dataset
df = pd.read_csv('updated_dataset.csv')

# Extract features and labels
features = df[['context_response_similarity', 'query_response_similarity', 'response_perplexity']].values
labels = df['label'].apply(lambda x: 1 if x == 'Hallucinatable' else 0).values

# Standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize classifiers
rf_clf = RandomForestClassifier(random_state=42)
gb_clf = GradientBoostingClassifier(random_state=42)

# Create a parameter grid for GridSearchCV
param_grid = {
    'random_forest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
    },
    'gradient_boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
    }
}

# Perform GridSearchCV to find the best hyperparameters
def perform_grid_search(clf, param_grid, X_train, y_train):
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

print("Performing Grid Search for Random Forest...")
best_rf_clf = perform_grid_search(rf_clf, param_grid['random_forest'], X_train, y_train)
print("Performing Grid Search for Gradient Boosting...")
best_gb_clf = perform_grid_search(gb_clf, param_grid['gradient_boosting'], X_train, y_train)

# Evaluate the classifiers
def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))
    return y_pred, y_prob

print("Evaluating Random Forest...")
rf_pred, rf_prob = evaluate_model(best_rf_clf, X_test, y_test)
print("Evaluating Gradient Boosting...")
gb_pred, gb_prob = evaluate_model(best_gb_clf, X_test, y_test)

# Plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Hallucinatable', 'Hallucinatable'], yticklabels=['Non-Hallucinatable', 'Hallucinatable'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

print("Plotting Confusion Matrix for Random Forest...")
plot_confusion_matrix(y_test, rf_pred, 'Confusion Matrix - Random Forest')
print("Plotting Confusion Matrix for Gradient Boosting...")
plot_confusion_matrix(y_test, gb_pred, 'Confusion Matrix - Gradient Boosting')

# Plot ROC curve
def plot_roc_curve(y_test, y_prob, title):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

print("Plotting ROC Curve for Random Forest...")
plot_roc_curve(y_test, rf_prob, 'ROC Curve - Random Forest')
print("Plotting ROC Curve for Gradient Boosting...")
plot_roc_curve(y_test, gb_prob, 'ROC Curve - Gradient Boosting')

# Save the best models and the scaler
joblib.dump(best_rf_clf, 'best_rf_clf.pkl')
joblib.dump(best_gb_clf, 'best_gb_clf.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Models and scaler saved.")
