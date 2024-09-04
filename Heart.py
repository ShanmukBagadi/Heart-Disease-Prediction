import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# Ignore warnings
warnings.filterwarnings('ignore')

# Read the CSV file into a DataFrame
heart_data = pd.read_csv("../Assignment/heart.csv")

# Print the first 5 rows of the dataset
print(heart_data.head())

# Print the last 5 rows of the dataset
print(heart_data.tail())

# Number of rows and columns
print(heart_data.shape)

# Getting some info about the data
print(heart_data.info())

# Checking for missing values
print(heart_data.isnull().sum())

# No missing values

# Get statistical measures about the data
print(heart_data.describe())

# Checking the distribution of target variable
print(heart_data['target'].value_counts())
# 1 have defects, 0 not present (Healthy)

# Split the features and Targets
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Set up the figure and axis
plt.figure(figsize=(10, 6))

# Plot density plot for cholesterol levels of patients with heart disease
sns.histplot(heart_data[heart_data['target'] == 1]['chol'], color='blue', label='Heart Disease', kde=True)

# Plot density plot for cholesterol levels of patients without heart disease
sns.histplot(heart_data[heart_data['target'] == 0]['chol'], color='red', label='No Heart Disease', kde=True)

# Set labels and title
plt.xlabel('Cholesterol Level')
plt.ylabel('Density')
plt.title('Distribution of Cholesterol Levels')
plt.legend()

# Show the plot
plt.show()

# Cross-tabulation
cross_tab = pd.crosstab(heart_data['cp'], heart_data['target'])
cross_tab.plot(kind='bar', stacked=True)
plt.xlabel('Number of Chest Pain Types')
plt.ylabel('Count')
plt.title('Distribution of Heart Disease by Chest Pain Types')
plt.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.show()

# Select the variables of interest for correlation
variables_of_interest = ['age', 'chol', 'thalach']

# Calculate the correlation matrix
correlation_matrix = heart_data[variables_of_interest].corr()

# Filter the correlation matrix to include only positive correlations
positive_correlation_matrix = correlation_matrix[correlation_matrix > 0]

# Find feature pairs with positive correlation
positive_correlation_features = positive_correlation_matrix[positive_correlation_matrix < 1].stack().reset_index()

# Rename the columns
positive_correlation_features.columns = ['Feature 1', 'Feature 2', 'Correlation']

# Filter out duplicate pairs and sort by correlation coefficient
positive_correlation_features = positive_correlation_features.drop_duplicates(subset='Correlation', keep='first').sort_values(by='Correlation', ascending=False)

# Print the features with positive correlation
print("Features with Positive Correlation:")
print(positive_correlation_features)

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Age, Cholesterol Levels, and Maximum Heart Rate')
plt.show()

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Check data
print(X.shape, X_train.shape, X_test.shape)

# Train Logistic Regression Model
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, Y_train)

# Make predictions using Logistic Regression
logistic_regression_test_predictions = logistic_regression_model.predict(X_test)

# Calculate accuracy for Logistic Regression
logistic_regression_test_accuracy = accuracy_score(Y_test, logistic_regression_test_predictions)
print('Logistic Regression - Test Accuracy:', logistic_regression_test_accuracy)

# Train Support Vector Machine Model
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, Y_train)

# Make predictions using SVM
svm_test_predictions = svm_classifier.predict(X_test)

# Calculate accuracy for SVM
svm_test_accuracy = accuracy_score(Y_test, svm_test_predictions)
print('Support Vector Machine - Test Accuracy:', svm_test_accuracy)

# Train K Nearest Neighbors Model
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, Y_train)

# Make predictions using KNN
knn_test_predictions = knn_classifier.predict(X_test)

# Calculate accuracy for KNN
knn_test_accuracy = accuracy_score(Y_test, knn_test_predictions)
print('K Nearest Neighbors - Test Accuracy:', knn_test_accuracy)

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, Y_train)

# Make predictions using Random Forest
rf_test_predictions = rf_classifier.predict(X_test)

# Calculate accuracy for Random Forest
rf_test_accuracy = accuracy_score(Y_test, rf_test_predictions)
print('Random Forest - Test Accuracy:', rf_test_accuracy)

# Train XGBoost Classifier
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_train, Y_train)

# Make predictions using XGBoost
xgb_test_predictions = xgb_classifier.predict(X_test)

# Calculate accuracy for XGBoost
xgb_test_accuracy = accuracy_score(Y_test, xgb_test_predictions)
print('XGBoost - Test Accuracy:', xgb_test_accuracy)

# Compare accuracies and select the best model
accuracies = {
    'Logistic Regression': logistic_regression_test_accuracy,
    'Support Vector Machine': svm_test_accuracy,
    'K Nearest Neighbors': knn_test_accuracy,
    'Random Forest': rf_test_accuracy,
    'XGBoost': xgb_test_accuracy
}

best_model = max(accuracies, key=accuracies.get)
print(f'The best-performing model is {best_model} with an accuracy of {accuracies[best_model]}.')

# Assuming X_train and y_train are your training data
# Initialize and train the XGBoost model
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_train, Y_train)

# Define the input data
input_data = np.array([(50, 1, 0, 150, 243, 0, 0, 128, 0, 2.6, 1, 0, 3)])

# Reshape the input data as we are predicting for only one instance
input_data_reshaped = input_data.reshape(1, -1)

# Make predictions using the trained model
prediction = xgb_classifier.predict(input_data_reshaped)

# Print the prediction
if prediction[0] == 0:
    print('The person does not have heart disease.')
else:
    print('The person has heart disease.')
