import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn import preprocessing


def process_csv(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    label_encoder = preprocessing.LabelEncoder()

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])

    # Perform data preprocessing and split into features and target
    X = df.drop(columns=['Age'])
    y = df['Age']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform Linear Regression
    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    linear_regression_predictions = linear_regression.predict(X_test)

    # Perform Logistic Regression
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    logistic_regression_predictions = logistic_regression.predict(X_test)

    # Perform Decision Tree
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)
    decision_tree_predictions = decision_tree.predict(X_test)

    # Perform Support Vector Machine
    svm = SVC()
    svm.fit(X_train, y_train)
    svm_predictions = svm.predict(X_test)

    # Perform Random Forest
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)
    random_forest_predictions = random_forest.predict(X_test)

    # Compute Confusion Matrix
    confusion_matrix_lr = confusion_matrix(y_test, logistic_regression_predictions)
    confusion_matrix_dt = confusion_matrix(y_test, decision_tree_predictions)
    confusion_matrix_svm = confusion_matrix(y_test, svm_predictions)
    confusion_matrix_rf = confusion_matrix(y_test, random_forest_predictions)

    # Visualize Confusion Matrix
    cm_visualizations = []
    cm_visualizations.append(save_confusion_matrix(confusion_matrix_lr, "Logistic Regression"))
    cm_visualizations.append(save_confusion_matrix(confusion_matrix_dt, "Decision Tree"))
    cm_visualizations.append(save_confusion_matrix(confusion_matrix_svm, "Support Vector Machine"))
    cm_visualizations.append(save_confusion_matrix(confusion_matrix_rf, "Random Forest"))

    # Return the computed results and visualization paths
    results = {
        "Linear Regression Predictions": linear_regression_predictions.tolist(),
        "Logistic Regression Predictions": logistic_regression_predictions.tolist(),
        "Decision Tree Predictions": decision_tree_predictions.tolist(),
        "Support Vector Machine Predictions": svm_predictions.tolist(),
        "Random Forest Predictions": random_forest_predictions.tolist(),
        "Confusion Matrix Visualizations": cm_visualizations
    }
    return results


def save_confusion_matrix(matrix, title):
    plt.figure(figsize=(5, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Save the visualization as an image in a temporary directory
    temp_dir = "C:\\Users\\User\\Documents\\temp_image"  # Update with your desired temporary directory
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    image_path = os.path.join(temp_dir, title.lower().replace(" ", "_") + ".png")
    plt.savefig(image_path)
    plt.close()

    return image_path


if __name__ == '__main__':
    file_path = sys.argv[1]
    results = process_csv(file_path)

    # Convert the results dictionary to JSON
    json_results = json.dumps(results)

    # Print the JSON response
    print(json_results)