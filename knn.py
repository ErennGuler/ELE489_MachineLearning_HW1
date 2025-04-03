import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Column names for the dataset
column_names = [
    'Class', 'Alcohol', 'Malic acid', 'Ash', 'Alkalinity of ash',
    'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
    'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315', 'Proline'
]
# Loading the dataset into environment
wine_data = pd.read_csv('wine.csv', header=None, names=column_names)

# Normalization Process
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  #apply standartization
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)  # Converting back to DataFrame
scaled_wine_data = pd.concat([X_scaled, y], axis=1)

# I split data into training and test sets here. (%20 test and %80 training)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

#My k-NN Implementation
class KNN_Classifier:
    def __init__(self, k=5, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric.lower()

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Define distance metrics
    def calculate_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2)) # Compute Euclidean distance
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2)) # Compute Manhattan distance

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            distances = []
            for _, train_row in self.X_train.iterrows():
                dist = self.calculate_distance(row.values, train_row.values)
                distances.append(dist)

            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train.iloc[k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)


# Try different k values for Euclidian and Manhattan distance metrics
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
metrics = ['euclidean', 'manhattan']
results = {'Euclidean': [], 'Manhattan': []}

#Output graph of Accuracy vs different k values
plt.figure(num=9, figsize=(10, 6), clear=True)
plt.title('k-NN Classification Accuracy vs Number of Neighbors')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)

for metric in metrics:
    accuracies = []
    for k in k_values:
        knn = KNN_Classifier(k=k, distance_metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        results[metric.capitalize()].append(accuracy)

    # Plot accuracy curve
    plt.plot(k_values, accuracies,
             marker='o' if metric == 'euclidean' else 's',
             label=f'{metric.capitalize()} Distance',
             linestyle='-')

plt.legend()
plt.show()

# Print results table
print("\nAccuracy Results:")
print(pd.DataFrame(results, index=k_values).round(4))

# Show classification reports and confusion matrices for all k values
for k in k_values:
    for metric in metrics:
        knn = KNN_Classifier(k=k, distance_metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        print(f"\n--- k={k}, {metric} distance ---")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        plt.figure()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d',
                    cmap='Blues' if metric == 'euclidean' else 'Reds',
                    xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
        plt.title(f'Confusion Matrix (k={k}, {metric.capitalize()})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
