import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('Titanic.csv')

# Display the first few rows of the dataset
print(data.head())

# Handle missing values (example: fill or drop missing values)
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop(columns=['Cabin'], inplace=True)  # Drop the 'Cabin' column (too many missing values)

# Encode categorical features
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Drop unnecessary columns
data.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Line Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Age', y='Fare', hue='Survived')
plt.title('Line Plot: Age vs Fare by Survival')
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Age', y='Fare', hue='Survived')
plt.title('Scatter Plot: Age vs Fare by Survival')
plt.show()

# Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='Pclass', y='Survived', hue='Sex')
plt.title('Bar Plot: Survival by Class and Gender')
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
data['Age'].hist(bins=20, edgecolor='black')
plt.title('Histogram of Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Pie Chart
plt.figure(figsize=(8, 8))
data['Survived'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Did not survive', 'Survived'], colors=['red', 'green'])
plt.title('Pie Chart: Survival Distribution')
plt.ylabel('')
plt.show()

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Survived', y='Age', hue='Sex')
plt.title('Box Plot: Age by Survival and Gender')
plt.show()

# Area Plot
plt.figure(figsize=(10, 6))
data.groupby('Age')['Survived'].sum().plot.area(alpha=0.5)
plt.title('Area Plot: Survival by Age')
plt.show()

# 3D Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['Pclass'], data['Fare'], data['Age'], c=data['Survived'], cmap='viridis')
ax.set_xlabel('Pclass')
ax.set_ylabel('Fare')
ax.set_zlabel('Age')
plt.title('3D Plot: Survival by Pclass, Fare, and Age')
plt.show()

# Summary Plot: Pair Plot
sns.pairplot(data, hue='Survived', diag_kind='hist')
plt.show()

# Split data into features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
