# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv('Movies.csv')

# Clean column names by stripping leading/trailing spaces
data.columns = data.columns.str.strip()

# Display dataset columns
print("Columns in dataset:", data.columns)

# Drop rows with missing values in key columns
features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']  
data = data.dropna(subset=features + ['Rating', 'Duration', 'Votes', 'Year'])

# Convert categorical columns to numeric using Label Encoding
label_encoder = LabelEncoder()
for feature in features:
    data[feature] = label_encoder.fit_transform(data[feature].astype(str))

# Ensure all columns are numeric before computing correlation
numeric_data = data.select_dtypes(include=[np.number])

# 1. Correlation Heatmap**
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 2. Genre Count Plot**
plt.figure(figsize=(10, 6))
sns.countplot(y=data['Genre'], order=data['Genre'].value_counts().index, palette='coolwarm')
plt.title('Number of Movies per Genre')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()

# 3. Top 10 Most Common Directors**
plt.figure(figsize=(10, 6))
top_directors = data['Director'].value_counts().nlargest(10)
sns.barplot(x=top_directors.values, y=top_directors.index, palette='viridis')
plt.title('Top 10 Most Common Directors')
plt.xlabel('Number of Movies')
plt.ylabel('Director')
plt.show()

# 4. Rating Distribution (Seaborn Histogram)**
plt.figure(figsize=(8, 5))
sns.histplot(data['Rating'], bins=20, kde=True, color='blue')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# 5. True vs. Predicted Ratings Scatter Plot**
# Prepare data for model training
X = data[features]  
y = data['Rating']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='red')
plt.xlabel('True Ratings')
plt.ylabel('Predicted Ratings')
plt.title('True vs. Predicted Ratings')
plt.show()

# 6. Votes vs. Rating Scatter Plot**
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['Votes'], y=data['Rating'], alpha=0.5)
plt.xscale('log')  # Log scale for better visualization
plt.xlabel('Number of Votes (log scale)')
plt.ylabel('Rating')
plt.title('Votes vs. Rating')
plt.show()

# 7. Movie Duration vs. Rating Scatter Plot**
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['Duration'], y=data['Rating'], alpha=0.5, color='purple')
plt.xlabel('Movie Duration (minutes)')
plt.ylabel('Rating')
plt.title('Movie Duration vs. Rating')
plt.show()

# 8. Top 10 Most Frequent Actors**
actors = pd.concat([data['Actor 1'], data['Actor 2'], data['Actor 3']])
top_actors = actors.value_counts().nlargest(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_actors.values, y=top_actors.index, palette='magma')
plt.title('Top 10 Most Frequent Actors')
plt.xlabel('Number of Movies')
plt.ylabel('Actor')
plt.show()

# 9. Area Plot: Movies Released Over Time**
plt.figure(figsize=(12, 6))
movies_per_year = data['Year'].value_counts().sort_index()
movies_per_year.plot(kind='area', color='skyblue', alpha=0.6)
plt.title('Number of Movies Released Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.show()

# 10. Histogram of Ratings (Matplotlib)**
plt.figure(figsize=(8, 5))
plt.hist(data['Rating'], bins=20, color='green', alpha=0.7, edgecolor='black')
plt.title('Histogram of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Model Performance Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")
