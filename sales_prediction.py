import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Define the correct file path
file_path = "/Users/vijayreddy/Downloads/advertising.csv"

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file at {file_path} was not found. Please check the path.")

# Load the dataset
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Dataset preview:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Drop rows with missing values (if any)
df = df.dropna()

# Visualize pairwise relationships
sns.pairplot(df)
plt.show()

# Define features (X) and target variable (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nModel Evaluation:')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# 1. Sales Distribution (Histogram)
plt.figure(figsize=(8, 5))
sns.histplot(df['Sales'], bins=20, kde=True, color='blue')
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# 3. Sales vs. Advertising Channels (Line Plot)
plt.figure(figsize=(10, 6))
sns.lineplot(x=df['TV'], y=df['Sales'], label='TV', color='red')
sns.lineplot(x=df['Radio'], y=df['Sales'], label='Radio', color='blue')
sns.lineplot(x=df['Newspaper'], y=df['Sales'], label='Newspaper', color='green')
plt.title('Sales Trend Over TV, Radio & Newspaper Ads')
plt.xlabel('Ad Spending')
plt.ylabel('Sales')
plt.legend()
plt.show()

# 4. Sales Distribution by Feature (Violin Plot)
plt.figure(figsize=(15, 5))
for i, feature in enumerate(['TV', 'Radio', 'Newspaper']):
    plt.subplot(1, 3, i+1)
    sns.violinplot(x=df[feature], y=df['Sales'])
    plt.title(f'Sales vs {feature}')
plt.tight_layout()
plt.show()

# 5. Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Actual Sales')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# 6. Feature Importance (Bar Chart of Coefficients)
coefficients = pd.Series(model.coef_, index=X.columns)
plt.figure(figsize=(8, 5))
coefficients.plot(kind='barh', color='green')
plt.title('Feature Importance')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()

# 7. 3D Scatter Plot (TV, Radio vs. Sales)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['TV'], df['Radio'], df['Sales'], color='purple')
ax.set_xlabel('TV')
ax.set_ylabel('Radio')
ax.set_zlabel('Sales')
ax.set_title('3D Scatter Plot: TV & Radio vs. Sales')
plt.show()

# 8. QQ Plot (Normality of Residuals)
sm.qqplot(residuals, line='45')
plt.title('QQ Plot of Residuals')
plt.show()

# 9. Actual vs. Predicted Sales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, label='Predictions', color='purple')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ideal Fit')
plt.legend()
plt.show()
