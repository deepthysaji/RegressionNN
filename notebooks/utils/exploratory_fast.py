# Fast version for data exploration without TensorFlow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os

print("Starting fast data exploration...")

# Load the data
data_path = os.path.join('..', 'data', 'CaliforniaHousing', 'cal_housing.data')

column_names = [
    'longitude', 'latitude', 'housingMedianAge', 'totalRooms', 
    'totalBedrooms', 'population', 'households', 'medianIncome', 
    'medianHouseValue'
]

# Load the data from CSV file
df = pd.read_csv(data_path, names=column_names, header=None)

# Convert target variable
df['MedHouseVal'] = df['medianHouseValue'] / 100000.0
df = df.drop('medianHouseValue', axis=1)

print("Data Head:")
print(df.head())

print("\nData Description:")
print(df.describe())

print("\nData shape:", df.shape)
print("Missing values:")
print(df.isnull().sum())

# Basic preprocessing
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData preprocessing completed successfully!")
print("Ready for model training...")

# Optional: Create visualizations
try:
    print("\nCreating visualizations...")
    plt.figure(figsize=(12, 8))
    df.hist(bins=30, figsize=(15, 10))
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("Visualizations saved!")
except Exception as e:
    print(f"Visualization error: {e}")
