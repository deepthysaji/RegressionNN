
# Step 1: Import necessary libraries
print("Loading libraries...")
print("- Loading pandas, numpy, os...")
import pandas as pd
import numpy as np
import os

print("- Loading matplotlib, seaborn...")
import matplotlib.pyplot as plt
import seaborn as sns

print("- Loading sklearn modules...")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

print("- Loading TensorFlow (this may take 10-15 seconds)...")
import tensorflow as tf
print("✓ All libraries loaded successfully!")

# %%
# ## Step 2: Load and Explore the Dataset
# We load the California Housing dataset from local data files.

# %%
# Create output directory for figures
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created: {output_dir}")

# %%
# Define the path to the data file
data_path = os.path.join('..', 'data', 'CaliforniaHousing', 'cal_housing.data')

# Define column names based on the domain file
column_names = [
    'longitude', 'latitude', 'housingMedianAge', 'totalRooms', 
    'totalBedrooms', 'population', 'households', 'medianIncome', 
    'medianHouseValue'
]

# Load the data from CSV file
df = pd.read_csv(data_path, names=column_names, header=None)

# Separate features and target variable
# Convert medianHouseValue from dollars to hundreds of thousands of dollars (to match sklearn format)
df['MedHouseVal'] = df['medianHouseValue'] / 100000.0
df = df.drop('medianHouseValue', axis=1)  # Remove the original target column

# %%
# Display the first few rows of the dataframe
print("Data Head:")
print(df.head())

# %%
# Get a statistical summary of the data
print("\nData Description:")
print(df.describe())

# %%
# Visualize the distribution of each feature to look for outliers and skewness
df.hist(bins=50, figsize=(20, 15))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 3: Preprocess the Data
# This involves splitting the data into training and testing sets, and scaling the features.
# Scaling is crucial for neural networks to ensure all features contribute fairly to the result.

# %%
# Separate features (X) from the target (y)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# %%
# Scale the features using StandardScaler
# This standardizes features by removing the mean and scaling to unit variance.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use the same scaler fitted on the training data

# %% [markdown]
# ## Step 4: Build the Neural Network Model
# We'll define a simple Sequential model with two hidden layers.

# %%
# Set a random seed for reproducibility
tf.random.set_seed(42)

# Define the model architecture
model = tf.keras.models.Sequential([
    # Input layer: The shape must match the number of features (8)
    tf.keras.layers.Dense(32, activation='relu', input_shape=[X_train.shape[1]], name='dense_1'),

    # Hidden layer 1
    tf.keras.layers.Dense(32, activation='relu', name='dense_2'),

    # Output layer: 1 neuron for the single output value (price)
    # No activation function is used for regression, which is equivalent to a linear activation.
    tf.keras.layers.Dense(1, name='output')
])

# %% [markdown]
# ## Step 5: Compile the Model
# We need to specify the optimizer, loss function, and any metrics to track.

# %%
# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error', # MSE is a standard loss function for regression
              metrics=['mae', 'mse']) # Track Mean Absolute Error and Mean Squared Error

# Print a summary of the model's architecture
model.summary()

# %% [markdown]
# ## Step 6: Train ("Fit") the Model
# Now we train the model on our scaled training data. We'll use the test set for validation
# to monitor performance on unseen data during training.

# %%
# Train the model
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=50, # An epoch is one full pass through the training data
    validation_data=(X_test_scaled, y_test),
    verbose=1 # Show progress
)

# %% [markdown]
# ## Step 7: Evaluate the Model
# Let's analyze the training history and evaluate the final model's performance on the test set.

# %%
# Plot the learning curves (loss over epochs)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Loss vs. Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.show()

# %%
# Evaluate the model on the test set
print("\n--- Model Evaluation on Test Set ---")
loss, mae, mse = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Set Mean Squared Error: {mse:.4f}")
print(f"Test Set Mean Absolute Error: {mae:.4f}")

# %%
# Make predictions on the test set
predictions = model.predict(X_test_scaled).flatten()

# Calculate R-squared score
r2 = r2_score(y_test, predictions)
print(f"Test Set R-squared: {r2:.4f}")

# Save evaluation metrics to file
metrics_file = os.path.join(output_dir, 'model_evaluation_metrics.txt')
with open(metrics_file, 'w') as f:
    f.write("=== Neural Network Model Evaluation Results ===\n")
    f.write(f"Dataset: California Housing Dataset\n")
    f.write(f"Model Architecture: Sequential Neural Network\n")
    f.write(f"Training Set Size: {X_train.shape[0]} samples\n")
    f.write(f"Test Set Size: {X_test.shape[0]} samples\n")
    f.write(f"Number of Features: {X_train.shape[1]}\n")
    f.write(f"Training Epochs: 50\n")
    f.write("\n--- Performance Metrics on Test Set ---\n")
    f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
    f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
    f.write(f"R-squared Score: {r2:.4f}\n")
    f.write(f"\n--- Model Summary ---\n")
    f.write(f"Final Training Loss: {history.history['loss'][-1]:.4f}\n")
    f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}\n")
    f.write(f"Final Training MAE: {history.history['mae'][-1]:.4f}\n")
    f.write(f"Final Validation MAE: {history.history['val_mae'][-1]:.4f}\n")

print(f"✓ Model evaluation metrics saved to: {metrics_file}")

# %%
# Visualize the predictions vs. the actual values
plt.figure(figsize=(8, 8))
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel("Actual Values (MedHouseVal)")
plt.ylabel("Predicted Values (MedHouseVal)")
plt.title("Actual vs. Predicted House Values")
# Plot a line for perfect predictions (y=x)
plt.plot([0, 5], [0, 5], color='red', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'predictions_vs_actual.png'), dpi=300, bbox_inches='tight')
plt.show()



