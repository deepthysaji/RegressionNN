# Improved Neural Network for California Housing Prediction
# This file contains various techniques to improve model performance

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

print("✓ Libraries loaded successfully!")

# Create output directory
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Load data (same as before)
data_path = os.path.join('..', 'data', 'CaliforniaHousing', 'cal_housing.data')
column_names = [
    'longitude', 'latitude', 'housingMedianAge', 'totalRooms', 
    'totalBedrooms', 'population', 'households', 'medianIncome', 
    'medianHouseValue'
]

df = pd.read_csv(data_path, names=column_names, header=None)
df['MedHouseVal'] = df['medianHouseValue'] / 100000.0
df = df.drop('medianHouseValue', axis=1)

# %%
# IMPROVEMENT 1: Feature Engineering
print("=== Feature Engineering ===")

# Create new features that might be more meaningful
df['rooms_per_household'] = df['totalRooms'] / df['households']
df['bedrooms_per_room'] = df['totalBedrooms'] / df['totalRooms']
df['population_per_household'] = df['population'] / df['households']

# Handle any potential division by zero or inf values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.median())

print(f"New features added. Dataset shape: {df.shape}")
print("New features:")
print("- rooms_per_household")
print("- bedrooms_per_room") 
print("- population_per_household")

# %%
# IMPROVEMENT 2: Better data preprocessing
print("\n=== Improved Data Preprocessing ===")

# Separate features and target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try RobustScaler instead of StandardScaler (better for outliers)
scaler = RobustScaler()  # Less sensitive to outliers than StandardScaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Using RobustScaler for better outlier handling")
print(f"Training set shape: {X_train_scaled.shape}")

# %%
# IMPROVEMENT 3: More sophisticated model architecture
def create_improved_model(input_shape, dropout_rate=0.3, l2_reg=0.001):
    """
    Create an improved neural network with:
    - More layers
    - Dropout for regularization
    - L2 regularization
    - Batch normalization
    """
    model = tf.keras.models.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu', 
                             kernel_regularizer=l2(l2_reg), name='dense_1'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        
        # Hidden layer 1
        tf.keras.layers.Dense(64, activation='relu', 
                             kernel_regularizer=l2(l2_reg), name='dense_2'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        
        # Hidden layer 2
        tf.keras.layers.Dense(32, activation='relu', 
                             kernel_regularizer=l2(l2_reg), name='dense_3'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        
        # Hidden layer 3
        tf.keras.layers.Dense(16, activation='relu', 
                             kernel_regularizer=l2(l2_reg), name='dense_4'),
        tf.keras.layers.Dropout(dropout_rate/2),
        
        # Output layer
        tf.keras.layers.Dense(1, name='output')
    ])
    
    return model

# %%
# IMPROVEMENT 4: Advanced training techniques
print("\n=== Training Improved Model ===")

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

# Create improved model
model_improved = create_improved_model(X_train_scaled.shape[1])

# Compile with different optimizer settings
model_improved.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
    loss='huber',  # Huber loss is more robust to outliers than MSE
    metrics=['mae', 'mse']
)

print("Model Architecture:")
model_improved.summary()

# %%
# IMPROVEMENT 5: Advanced callbacks
# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate when loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1
)

# Train the improved model
print("\nTraining improved model...")
history_improved = model_improved.fit(
    X_train_scaled,
    y_train,
    epochs=200,  # More epochs, but early stopping will prevent overfitting
    validation_data=(X_test_scaled, y_test),
    callbacks=[early_stopping, reduce_lr],
    batch_size=32,  # Smaller batch size for better gradient estimates
    verbose=1
)

# %%
# IMPROVEMENT 6: Ensemble approach (simple)
def create_simple_model(input_shape):
    """Create a simpler model for ensemble"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),  # Use Input layer instead
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    return model

# Train multiple models for ensemble
print("\n=== Training Ensemble Models ===")
ensemble_models = []
ensemble_predictions = []

for i in range(3):
    print(f"Training ensemble model {i+1}/3...")
    
    # Create and compile model
    model_ens = create_simple_model(X_train_scaled.shape[1])
    model_ens.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train with different random seeds
    tf.random.set_seed(42 + i)
    history_ens = model_ens.fit(
        X_train_scaled, y_train,
        epochs=100,
        validation_data=(X_test_scaled, y_test),
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0
    )
    
    # Make predictions
    pred_ens = model_ens.predict(X_test_scaled, verbose=0).flatten()
    ensemble_predictions.append(pred_ens)
    ensemble_models.append(model_ens)

# Average ensemble predictions
ensemble_pred = np.mean(ensemble_predictions, axis=0)

# %%
# Evaluate all models
print("\n=== Model Comparison ===")

# Original model predictions (you'd need to run this after training your original model)
# For now, let's use the improved model
predictions_improved = model_improved.predict(X_test_scaled, verbose=0).flatten()

# Calculate metrics for improved model
r2_improved = r2_score(y_test, predictions_improved)
mae_improved = mean_absolute_error(y_test, predictions_improved)
mse_improved = mean_squared_error(y_test, predictions_improved)

# Calculate metrics for ensemble
r2_ensemble = r2_score(y_test, ensemble_pred)
mae_ensemble = mean_absolute_error(y_test, ensemble_pred)
mse_ensemble = mean_squared_error(y_test, ensemble_pred)

print("Improved Single Model:")
print(f"  R-squared: {r2_improved:.4f}")
print(f"  MAE: {mae_improved:.4f}")
print(f"  MSE: {mse_improved:.4f}")

print("\nEnsemble Model:")
print(f"  R-squared: {r2_ensemble:.4f}")
print(f"  MAE: {mae_ensemble:.4f}")
print(f"  MSE: {mse_ensemble:.4f}")

# %%
# ANALYSIS: Check for overfitting signs
print("\n=== Overfitting Analysis ===")
final_train_loss = history_improved.history['loss'][-1]
final_val_loss = history_improved.history['val_loss'][-1]
gap = final_val_loss - final_train_loss

print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")
print(f"Train-Val Gap: {gap:.4f}")

if gap > 0.1:
    print("⚠️  OVERFITTING DETECTED - Large gap between train and validation loss")
    print("   Recommendations: More regularization, less complex model, more data")
elif gap > 0.05:
    print("⚠️  MILD OVERFITTING - Some overfitting present")
    print("   Recommendations: Slight increase in regularization")
else:
    print("✅ GOOD GENERALIZATION - Model is not overfitting")
    print("   Safe to try further improvements")

# Check if validation loss is still decreasing
if len(history_improved.history['val_loss']) > 10:
    recent_val_losses = history_improved.history['val_loss'][-10:]
    if recent_val_losses[-1] < recent_val_losses[0]:
        print("📈 Validation loss still decreasing - could train longer")
    else:
        print("📉 Validation loss plateaued - current training length is appropriate")

# Compare with original baseline
print(f"\n=== Improvement Summary ===")
baseline_r2 = 0.7815  # Your original model
improvement = r2_improved - baseline_r2
ensemble_improvement = r2_ensemble - baseline_r2

print(f"Original Model R²: {baseline_r2:.4f}")
print(f"Improved Model R²: {r2_improved:.4f} (+{improvement:.4f})")
print(f"Ensemble Model R²: {r2_ensemble:.4f} (+{ensemble_improvement:.4f})")

if r2_improved > 0.85:
    print("🎯 EXCELLENT performance - diminishing returns likely")
elif r2_improved > 0.82:
    print("🎯 VERY GOOD performance - still room for improvement")
elif r2_improved > baseline_r2:
    print("🎯 GOOD improvement - significant room for further gains")
else:
    print("⚠️  No improvement - need different approach")

# %%
# Save comprehensive results
results_file = os.path.join(output_dir, 'improved_model_results.txt')
with open(results_file, 'w') as f:
    f.write("=== Improved Neural Network Model Results ===\n")
    f.write(f"Dataset: California Housing Dataset\n")
    f.write(f"Improvements Applied:\n")
    f.write("  - Feature Engineering (3 new features)\n")
    f.write("  - RobustScaler for better outlier handling\n")
    f.write("  - Deeper architecture (4 hidden layers: 128-64-32-16)\n")
    f.write("  - Regularization (Dropout + L2)\n")
    f.write("  - Batch Normalization\n")
    f.write("  - Huber loss (robust to outliers)\n")
    f.write("  - Advanced callbacks (EarlyStopping, ReduceLROnPlateau)\n")
    f.write("  - Ensemble of 3 models\n")
    f.write("\n--- Results Comparison ---\n")
    f.write("Improved Single Model:\n")
    f.write(f"  R-squared: {r2_improved:.4f}\n")
    f.write(f"  MAE: {mae_improved:.4f}\n")
    f.write(f"  MSE: {mse_improved:.4f}\n")
    f.write("\nEnsemble Model:\n")
    f.write(f"  R-squared: {r2_ensemble:.4f}\n")
    f.write(f"  MAE: {mae_ensemble:.4f}\n")
    f.write(f"  MSE: {mse_ensemble:.4f}\n")

print(f"\n✓ Detailed results saved to: {results_file}")

# %%
# Visualize improvements
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Training history comparison
if len(history_improved.history['loss']) > 0:
    axes[0,0].plot(history_improved.history['loss'], label='Training Loss', alpha=0.8)
    axes[0,0].plot(history_improved.history['val_loss'], label='Validation Loss', alpha=0.8)
    axes[0,0].set_title('Improved Model Training History')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)

# Plot 2: Predictions vs Actual (Improved Model)
axes[0,1].scatter(y_test, predictions_improved, alpha=0.6, color='blue', label='Improved Model')
axes[0,1].plot([0, 5], [0, 5], 'r--', label='Perfect Prediction')
axes[0,1].set_xlabel('Actual Values')
axes[0,1].set_ylabel('Predicted Values')
axes[0,1].set_title(f'Improved Model (R² = {r2_improved:.4f})')
axes[0,1].legend()
axes[0,1].grid(True)

# Plot 3: Predictions vs Actual (Ensemble)
axes[1,0].scatter(y_test, ensemble_pred, alpha=0.6, color='green', label='Ensemble Model')
axes[1,0].plot([0, 5], [0, 5], 'r--', label='Perfect Prediction')
axes[1,0].set_xlabel('Actual Values')
axes[1,0].set_ylabel('Predicted Values')
axes[1,0].set_title(f'Ensemble Model (R² = {r2_ensemble:.4f})')
axes[1,0].legend()
axes[1,0].grid(True)

# Plot 4: Feature importance (correlation with target)
feature_importance = X.corrwith(y).abs().sort_values(ascending=False)
axes[1,1].barh(range(len(feature_importance)), feature_importance.values)
axes[1,1].set_yticks(range(len(feature_importance)))
axes[1,1].set_yticklabels(feature_importance.index)
axes[1,1].set_xlabel('Absolute Correlation with Target')
axes[1,1].set_title('Feature Importance')
axes[1,1].grid(True, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_improvements_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Next Steps for Further Improvement ===")
print("1. Hyperparameter tuning (learning rate, architecture, regularization)")
print("2. Cross-validation for more robust evaluation")
print("3. Try different algorithms (XGBoost, Random Forest)")
print("4. More advanced feature engineering")
print("5. Collect more data if possible")
