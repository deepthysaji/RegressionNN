# Safe Model Improvements - Preventing Overfitting
# Advanced techniques that focus on generalization

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import warnings
warnings.filterwarnings('ignore')

print("✓ Libraries loaded for safe model improvements!")

# Create output directory
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Load and prepare data
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
# SAFE IMPROVEMENT 1: Advanced Feature Engineering (without overfitting)
print("=== Safe Feature Engineering ===")

# Original engineered features
df['rooms_per_household'] = df['totalRooms'] / df['households']
df['bedrooms_per_room'] = df['totalBedrooms'] / df['totalRooms']
df['population_per_household'] = df['population'] / df['households']

# Additional safe features (based on domain knowledge)
df['income_per_person'] = df['medianIncome'] / df['population_per_household']
df['bedroom_ratio'] = df['totalBedrooms'] / df['totalRooms']
df['location_cluster'] = np.sqrt(df['longitude']**2 + df['latitude']**2)

# Clean data
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.median())

print(f"Dataset shape after feature engineering: {df.shape}")

# %%
# SAFE IMPROVEMENT 2: Cross-validation for robust evaluation
print("\n=== Cross-Validation Setup ===")

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split into train/test first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split training into train/validation for unbiased evaluation
X_train_cv, X_val, y_train_cv, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_cv)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Train set: {X_train_scaled.shape}")
print(f"Validation set: {X_val_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")

# %%
# SAFE IMPROVEMENT 3: Conservative model with strong regularization
def create_safe_model(input_shape, l1_reg=0.001, l2_reg=0.001, dropout_rate=0.4):
    """
    Create a conservative model that prioritizes generalization
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        
        # Layer 1 - Conservative size with strong regularization
        tf.keras.layers.Dense(64, activation='relu', 
                             kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        
        # Layer 2 - Smaller to prevent overfitting
        tf.keras.layers.Dense(32, activation='relu',
                             kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        
        # Layer 3 - Even smaller
        tf.keras.layers.Dense(16, activation='relu',
                             kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        tf.keras.layers.Dropout(dropout_rate/2),
        
        # Output
        tf.keras.layers.Dense(1)
    ])
    
    return model

# %%
# SAFE IMPROVEMENT 4: Multiple validation strategies
print("\n=== Training Safe Models ===")

# Model 1: Conservative with high regularization
tf.random.set_seed(42)
model_safe = create_safe_model(X_train_scaled.shape[1], l1_reg=0.002, l2_reg=0.002, dropout_rate=0.5)

model_safe.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Lower learning rate
    loss='huber',
    metrics=['mae', 'mse']
)

# Very conservative training with early stopping
early_stop_conservative = EarlyStopping(
    monitor='val_loss',
    patience=20,  # More patience
    restore_best_weights=True,
    verbose=1
)

print("Training conservative model...")
history_safe = model_safe.fit(
    X_train_scaled, y_train_cv,
    validation_data=(X_val_scaled, y_val),
    epochs=150,
    batch_size=64,  # Larger batch size for stability
    callbacks=[early_stop_conservative],
    verbose=1
)

# %%
# SAFE IMPROVEMENT 5: Model averaging with different architectures
print("\n=== Training Diverse Ensemble ===")

def create_diverse_models(input_shape):
    """Create models with different architectures for ensemble"""
    
    # Model A: Wide and shallow
    model_a = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1)
    ])
    
    # Model B: Narrow and deep
    model_b = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    
    # Model C: Medium with different activation
    model_c = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='elu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation='elu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1)
    ])
    
    return [model_a, model_b, model_c]

# Create diverse ensemble
diverse_models = create_diverse_models(X_train_scaled.shape[1])
diverse_predictions = []

for i, model in enumerate(diverse_models):
    print(f"Training diverse model {i+1}/3...")
    
    model.compile(optimizer='adam', loss='huber', metrics=['mae'])
    
    tf.random.set_seed(42 + i)
    history = model.fit(
        X_train_scaled, y_train_cv,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
        verbose=0
    )
    
    pred = model.predict(X_test_scaled, verbose=0).flatten()
    diverse_predictions.append(pred)

# Average diverse ensemble
diverse_ensemble_pred = np.mean(diverse_predictions, axis=0)

# %%
# SAFE IMPROVEMENT 6: Compare with traditional ML (often more robust)
print("\n=== Comparing with Random Forest ===")

# Random Forest is often more robust and less prone to overfitting
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,  # Limit depth to prevent overfitting
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

# Scale back to original for Random Forest (doesn't need scaling)
rf_model.fit(X_train_cv, y_train_cv)
rf_pred = rf_model.predict(X_test)

# %%
# Evaluate all approaches
print("\n=== Safe Model Comparison ===")

# Safe neural network
safe_pred = model_safe.predict(X_test_scaled, verbose=0).flatten()
r2_safe = r2_score(y_test, safe_pred)
mae_safe = mean_absolute_error(y_test, safe_pred)

# Diverse ensemble
r2_diverse = r2_score(y_test, diverse_ensemble_pred)
mae_diverse = mean_absolute_error(y_test, diverse_ensemble_pred)

# Random Forest
r2_rf = r2_score(y_test, rf_pred)
mae_rf = mean_absolute_error(y_test, rf_pred)

print(f"Safe Neural Network:")
print(f"  R-squared: {r2_safe:.4f}")
print(f"  MAE: {mae_safe:.4f}")

print(f"\nDiverse Ensemble:")
print(f"  R-squared: {r2_diverse:.4f}")
print(f"  MAE: {mae_diverse:.4f}")

print(f"\nRandom Forest:")
print(f"  R-squared: {r2_rf:.4f}")
print(f"  MAE: {mae_rf:.4f}")

# %%
# Check for overfitting in safe model
print("\n=== Overfitting Check for Safe Model ===")
final_train_loss = history_safe.history['loss'][-1]
final_val_loss = history_safe.history['val_loss'][-1]
gap = final_val_loss - final_train_loss

print(f"Training Loss: {final_train_loss:.4f}")
print(f"Validation Loss: {final_val_loss:.4f}")
print(f"Gap: {gap:.4f}")

if gap < 0.02:
    print("✅ Excellent generalization - very safe model")
elif gap < 0.05:
    print("✅ Good generalization - safe to continue")
else:
    print("⚠️ Still some overfitting - need more regularization")

# %%
# Save comprehensive safe results
safe_results_file = os.path.join(output_dir, 'safe_model_comparison.txt')
with open(safe_results_file, 'w') as f:
    f.write("=== Safe Model Improvement Results ===\n")
    f.write("Focus: Prevent overfitting while improving performance\n\n")
    
    f.write("Baseline (Original):\n")
    f.write("  R-squared: 0.7815\n\n")
    
    f.write("Safe Neural Network:\n")
    f.write(f"  R-squared: {r2_safe:.4f}\n")
    f.write(f"  MAE: {mae_safe:.4f}\n")
    f.write(f"  Overfitting Gap: {gap:.4f}\n\n")
    
    f.write("Diverse Ensemble:\n")
    f.write(f"  R-squared: {r2_diverse:.4f}\n")
    f.write(f"  MAE: {mae_diverse:.4f}\n\n")
    
    f.write("Random Forest:\n")
    f.write(f"  R-squared: {r2_rf:.4f}\n")
    f.write(f"  MAE: {mae_rf:.4f}\n\n")
    
    f.write("Safe Improvement Techniques Used:\n")
    f.write("- Strong L1+L2 regularization\n")
    f.write("- High dropout rates (up to 50%)\n")
    f.write("- Conservative learning rate\n")
    f.write("- Early stopping with patience\n")
    f.write("- Diverse model architectures\n")
    f.write("- Cross-validation splitting\n")
    f.write("- Comparison with traditional ML\n")

print(f"\n✓ Safe model results saved to: {safe_results_file}")

# %%
# Visualize safe improvements
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Safe model training history
axes[0,0].plot(history_safe.history['loss'], label='Training Loss', alpha=0.8)
axes[0,0].plot(history_safe.history['val_loss'], label='Validation Loss', alpha=0.8)
axes[0,0].set_title('Safe Model: Training vs Validation Loss')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].legend()
axes[0,0].grid(True)

# Plot 2: Model comparison
models = ['Original', 'Safe NN', 'Diverse Ensemble', 'Random Forest']
r2_scores = [0.7815, r2_safe, r2_diverse, r2_rf]
colors = ['red', 'blue', 'green', 'orange']

bars = axes[0,1].bar(models, r2_scores, color=colors, alpha=0.7)
axes[0,1].set_title('Model Performance Comparison')
axes[0,1].set_ylabel('R-squared Score')
axes[0,1].set_ylim(0.75, max(r2_scores) + 0.02)
axes[0,1].grid(True, axis='y')

# Add value labels on bars
for bar, score in zip(bars, r2_scores):
    axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                   f'{score:.4f}', ha='center', va='bottom')

# Plot 3: Predictions vs Actual (Best Safe Model)
best_safe_pred = diverse_ensemble_pred if r2_diverse > r2_safe else safe_pred
best_r2 = max(r2_diverse, r2_safe)

axes[1,0].scatter(y_test, best_safe_pred, alpha=0.6, color='green')
axes[1,0].plot([0, 5], [0, 5], 'r--', label='Perfect Prediction')
axes[1,0].set_xlabel('Actual Values')
axes[1,0].set_ylabel('Predicted Values')
axes[1,0].set_title(f'Best Safe Model (R² = {best_r2:.4f})')
axes[1,0].legend()
axes[1,0].grid(True)

# Plot 4: Feature importance from Random Forest
feature_names = X.columns
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]  # Top 10 features

axes[1,1].barh(range(len(indices)), importances[indices])
axes[1,1].set_yticks(range(len(indices)))
axes[1,1].set_yticklabels([feature_names[i] for i in indices])
axes[1,1].set_xlabel('Feature Importance')
axes[1,1].set_title('Top 10 Feature Importances (Random Forest)')
axes[1,1].grid(True, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'safe_model_analysis.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Recommendations for Further Safe Improvement ===")
print("1. ✅ Hyperparameter tuning with cross-validation")
print("2. ✅ More sophisticated ensemble methods (stacking)")
print("3. ✅ External data sources (economic indicators, geographic data)")
print("4. ✅ Advanced feature selection techniques")
print("5. ⚠️  Be cautious with: deeper networks, lower regularization")
print("6. 🚫 Avoid: Complex architectures without validation, data leakage")
