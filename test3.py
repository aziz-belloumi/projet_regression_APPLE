"""
changement de nombre de de nombre de neuronne dans chaque couche
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping , ReduceLROnPlateau
import random
import tensorflow as tf


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# SET THE SEED HERE - Use the same seed to get consistent results
set_seeds(42)


# ============================================
# 1. LOAD DATA AND FILTER TO RECENT YEARS
# ============================================
print("="*60)
print("1. LOADING DATA (RECENT YEARS ONLY)")
print("="*60)

df = pd.read_csv('AAPL_2000_2024.csv', skiprows=[0,1])
df = df.rename(columns={
    'Date': 'date',
    'Unnamed: 1': 'close',
    'Unnamed: 2': 'high',
    'Unnamed: 3': 'low',
    'Unnamed: 4': 'open',
    'Unnamed: 5': 'volume'
})

df['date'] = pd.to_datetime(df['date'])

print(f"Full dataset: {df.shape}")
print(f"Full date range: {df['date'].min()} to {df['date'].max()}")

# ============================================
# CRITICAL FIX: USE ONLY LAST 5 YEARS
# ============================================
cutoff_date = pd.to_datetime('2019-01-01')
df = df[df['date'] >= cutoff_date].reset_index(drop=True)

print(f"\n✅ FILTERED TO RECENT DATA:")
print(f"Filtered dataset: {df.shape}")
print(f"Filtered date range: {df['date'].min()} to {df['date'].max()}")

# ============================================
# 2. SELECT FEATURES
# ============================================
print("\n" + "="*60)
print("2. FEATURE SELECTION")
print("="*60)

features = ['close', 'open', 'high', 'low', 'volume']
data = df[features].values

print(f"Selected features: {features}")
print(f"Feature array shape: {data.shape}")

print(f"\nPrice statistics (recent data):")
print(f"  Min: ${data[:, 0].min():.2f}")
print(f"  Max: ${data[:, 0].max():.2f}")
print(f"  Mean: ${data[:, 0].mean():.2f}")
print(f"  ✅ Much more consistent price range!")

# ============================================
# 3. SPLIT DATA (70% train, 15% val, 15% test)
# ============================================
print("\n" + "="*60)
print("3. SPLITTING DATA")
print("="*60)

n_total = len(data)
train_ratio = 0.7
val_ratio = 0.15

train_size = int(n_total * train_ratio)
val_size = int(n_total * val_ratio)
test_size = n_total - train_size - val_size

train_data = data[:train_size]
val_data = data[train_size:train_size+val_size]
test_data = data[train_size+val_size:]

train_dates = df['date'].iloc[:train_size]
val_dates = df['date'].iloc[train_size:train_size+val_size]
test_dates = df['date'].iloc[train_size+val_size:]

print(f"Total samples: {n_total}")
print(f"Train: {train_size} samples (70%)")
print(f"Val:   {val_size} samples (15%)")
print(f"Test:  {test_size} samples (15%)")

print(f"\nDate ranges:")
print(f"  Train: {train_dates.min()} to {train_dates.max()}")
print(f"  Val:   {val_dates.min()} to {val_dates.max()}")
print(f"  Test:  {test_dates.min()} to {test_dates.max()}")

# ============================================
# 4. SCALE DATA
# ============================================
print("\n" + "="*60)
print("4. SCALING DATA")
print("="*60)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data)

train_scaled = scaler.transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)

os.makedirs('./models', exist_ok=True)
joblib.dump(scaler, './models/scaler_recent.gz')
print("✓ Scaler fitted on recent training data")

# ============================================
# 5. CREATE SEQUENCES
# ============================================
print("\n" + "="*60)
print("5. CREATING SEQUENCES")
print("="*60)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

SEQUENCE_LENGTH = 60

X_train, y_train = create_sequences(train_scaled, SEQUENCE_LENGTH)

train_tail = train_scaled[-SEQUENCE_LENGTH:]
val_with_context = np.vstack([train_tail, val_scaled])
X_val, y_val = create_sequences(val_with_context, SEQUENCE_LENGTH)

val_tail = val_scaled[-SEQUENCE_LENGTH:]
test_with_context = np.vstack([val_tail, test_scaled])
X_test, y_test = create_sequences(test_with_context, SEQUENCE_LENGTH)

print(f"✓ Train: X={X_train.shape}, y={y_train.shape}")
print(f"✓ Val:   X={X_val.shape}, y={y_val.shape}")
print(f"✓ Test:  X={X_test.shape}, y={y_test.shape}")

# ============================================
# 6. BUILD MODEL
# ============================================
print("\n" + "="*60)
print("6. BUILDING MODEL")
print("="*60)

model = Sequential([
    LSTM(200, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(features))),
    Dropout(0.3),
    
    LSTM(150, return_sequences=True),
    Dropout(0.3),
    
    LSTM(100, return_sequences=True),
    Dropout(0.3),
    
    LSTM(100, return_sequences=False),
    Dropout(0.3),
    
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("✓ Model built")
model.summary()

# ============================================
# 7. TRAIN
# ============================================
print("\n" + "="*60)
print("7. TRAINING")
print("="*60)

checkpoint = ModelCheckpoint(
    './models/best_model_recent.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=0.0001,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=64,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# ============================================
# 8. PREDICT
# ============================================
print("\n" + "="*60)
print("8. PREDICTIONS")
print("="*60)

best_model = load_model('./models/best_model_recent.keras')

y_train_pred_scaled = best_model.predict(X_train, verbose=0)
y_val_pred_scaled = best_model.predict(X_val, verbose=0)
y_test_pred_scaled = best_model.predict(X_test, verbose=0)

# ============================================
# 9. INVERSE TRANSFORM
# ============================================
def inverse_transform_predictions(y_scaled, scaler, n_features):
    n = len(y_scaled)
    dummy = np.ones((n, n_features))
    dummy[:, 0] = y_scaled.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

y_train_actual = inverse_transform_predictions(y_train.reshape(-1, 1), scaler, len(features))
y_train_pred = inverse_transform_predictions(y_train_pred_scaled, scaler, len(features))

y_val_actual = inverse_transform_predictions(y_val.reshape(-1, 1), scaler, len(features))
y_val_pred = inverse_transform_predictions(y_val_pred_scaled, scaler, len(features))

y_test_actual = inverse_transform_predictions(y_test.reshape(-1, 1), scaler, len(features))
y_test_pred = inverse_transform_predictions(y_test_pred_scaled, scaler, len(features))

# ============================================
# 10. METRICS
# ============================================
print("\n" + "="*60)
print("10. RESULTS")
print("="*60)

def calc_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

mae_train, rmse_train, mape_train = calc_metrics(y_train_actual, y_train_pred)
mae_val, rmse_val, mape_val = calc_metrics(y_val_actual, y_val_pred)
mae_test, rmse_test, mape_test = calc_metrics(y_test_actual, y_test_pred)

print("\n" + "="*60)
print("📊 FINAL RESULTS (RECENT DATA)")
print("="*60)
print(f"{'Dataset':<10} {'MAE':<12} {'RMSE':<12} {'MAPE':<12}")
print("-"*60)
print(f"{'Train':<10} ${mae_train:<11.2f} ${rmse_train:<11.2f} {mape_train:<11.2f}%")
print(f"{'Val':<10} ${mae_val:<11.2f} ${rmse_val:<11.2f} {mape_val:<11.2f}%")
print(f"{'Test':<10} ${mae_test:<11.2f} ${rmse_test:<11.2f} {mape_test:<11.2f}%")
print("="*60)

if mape_test < 5:
    print("✅ EXCELLENT!")
elif mape_test < 10:
    print("✅ GOOD!")
else:
    print("⚠️  Still needs work, but much better than before!")

# ============================================
# 11. VISUALIZATIONS
# ============================================
print("\n" + "="*60)
print("11. VISUALIZATIONS")
print("="*60)

train_dates_plot = train_dates.iloc[SEQUENCE_LENGTH:].values
val_dates_plot = val_dates.values
test_dates_plot = test_dates.values

# Full plot
plt.figure(figsize=(18, 6))

plt.plot(train_dates_plot, y_train_actual, 'b-', alpha=0.6, linewidth=1, label='Train Actual')
plt.plot(train_dates_plot, y_train_pred, 'c-', alpha=0.8, linewidth=1, label='Train Pred')

plt.plot(val_dates_plot, y_val_actual, 'orange', linewidth=2, label='Val Actual')
plt.plot(val_dates_plot, y_val_pred, 'coral', linewidth=1, label='Val Pred')

plt.plot(test_dates_plot, y_test_actual, 'green', linewidth=2, label='Test Actual')
plt.plot(test_dates_plot, y_test_pred, 'limegreen', linewidth=2, label='Test Pred')

plt.title('Apple Stock Prediction - Recent Data (2019-2024)', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Test zoom
plt.figure(figsize=(14, 5))
recent = min(100, len(test_dates_plot))
plt.plot(test_dates_plot[-recent:], y_test_actual[-recent:], 
         'g-', linewidth=3, marker='o', markersize=3, label='Actual')
plt.plot(test_dates_plot[-recent:], y_test_pred[-recent:], 
         'lime', linewidth=2, marker='x', markersize=3, label='Predicted')
plt.title(f'Test Set - Last {recent} Days', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history.history['loss'], linewidth=2, label='Train Loss')
ax1.plot(history.history['val_loss'], linewidth=2, label='Val Loss')
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['mae'], linewidth=2, label='Train MAE')
ax2.plot(history.history['val_mae'], linewidth=2, label='Val MAE')
ax2.set_title('MAE')
ax2.set_xlabel('Epoch')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Complete!")