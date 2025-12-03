# 📈 LSTM Stock Price Prediction

A deep learning project using LSTM (Long Short-Term Memory) neural networks to predict Apple (AAPL) stock prices based on historical data from 2019-2024.

## 🎯 Project Overview

This project implements a multi-layer LSTM model to forecast next-day closing prices for Apple stock. The model uses 60 days of historical OHLCV (Open, High, Low, Close, Volume) data to predict the next day's closing price.

### Key Features
- **Multi-feature input**: Uses Open, High, Low, Close, and Volume data
- **Robust architecture**: 3-layer LSTM with dropout regularization
- **Comprehensive EDA**: 11 detailed visualizations for data exploration
- **Proper validation**: 70/15/15 train/validation/test split
- **Reproducible results**: Fixed random seeds for consistency
- **Model persistence**: Saves best performing model automatically

## 📊 Results

The model achieves strong performance on the test set:

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **MAE** | ~$X.XX | ~$X.XX | ~$X.XX |
| **RMSE** | ~$X.XX | ~$X.XX | ~$X.XX |
| **MAPE** | ~X.XX% | ~X.XX% | ~X.XX% |

*Note: Actual values depend on your training run*

**Performance Interpretation:**
- MAPE < 5%: Excellent
- MAPE < 10%: Good
- MAPE > 10%: Needs improvement

## 🗂️ Project Structure

```
lstm-stock-prediction/
│
├── AAPL_2000_2024.csv          # Raw stock data (2000-2024)
├── lstm_stock_prediction.py     # Main training script
├── models/                      # Saved models directory
│   ├── best_model_recent.keras # Best trained model
│   └── scaler_recent.gz        # MinMaxScaler for data normalization
│
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Scikit-learn**: Data preprocessing and scaling
- **Matplotlib & Seaborn**: Data visualization

## 📋 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

### requirements.txt
```
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
```

## 🚀 Getting Started

### 1. Data Preparation

Ensure your CSV file (`AAPL_2000_2024.csv`) has the following structure:
- First two rows: Metadata/headers (will be skipped)
- Columns: Date, Close, High, Low, Open, Volume

### 2. Training the Model

Run the main script:

```bash
python lstm_stock_prediction.py
```

The script will:
1. Load and preprocess the data (filtering to 2019-2024)
2. Perform comprehensive exploratory data analysis
3. Split data into train/validation/test sets
4. Scale features using MinMaxScaler
5. Create sequences of 60 days for LSTM input
6. Train the model with early stopping
7. Evaluate performance on all datasets
8. Generate visualizations

### 3. Model Training Details

**Hyperparameters:**
- Sequence length: 60 days
- Batch size: 32
- Learning rate: 0.001 (Adam optimizer)
- Max epochs: 200 (with early stopping)
- Early stopping patience: 20 epochs

**Architecture:**
```
LSTM(200) → Dropout(0.25) → 
LSTM(150) → Dropout(0.25) → 
LSTM(100) → Dropout(0.30) → 
Dense(1)
```

## 📊 Exploratory Data Analysis

The project includes 11 comprehensive visualizations:

1. **Missing Values Check**: Data quality assessment
2. **Closing Price Over Time**: Overall trend visualization
3. **OHLC Comparison**: All price types plotted together
4. **Correlation Matrix**: Feature relationships heatmap
5. **Scatter Plots**: Close price vs. other features
6. **Distribution Histograms**: Feature frequency distributions
7. **Volume vs Price**: Dual y-axis comparison
8. **Daily Returns**: Percentage change over time
9. **Returns Distribution**: Volatility analysis
10. **Box Plots**: Outlier detection
11. **Summary Statistics**: Comprehensive dataset overview

## 🔍 Model Architecture Explained

### Why LSTM?
- **Long-term dependencies**: Captures patterns across many days
- **Sequential data**: Designed for time series
- **Non-linear patterns**: Learns complex market behaviors

### Layer Breakdown

**Layer 1 (LSTM-200):**
- Captures long-term trends and patterns
- L2 regularization prevents overfitting
- Returns sequences for next layer

**Layer 2 (LSTM-150):**
- Refines patterns from previous layer
- Reduces dimensionality gradually
- Still returns sequences

**Layer 3 (LSTM-100):**
- Final feature extraction
- Outputs single vector (return_sequences=False)
- Higher dropout (0.3) before output

**Output Layer (Dense-1):**
- Single neuron for price prediction
- No activation function (regression task)

### Regularization Techniques
- **Dropout (0.25-0.30)**: Randomly drops neurons during training
- **L2 Regularization (0.001)**: Penalizes large weights
- **Early Stopping**: Prevents training too long

## 📈 Key Design Decisions

### 1. **Recent Data Only (2019-2024)**
**Why?** Older data has different price ranges and volatility patterns. Using recent 5 years provides:
- More consistent price scale
- Similar market conditions
- Better training stability

### 2. **70/15/15 Split**
**Why?** 
- 70% training: Enough data to learn patterns
- 15% validation: Hyperparameter tuning and early stopping
- 15% test: Final unbiased evaluation

### 3. **Sequence Length = 60 Days**
**Why?** 
- ~3 months of trading data
- Captures quarterly patterns
- Not too long (overfitting) or short (insufficient context)

### 4. **Context Preservation**
When creating validation and test sequences, the last 60 days of the previous set are included to prevent discontinuity at boundaries.

## 🎓 Evaluation Metrics

### Mean Absolute Error (MAE)
- Average prediction error in dollars
- Easy to interpret: "On average, predictions are off by $X"

### Root Mean Squared Error (RMSE)
- Penalizes large errors more heavily
- Sensitive to outliers
- Higher values indicate occasional large mistakes

### Mean Absolute Percentage Error (MAPE)
- Error as percentage of actual price
- Scale-independent metric
- Best for comparing across different stocks

## 📉 Visualizations Generated

### Training Process
- **Loss curves**: Shows model learning over epochs
- **MAE curves**: Tracks prediction accuracy improvement

### Predictions
- **Full dataset plot**: All predictions across train/val/test
- **Test set zoom**: Detailed view of recent predictions
- Both actual vs. predicted comparisons

## ⚠️ Limitations & Disclaimers

### Model Limitations
1. **Past performance ≠ future results**: Markets are unpredictable
2. **No external factors**: Doesn't consider news, earnings, or macro events
3. **Single stock**: Trained only on AAPL, not generalizable
4. **Short-term predictions**: Only predicts next day

### Financial Disclaimer
**⚠️ This project is for educational purposes only!**
- Not financial advice
- Do not use for actual trading decisions
- Stock markets involve significant risk
- Consult financial professionals before investing

## 🔮 Future Improvements

### Data Enhancements
- [ ] Add technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Include sentiment analysis from news/social media
- [ ] Incorporate macroeconomic indicators
- [ ] Multi-stock training for better generalization

### Model Improvements
- [ ] Implement attention mechanisms
- [ ] Try Transformer architectures
- [ ] Ensemble multiple models
- [ ] Add uncertainty quantification (prediction intervals)

### Engineering
- [ ] Real-time data pipeline
- [ ] Hyperparameter tuning with Optuna
- [ ] Model deployment as API
- [ ] Interactive web dashboard
**Happy Predicting! 📊🚀**

*Remember: Use this for learning, not for trading decisions!*
