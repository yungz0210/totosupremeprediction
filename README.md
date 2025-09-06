# totosupremeprediction
Predict Toto Star, Power and Supreme numbers

# Toto Predictor — Enhanced Lottery Prediction App

A Streamlit-based Toto/lottery number predictor with multiple prediction methods and analysis features.

## Features

- **Games supported**: Star 6/50, Power 6/55, Supreme 6/58
- **Prediction methods**:
  - **Hot Numbers**: Picks frequently drawn numbers
  - **Cold Numbers**: Picks least frequently drawn numbers
  - **Monte Carlo**: Simulates random draws with frequency weighting
  - **ML Logistic Regression**: Predicts numbers based on historical draws
  - **LSTM (optional)**: Sequence-based deep learning prediction
  - **Hybrid (Freq + ML)**: Combines frequency and ML scores
  - **Markov Chain**: Predicts numbers based on previous draw patterns
- **Analysis tools**:
  - Number frequency, pairs, triples
  - Gap analysis
  - Co-occurrence heatmap
  - Yearly, monthly, weekday trends
  - Clustering (KMeans)
- **Simulation/backtesting**: Evaluate strategies on historical draws
- **Global filters**: Filter by year range, month, or weekday
- **Data export**: Download filtered draws or backtest results

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/toto-predictor.git
cd toto-predictor
