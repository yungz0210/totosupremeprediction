# Toto Predictor — Enhanced with LSTM, Global Filters, Pages

A **Streamlit web app** for Malaysian lottery number analysis and prediction.  
Includes statistical analysis, ML models, LSTM predictions (if TensorFlow is installed), Markov chains, and hybrid strategies.

---

## Features
- Import historical draws from Google Sheets for **Star 6/50**, **Power 6/55**, and **Supreme 6/58**.
- **Analysis**: frequency, hot/cold numbers, pairs/triples, gap analysis, co-occurrence heatmaps, yearly/monthly/weekday trends, clustering.
- **Prediction**:
  - Hot/Cold
  - Monte Carlo
  - Logistic Regression ML
  - Optional LSTM
  - Hybrid (Frequency + ML)
  - Markov Chain (order 1,2,3)
- **Simulation/Backtesting**: test strategies against historical draws.
- Download filtered data, backtest results, or Markov summaries as CSV.

---

## Installation

```bash
git clone https://github.com/yourusername/toto-predictor.git
cd toto-predictor
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
