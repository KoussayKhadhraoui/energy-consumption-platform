# ⚡ Energy Consumption Prediction Platform

A Machine Learning project that analyzes household electricity consumption and predicts the next hour's average active power using historical consumption data.

## 📌 Project Overview

This project uses the **Individual Household Electric Power Consumption** dataset to explore electricity usage patterns, engineer time-series features, train a regression model, and visualize results through an interactive Streamlit dashboard.

The objective is to demonstrate a complete Data Science workflow, from data preprocessing to model evaluation and interactive prediction.

## ✨ Features

- **Data preprocessing:** Clean and prepare household electricity consumption data.
- **Exploratory Data Analysis:** Analyze consumption patterns and trends.
- **Feature engineering:** Create time-based features, lag features, and rolling averages.
- **Machine Learning:** Train a Random Forest Regressor.
- **Model evaluation:** Evaluate predictions using MAE, RMSE, and R².
- **Interactive dashboard:** Explore consumption data and model predictions with Streamlit.

## 🛠️ Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib

## 📊 Dataset

The project uses the UCI Machine Learning Repository's **Individual Household Electric Power Consumption** dataset.

Dataset source: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The original dataset contains approximately 2 million measurements collected from a household over several years.

The data is aggregated into hourly observations before feature engineering and model training.

**Note:** The raw dataset is not included in this repository. Download it from the official source and place the required text file at:

`data/raw/household_power_consumption.txt`

## 🤖 Machine Learning Model

A `RandomForestRegressor` is used to predict the next hour's average active power.

The model uses historical consumption information, time-based variables, lag features, and rolling averages.

### Model Performance

Evaluation results on the chronological test set:

| Metric | Result |
|---|---:|
| MAE | 0.3271 kW |
| RMSE | 0.4746 kW |
| R² Score | 0.5753 |

**Interpretation:** The model achieves an MAE of approximately 0.33 kW. The R² score indicates that the model captures part of the variation in electricity consumption, while further improvements remain possible.

These metrics describe performance on the current test split and are not a guarantee of future prediction accuracy.

## 📁 Project Structure

```text
energy-consumption-platform/
├── app.py
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── energy_model.pkl
├── src/
│   ├── analysis/
│   ├── data/
│   ├── models/
│   └── visualization/
└── tests/
    ├── eda_test.py
    ├── evaluation_test.py
    ├── feature_test.py
    ├── ml_test.py
    ├── prediction_test.py
    └── test_preprocessing.py
```

## 🚀 Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/KoussayKhadhraoui/energy-consumption-platform.git
cd energy-consumption-platform
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the dataset from the official UCI source and place the required file in `data/raw/`.

### 5. Run the dashboard

```bash
python -m streamlit run app.py
```

### 6. Train the model

After preparing the dataset, run:

```bash
python -m tests.ml_test
```

## 🔮 Future Improvements

- Compare additional regression algorithms.
- Improve feature engineering and time-series validation.
- Add prediction uncertainty estimates.
- Optimize model performance.
- Deploy the dashboard online.

## 👨‍💻 Author

**Koussay Khadhraoui**

Computer Science Student — Data Engineering and Decision Support Systems

GitHub: https://github.com/KoussayKhadhraoui

---

*This project was developed as a practical application of data analysis, feature engineering, and Machine Learning.*
