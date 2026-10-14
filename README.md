## 📊 Dataset

This project uses the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository.

**Official dataset:** https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The dataset contains household electricity measurements collected over several years. The project processes these measurements into hourly observations to analyze consumption patterns and predict the next hour's average active power.

### Download and Prepare the Dataset

Follow these steps before running the application.

**Step 1: Download the dataset**

1. Open the [official UCI dataset page](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).
2. Download the dataset archive.
3. Extract the downloaded ZIP file.
4. Locate the file named `household_power_consumption.txt`.

**Step 2: Create the required folders**

From the project root directory, create the folders if they do not already exist:

```powershell
New-Item -ItemType Directory -Force data\raw
```

**Step 3: Place the dataset**

Copy `household_power_consumption.txt` into:

```text
energy-consumption-platform/
└── data/
    └── raw/
        └── household_power_consumption.txt
```

Make sure the filename and folder structure match exactly.

**Important:** The raw dataset is excluded from GitHub to avoid uploading large data files. Each user must download and prepare the dataset locally before running the application.

## 🚀 Installation and Usage

### 1. Clone the repository

Make sure Git and Git LFS are installed, then run:

```powershell
git lfs install
git clone https://github.com/KoussayKhadhraoui/energy-consumption-platform.git
cd energy-consumption-platform
```

Git LFS is required to download the saved model file tracked with Git LFS.

### 2. Create a virtual environment

```powershell
python -m venv venv
```

Activate it in Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you can use the virtual environment's Python directly without activating it.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

If you did not activate the virtual environment, run:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Prepare the dataset

Complete the steps in the **Download and Prepare the Dataset** section above.

Verify that the following file exists:

```text
data/raw/household_power_consumption.txt
```

### 5. Launch the Streamlit dashboard

With the virtual environment activated:

```powershell
python -m streamlit run app.py
```

Without activating the environment:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the dashboard.

### 6. Train the Machine Learning model

If you want to retrain the model using the prepared dataset, run:

```powershell
python -m tests.ml_test
```

If the virtual environment is not activated:

```powershell
.\venv\Scripts\python.exe -m tests.ml_test
```

Training requires the dataset to be available locally. It may replace the saved model, so keep a backup if you want to preserve the existing version.

## 🛠️ Troubleshooting

- **Dataset file not found:** Check that `household_power_consumption.txt` is inside `data/raw/`.
- **Module not found:** Make sure you installed the dependencies in the same Python environment used to launch Streamlit.
- **PowerShell activation blocked:** Use `.\venv\Scripts\python.exe` to run Python commands without activating the environment.
- **Model loading error:** Make sure Git LFS downloaded the actual model file rather than leaving an LFS pointer in its place.
## 📸 Dashboard Screenshots

### Overview
![Overview Dashboard](screenshots/overview.png)

### Data Analysis
![Data Analysis](screenshots/analysis.png)

### Energy Prediction
![Energy Prediction](screenshots/prediction.png)

### Model Performance
![Model Performance](screenshots/model.png)

