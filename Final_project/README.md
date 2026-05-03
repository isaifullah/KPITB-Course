# Talk2Data – AI-Powered Data Assistant

## 📌 Project Overview

**Talk2Data** is an intelligent web application that allows users to upload tabular datasets (CSV, Excel) and interact with them using natural language. It answers questions, generates Python code, creates visualizations, and provides advanced data manipulation features like column renaming, encoding, binning, merging, and feature engineering – all through a user‑friendly chat interface. The assistant is powered by **GPT‑4o Mini**.

---

## 🚀 Features

### 1. Smart Chat Assistant
- Ask natural language questions about your data
- Automatically shows rows (`show first 5 rows`, `row 10`)
- Detects missing values, duplicates, outliers
- Generates ready‑to‑run Python code with copy button

### 2. Dataset Profiling
- Displays rows, columns, missing values, duplicates on upload
- Column Info tab shows detailed statistics (mean, std, quartiles, top values)

### 3. Interactive Visualizations
- Matplotlib, Seaborn, or Plotly
- 6+ plot types: histogram, bar chart, scatter, box, line, pie, heatmap

### 4. Data Manipulation (Persistent)
- Rename columns
- Label, One‑Hot, or Ordinal encoding
- Feature engineering: binning, merging columns, custom expressions

### 5. Algorithm Suggestions
- Recommends regression or classification algorithms (15+ algorithms)

### 6. Data Quality Score
- Completeness, uniqueness, numeric ratio → grade (A‑D)

### 7. Download Processed Data
- Export as CSV with custom filename

---

## 🛠️ Technology Stack

| Component | Tools |
|-----------|-------|
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy, scikit-learn |
| Visualizations | Matplotlib, Seaborn, Plotly |
| LLM | OpenAI GPT-4o Mini |
| Environment | Python 3.9+, dotenv |

---

## 📁 Project Structure
Final_project/
├── app.py # Complete Streamlit application
├── requirements.txt # Dependencies
├── .env # OpenAI API key
└── talk2data.ipynb # Jupyter notebook with all functions

---

## 🔧 Installation & Setup

### 1. Create project folder
```bash
mkdir Final_project
cd inal_project

python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

install dependencies
pip install -r requirements.txt

create an api file and store your api key
OPENAI_API_KEY=your_api_key_here

run the app
streamlit run app.py