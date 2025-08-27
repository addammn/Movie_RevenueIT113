# 🎬 Movie_IT113 — Two-Model Movie Revenue Prediction

Simple goal: **predict movie revenue** and **compare two models** side-by-side.

## 🚀 What we built
- **Model A (Baseline):** Linear Regression  
- **Model B (Stronger):** Random Forest Regressor  
- Target: `revenue` (in millions USD)  
- Dataset file in repo: `movie_boxoffice.csv`

## 📂 Key files
- `Two_Models_Training_Colab.ipynb` — open in **Google Colab**, runs end-to-end
- `movie_boxoffice.csv` — demo dataset (realistic, small)
- `artifacts/` — trained models & metrics (commit these after running the notebook)
  - `model_A.joblib`, `model_B.joblib`
  - `metrics_A.json`, `metrics_B.json`
  - `feature_columns.json`
- `streamlit_app.py` — simple UI to make predictions with both models
- `requirements.txt` — dependencies

## 📝 Run in Google Colab
Open directly from GitHub:

