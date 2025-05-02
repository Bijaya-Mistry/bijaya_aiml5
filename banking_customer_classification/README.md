# 🏦 Banking Customer Classification API

A machine learning service to classify banking customers as potential long-term investors. This API uses an **XGBoost classifier** with **FastAPI** for serving models, and supports CSV file upload, JSON prediction, and automatic feature preprocessing.

---

## 📁 Project Structure

Banking customer classification/
├── app/
│ ├── main.py # FastAPI app and route definitions
│ ├── processor.py # ML logic: training, predicting, model persistence
│ ├── schemas.py # Pydantic models for request/response
│ ├── utils.py # Utility functions
│ └── init.py
├── uploads/ # Folder for uploaded CSV files
├── .env # Environment variables (e.g., OpenAI keys, paths)
├── requirements.txt # Python dependencies
├── README.md # This file
└── .gitignore

yaml
Copy
Edit

---

## 🚀 Features

- 📊 **Train and deploy** an XGBoost model to classify potential investors
- 🧹 **Preprocess numeric and categorical features** via pipelines
- 🧪 **Hyperparameter tuning** using GridSearchCV
- 💾 **Model persistence** with joblib
- 🔁 **REST API** endpoints for training, predicting (JSON and CSV), and inspecting model info
- 🔐 `.env` file support for environment configuration

