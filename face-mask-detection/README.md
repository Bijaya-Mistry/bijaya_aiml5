 # 😷 Face Mask Detection API

This project uses a computer vision model to detect whether people are wearing face masks or not, and serves predictions via a FastAPI backend.

## 📂 Project Structure

face-mask-detection/
├── app/                      # FastAPI application
│   ├── main.py               # Entry point for the API
│   ├── model.py              # Model loading and prediction logic
│   ├── schemas.py            # Pydantic models for request/response
│   ├── utils.py              # Helper functions (optional)
│   └── __init__.py
│
├── models/                   # Directory to store trained model files
│   └── mask_detector.model   # Exported model file
│
├── notebooks/                # Jupyter notebooks for development
│   └── face-mask-detection.ipynb
│
├── train_export_model.py     # Script to train and export the model
├── .env                      # Environment variables like model path
├── requirements.txt          # Project dependencies
├── .gitignore                # Files and folders to ignore in git
└── README.md                 # Project documentation
