# Project Overview
This repository contain my Degree Data Analytics Final Year Project Code:

📈 Enhancing Stock Forecast Accuracy via AI-Driven Financial Sentiment Analysis

This project investigates the use of AI-driven financial sentiment analysis to support market trend prediction in cryptocurrency and stock markets.
By leveraging FinBERT, traditional machine learning models, and resampling strategies, the system extracts sentiment signals from financial news and social media data to improve predictive accuracy for traders and investors.


The project covers:

- Financial text preprocessing
- Sentiment classification using FinBERT
- Handling class imbalance (ROS, RUS, Class Weighted Loss)
- Model evaluation and comparison
- Deployment via a Streamlit


# Models Used

- FinBERT (Transformer-based financial sentiment model) 
- LSTM
- Linear SVC
- Random Forest
- LightGBM
- Multinomial Naive Bayes

Resampling strategies:

- Random Over-Sampling (ROS)
- Random Under-Sampling (RUS)
- Class-weighted loss

# Dataset

Due to GitHub file size limitations, full datasets are hosted externally.

Dataset (Hugging Face):
https://huggingface.co/datasets/JohnTz/FYP_Dataset

The datasets include:
- Financial news articles 
- Twitter financial sentiment data

# Trained Models

Final trained models are hosted externally to comply with GitHub storage constraints.

FinBERT Models (Hugging Face):
https://huggingface.co/JohnTz/FYP_Model

# Environment Setup
1️⃣ Create Virtual Environment (Optional)

	python -m venv venv
	venv\Scripts\activate

2️⃣ Install Dependencies

	pip install -r requirements.txt


# ▶️ How to Run the Project
Step 1: Data Understanding + Preprocessing
- newsdata_prep.ipynb
- twitter_news_eda.ipynb


Step 2: Assign Sentiment Label
- finbert_classify.ipynb

Step 3: Data Integration + Balancing
- integration_sampling.ipynb

Step 4: Train FinBERT Models
- finbert_train.ipynb
- ml_train.ipynb
- ml_train_rus.ipynb

Step 5: Run Application (Optional)
python app.py
- Run This Code In Terminal To Open Web Application

		streamlit run app.py

# Results & Evaluation

Model performance is evaluated using:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- Comparative analysis between baseline ML models and FinBERT variants

Final evaluation results are stored as summary metrics, not raw checkpoints.

# Notes on Reproducibility
- Logs are excluded due to size constraints.
- All experiments are fully reproducible using the provided notebooks.
- Random seeds are fixed where applicable to ensure consistent results.

# Academic Note

This repository follows industry-standard machine learning practices, where:
- Source code is version-controlled
- Large datasets and trained models are stored in dedicated artifact repositories
- Experimental methodology and results remain transparent and reproducible

# Author

Tan Wei Ming

Final Year Project (FYP)

Bachelor of Computer Science Specialism With Data Analytics

# License

This project is for academic and research purposes only.
