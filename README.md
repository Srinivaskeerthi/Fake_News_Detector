# Fake News Detector

An advanced machine learning application that detects fake news using multiple classification algorithms and text preprocessing techniques.

## Features

- **Multiple Classifiers**: Choose between Logistic Regression, Random Forest, Multinomial Naive Bayes, and XGBoost
- **Text Preprocessing Options**: Customize with lowercase conversion, punctuation removal, stopword removal, stemming, and lemmatization
- **Interactive UI**: Clean Streamlit interface with color-coded predictions
- **Batch Processing**: Upload CSV files for bulk analysis
- **Performance Metrics**: View model accuracy and confusion matrix
- **Export Results**: Save predictions to CSV

## Requirements

- Python 3.6+
- Required packages listed in `requirements.txt`

## Installation

1. Clone or download this repository
2. Install required packages:

```bash
pip install -r requirements.txt

# Download NLTK resources (if using text preprocessing)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Dataset

You need a dataset with news articles labeled as real or fake. The program expects a CSV file with at least two columns:
- `text`: The news article text
- `label`: Binary label (0 for real news, 1 for fake news)

You can:
1. Use the provided sample dataset for testing
2. Download a larger dataset from Kaggle
3. Use your own dataset (ensure it has the required columns)

## Usage

### Training Models

```bash
python train_model.py
```

This will:
- Train multiple classifier models with various preprocessing options
- Save the best models to the `models` directory
- Generate performance metrics for comparison

### Running the Application

```bash
streamlit run streamlit_app.py
```

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Deploy by pointing to your repository and `streamlit_app.py`

### Heroku

1. Create a `Procfile` with:
   ```
   web: streamlit run streamlit_app.py
   ```

2. Deploy using Heroku CLI:
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

## Project Structure

```
fake-news-detector/
├── data/                  # Dataset files
├── models/                # Saved model files
├── train_model.py         # Script for training models
├── streamlit_app.py       # Streamlit application
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## About the Model

This application uses Natural Language Processing (NLP) and machine learning to classify news as real or fake. The process involves:

1. **Text Preprocessing**: Cleaning and normalizing text
2. **Feature Extraction**: Converting text to numerical features using TF-IDF
3. **Classification**: Using machine learning algorithms to predict authenticity
4. **Evaluation**: Measuring performance with accuracy, precision, recall, and F1-score

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Srinivaskeerthi/Fake_News_Detector 
