import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(filepath):
    """
    Load the dataset from a CSV file
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Dataset loaded with shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_data(data, text_column, label_column):
    """
    Preprocess the data by handling missing values and preparing for model training
    """
    # Drop rows with missing values in text or label columns
    data = data.dropna(subset=[text_column, label_column])
    
    # Extract features and labels
    X = data[text_column].values
    y = data[label_column].values
    
    return X, y

def extract_features(X_train, X_test):
    """
    Extract features using TF-IDF Vectorizer
    """
    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    # Fit and transform training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    
    # Transform test data
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

def train_model(X_train, y_train):
    """
    Train a Logistic Regression model
    """
    # Initialize Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    return accuracy, report, conf_matrix

def save_model(model, vectorizer, model_path, vectorizer_path):
    """
    Save the trained model and vectorizer to disk
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

def predict_news(text, model, vectorizer):
    """
    Predict if a news article is fake or real
    """
    # Transform the text using the vectorizer
    text_tfidf = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    
    # Get probability if available
    probability = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(text_tfidf)[0]
        # For string labels like 'real'/'fake'
        if hasattr(model, 'classes_') and len(model.classes_) > 1:
            classes = list(model.classes_)
            if 'fake' in classes:
                fake_idx = classes.index('fake')
                probability = proba[fake_idx]
            elif 1 in classes:  # Assuming 1 = fake, 0 = real
                fake_idx = classes.index(1)
                probability = proba[fake_idx]
            else:
                # Default to second class (typically fake)
                probability = proba[1] if len(proba) > 1 else proba[0]
    
    return prediction, probability

def main():
    # Define file paths
    data_path = "fake_news_dataset.csv"  # Using the Kaggle dataset you added
    model_path = "model.pkl"
    vectorizer_path = "vectorizer.pkl"
    
    # Define column names (update these based on your dataset)
    text_column = "text"  # Column containing news text
    label_column = "label"  # Column containing labels (fake/real)
    
    # Load data
    data = load_data(data_path)
    
    if data is not None:
        # Preprocess data
        X, y = preprocess_data(data, text_column, label_column)
        
        # Split data into training and testing sets (80-20 split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        
        # Extract features
        X_train_tfidf, X_test_tfidf, tfidf_vectorizer = extract_features(X_train, X_test)
        
        # Train model
        model = train_model(X_train_tfidf, y_train)
        
        # Evaluate model
        evaluate_model(model, X_test_tfidf, y_test)
        
        # Save model and vectorizer
        save_model(model, tfidf_vectorizer, model_path, vectorizer_path)
        
        # Example of prediction
        sample_text = "This is a sample news article for testing prediction."
        prediction = predict_news(sample_text, model, tfidf_vectorizer)
        print(f"\nSample prediction: {prediction}")

if __name__ == "__main__":
    main()