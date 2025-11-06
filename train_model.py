import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Download NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    print("NLTK resources already downloaded or couldn't be downloaded.")

class FakeNewsDetector:
    def __init__(self, model_type='logistic_regression'):
        """
        Initialize the Fake News Detector with the specified model type
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('logistic_regression', 'random_forest', 'multinomial_nb', 'xgboost')
        """
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.metrics = {}
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
    def load_data(self, filepath):
        """
        Load the dataset from a CSV file
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
            
        Returns:
        --------
        pandas.DataFrame or None
            Loaded dataset or None if loading failed
        """
        try:
            data = pd.read_csv(data_path)
            
            print(f"Dataset loaded with shape: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def preprocess_text(self, text, lowercase=True, remove_punctuation=True, 
                        remove_stopwords=True, stemming=False, lemmatization=True):
        """
        Preprocess text data with various options
        
        Parameters:
        -----------
        text : str
            Text to preprocess
        lowercase : bool
            Whether to convert text to lowercase
        remove_punctuation : bool
            Whether to remove punctuation
        remove_stopwords : bool
            Whether to remove stopwords
        stemming : bool
            Whether to apply stemming
        lemmatization : bool
            Whether to apply lemmatization
            
        Returns:
        --------
        str
            Preprocessed text
        """
        if lowercase:
            text = text.lower()
            
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
            
        if remove_stopwords or stemming or lemmatization:
            tokens = nltk.word_tokenize(text)
            
            if remove_stopwords:
                stop_words = set(stopwords.words('english'))
                tokens = [token for token in tokens if token not in stop_words]
                
            if stemming:
                tokens = [self.stemmer.stem(token) for token in tokens]
                
            if lemmatization:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
                
            text = ' '.join(tokens)
            
        return text
    
    def preprocess_data(self, data, text_column, label_column, preprocessing_options=None):
        """
        Preprocess the data by handling missing values and preparing for model training
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Dataset to preprocess
        text_column : str
            Name of the column containing text data
        label_column : str
            Name of the column containing labels
        preprocessing_options : dict or None
            Options for text preprocessing
            
        Returns:
        --------
        tuple
            Preprocessed features and labels
        """
        # Set default preprocessing options if not provided
        if preprocessing_options is None:
            preprocessing_options = {
                'lowercase': True,
                'remove_punctuation': True,
                'remove_stopwords': True,
                'stemming': False,
                'lemmatization': True
            }
        
        # Drop rows with missing values in text or label columns
        data = data.dropna(subset=[text_column, label_column])
        
        # Apply text preprocessing
        data['processed_text'] = data[text_column].apply(
            lambda x: self.preprocess_text(
                x,
                lowercase=preprocessing_options['lowercase'],
                remove_punctuation=preprocessing_options['remove_punctuation'],
                remove_stopwords=preprocessing_options['remove_stopwords'],
                stemming=preprocessing_options['stemming'],
                lemmatization=preprocessing_options['lemmatization']
            )
        )
        
        # Extract features and labels
        X = data['processed_text'].values
        y = data[label_column].values
        
        return X, y
    
    def extract_features(self, X_train, X_test, max_features=5000, ngram_range=(1, 2)):
        """
        Extract features using TF-IDF Vectorizer
        
        Parameters:
        -----------
        X_train : array-like
            Training text data
        X_test : array-like
            Testing text data
        max_features : int
            Maximum number of features to extract
        ngram_range : tuple
            Range of n-grams to consider
            
        Returns:
        --------
        tuple
            Transformed training and testing data, and the vectorizer
        """
        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        
        # Fit and transform training data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        # Transform test data
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        return X_train_tfidf, X_test_tfidf
    
    def get_model(self):
        """
        Get the appropriate model based on the model_type
        
        Returns:
        --------
        object
            Initialized model
        """
        if self.model_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'multinomial_nb':
            return MultinomialNB(alpha=0.1)
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        else:
            print(f"Unknown model type: {self.model_type}. Using Logistic Regression as default.")
            return LogisticRegression(max_iter=1000)
    
    def train_model(self, X_train, y_train):
        """
        Train the selected model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
            
        Returns:
        --------
        object
            Trained model
        """
        # Get the appropriate model
        self.model = self.get_model()
        
        # XGBoost requires numeric labels; map if labels are strings
        if self.model_type == 'xgboost':
            unique_vals = np.unique(y_train)
            if set(unique_vals.tolist()) == set(['fake', 'real']):
    
                label_map = {'real': 0, 'fake': 1}
                y_train = np.array([label_map[y] for y in y_train])

        # Train the model
        self.model.fit(X_train, y_train)
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model performance
        
        Parameters:
        -----------
        X_test : array-like
            Testing features
        y_test : array-like
            Testing labels
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        # Make predictions
        y_pred = self.model.predict(X_test)

        # For XGBoost trained on numeric labels, convert predictions back to strings
        if self.model_type == 'xgboost':
            # If test labels are strings and predictions are numeric, map predictions
            if isinstance(y_test[0], str):
                inv_map = {0: 'real', 1: 'fake'}
                try:
                    y_pred = np.array([inv_map[int(p)] for p in y_pred])
                except Exception:
                    pass
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store metrics
        self.metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
        
        # Print metrics
        print(f"Model: {self.model_type}")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        return self.metrics
    
    def plot_confusion_matrix(self, save_path=None):
        """
        Plot the confusion matrix
        
        Parameters:
        -----------
        save_path : str or None
            Path to save the plot, if None the plot is displayed
        """
        if 'confusion_matrix' not in self.metrics:
            print("No confusion matrix available. Evaluate the model first.")
            return
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            self.metrics['confusion_matrix'], 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake']
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {self.model_type.replace("_", " ").title()}')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_model(self, model_dir='models'):
        """
        Save the trained model and vectorizer to disk
        
        Parameters:
        -----------
        model_dir : str
            Directory to save the model and vectorizer
        """
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Define file paths
        model_path = os.path.join(model_dir, f"{self.model_type}_model.pkl")
        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
        metrics_path = os.path.join(model_dir, f"{self.model_type}_metrics.pkl")
        
        # Save model, vectorizer, and metrics
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.metrics, metrics_path)
        
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
        print(f"Metrics saved to {metrics_path}")
    
    def load_model(self, model_path, vectorizer_path):
        """
        Load a trained model and vectorizer from disk
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
        vectorizer_path : str
            Path to the saved vectorizer
        """
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print(f"Model loaded from {model_path}")
            print(f"Vectorizer loaded from {vectorizer_path}")
            return True
        except Exception as e:
            print(f"Error loading model or vectorizer: {e}")
            return False
    
    def predict(self, text, preprocess=True, preprocessing_options=None):
        """
        Predict if a news article is fake or real
        
        Parameters:
        -----------
        text : str
            Text to predict
        preprocess : bool
            Whether to preprocess the text
        preprocessing_options : dict or None
            Options for text preprocessing
            
        Returns:
        --------
        tuple
            Prediction (0 for real, 1 for fake) and probability
        """
        if self.model is None or self.vectorizer is None:
            print("Model or vectorizer not loaded. Train or load a model first.")
            return None, None
        
        # Set default preprocessing options if not provided
        if preprocessing_options is None:
            preprocessing_options = {
                'lowercase': True,
                'remove_punctuation': True,
                'remove_stopwords': True,
                'stemming': False,
                'lemmatization': True
            }
        
        # Preprocess text if required
        if preprocess:
            processed_text = self.preprocess_text(
                text,
                lowercase=preprocessing_options['lowercase'],
                remove_punctuation=preprocessing_options['remove_punctuation'],
                remove_stopwords=preprocessing_options['remove_stopwords'],
                stemming=preprocessing_options['stemming'],
                lemmatization=preprocessing_options['lemmatization']
            )
        else:
            processed_text = text
        
        # Transform the text using the vectorizer
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(text_tfidf)[0]
        
        # Get prediction probability
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(text_tfidf)[0]
            classes = getattr(self.model, 'classes_', None)
            
            # Always get the probability for the FAKE class
            fake_idx = None
            if classes is not None:
                classes_list = list(classes)
                # Try to find 'fake' or '1' in classes
                if 'fake' in classes_list:
                    fake_idx = classes_list.index('fake')
                elif 1 in classes_list:
                    fake_idx = classes_list.index(1)
                else:
                    # Assume binary classification with second class as fake
                    fake_idx = 1 if len(classes_list) > 1 else 0
                
                # Get probability for fake class
                prob_value = float(proba[fake_idx]) if fake_idx is not None else float(proba[1]) if len(proba) > 1 else float(proba[0])
            else:
                if isinstance(prediction, (int, np.integer)):
                    # For numeric labels, assume 1 = fake, 0 = real
                    prob_value = float(proba[1]) if prediction == 1 and len(proba) > 1 else float(proba[0])
                else:
                    # For string labels, assume 'fake' = fake, 'real' = real
                    pred_str = str(prediction).lower()
                    if pred_str == 'fake':
                        prob_value = float(proba[1]) if len(proba) > 1 else float(proba[0])
                    else:
                        prob_value = float(1.0 - (float(proba[1]) if len(proba) > 1 else float(proba[0])))
        else:
            # If predict_proba is not available, use a default probability
            prob_value = 1.0 if prediction in [1, 'fake'] else 0.0
        
        # Return prediction and probability for FAKE class
        return prediction, prob_value

def train_and_evaluate_models(data_path, text_column, label_column, model_types=None, 
                             preprocessing_options=None, test_size=0.2, random_state=42):
    """
    Train and evaluate multiple models
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset
    text_column : str
        Name of the column containing text data
    label_column : str
        Name of the column containing labels
    model_types : list or None
        List of model types to train
    preprocessing_options : dict or None
        Options for text preprocessing
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing trained models and their metrics
    """
    if model_types is None:
        model_types = ['logistic_regression', 'random_forest', 'multinomial_nb', 'xgboost']
    
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {model_type.replace('_', ' ').title()} model")
        print(f"{'='*50}")
        
        # Initialize detector with the current model type
        detector = FakeNewsDetector(model_type=model_type)
        
        # Load data
        data = detector.load_data(data_path)
        
        if data is not None:
            # Preprocess data
            X, y = detector.preprocess_data(data, text_column, label_column, preprocessing_options)
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            print(f"Training set size: {len(X_train)}")
            print(f"Testing set size: {len(X_test)}")
            
            # Extract features
            X_train_tfidf, X_test_tfidf = detector.extract_features(X_train, X_test)
            
            # Train model
            detector.train_model(X_train_tfidf, y_train)
            
            # Evaluate model
            metrics = detector.evaluate_model(X_test_tfidf, y_test)
            
            # Plot confusion matrix
            detector.plot_confusion_matrix(save_path=f"{model_type}_confusion_matrix.png")
            
            # Save model
            detector.save_model()
            
            # Store results
            results[model_type] = {
                'detector': detector,
                'metrics': metrics
            }
    
    # Find the best model based on accuracy
    best_model = max(results.items(), key=lambda x: x[1]['metrics']['accuracy'])
    print(f"\n{'='*50}")
    print(f"Best model: {best_model[0].replace('_', ' ').title()} with accuracy: {best_model[1]['metrics']['accuracy']:.4f}")
    print(f"{'='*50}")
    
    return results, best_model[0]

if __name__ == "__main__":
    # Define file paths
    data_path = "fake_news_dataset.csv"  # Using the sample dataset
    
    # Define column names (update these based on your dataset)
    text_column = "text"  # Column containing news text
    label_column = "label"  # Column containing labels (fake/real)
    
    # Define preprocessing options
    preprocessing_options = {
        'lowercase': True,
        'remove_punctuation': True,
        'remove_stopwords': True,
        'stemming': False,
        'lemmatization': True
    }
    
    # Define model types to train
    model_types = ['logistic_regression', 'random_forest', 'multinomial_nb', 'xgboost']
    
    # Train and evaluate models
    results, best_model = train_and_evaluate_models(
        data_path, 
        text_column, 
        label_column, 
        model_types=model_types,
        preprocessing_options=preprocessing_options
    )
    
    print("\nTraining complete! Models saved in the 'models' directory.")