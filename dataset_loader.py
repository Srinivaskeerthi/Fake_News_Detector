import pandas as pd
import os
import urllib.request
import zipfile

def download_dataset():
    """
    Downloads a sample fake news dataset from a public source
    """
    # URL for a sample fake news dataset
    dataset_url = "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/download"
    
    print("Note: This script provides a URL to download a dataset from Kaggle.")
    print("You'll need to:")
    print("1. Visit the URL manually in your browser")
    print("2. Sign in to Kaggle (or create an account)")
    print("3. Download the dataset")
    print("4. Extract the files to the project directory")
    print(f"Dataset URL: {dataset_url}")
    
    print("\nAlternatively, you can use any fake news dataset with 'text' and 'label' columns.")
    print("Place the CSV file in this directory and name it 'news_dataset.csv'")
    
    # Instructions for manual dataset preparation
    print("\nAfter downloading, you may need to prepare the dataset:")
    print("1. Combine fake and real news into a single file if they're separate")
    print("2. Ensure the dataset has 'text' and 'label' columns")
    print("3. For the 'label' column, use 0 for real news and 1 for fake news")

def prepare_sample_dataset():
    """
    Creates a tiny sample dataset for testing purposes
    """
    # Create a small sample dataset for testing
    data = {
        'text': [
            "Scientists discover new species in Amazon rainforest after extensive research.",
            "Breaking: Celebrity secretly an alien, sources close to family reveal shocking truth.",
            "New study shows regular exercise improves heart health and reduces stress.",
            "Government hiding evidence of flat earth, conspiracy theorists claim in viral video.",
            "Local community comes together to rebuild after devastating storm.",
            "Miracle cure for all diseases found in common household item, doctors hate it!",
            "Research indicates climate change accelerating faster than previously predicted.",
            "Elvis spotted alive on moon, exclusive photos inside this article!",
            "New technology enables more efficient solar energy collection, study finds.",
            "Shocking: Water found to cause cancer according to unreleased study."
        ],
        'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 for real, 1 for fake
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('sample_news_dataset.csv', index=False)
    print("Created sample_news_dataset.csv with 10 examples for testing purposes.")
    print("Note: This is a tiny dataset for demonstration only and not suitable for real model training.")

if __name__ == "__main__":
    download_dataset()
    prepare_sample_dataset()