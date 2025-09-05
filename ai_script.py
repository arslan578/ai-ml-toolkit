#!/usr/bin/env python3
"""
AI Script - A versatile AI toolkit with ML and NLP capabilities
Author: Arslan Rajpoot
Description: A clean, easy-to-run AI script with machine learning and natural language processing features
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Core ML and NLP libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Text processing
from textblob import TextBlob

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_script.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AIScript:
    """
    Main AI Script class with ML and NLP capabilities
    """
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.setup_nltk()
        
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            logger.info("NLTK setup completed successfully")
        except Exception as e:
            logger.warning(f"NLTK setup warning: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for NLP tasks
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        stop_words = set(stopwords.words('english'))
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Perform sentiment analysis on text
        
        Args:
            text (str): Input text for sentiment analysis
            
        Returns:
            Dict: Sentiment analysis results
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Determine sentiment label
            if polarity > 0.1:
                sentiment = "Positive"
            elif polarity < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            return {
                "text": text,
                "sentiment": sentiment,
                "polarity": round(polarity, 3),
                "subjectivity": round(subjectivity, 3),
                "confidence": round(abs(polarity), 3)
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"error": str(e)}
    
    def train_classifier(self, data: List[Dict], text_column: str, label_column: str) -> Dict[str, Any]:
        """
        Train a text classification model
        
        Args:
            data (List[Dict]): Training data
            text_column (str): Name of text column
            label_column (str): Name of label column
            
        Returns:
            Dict: Training results and model info
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Preprocess text
            df['processed_text'] = df[text_column].apply(self.preprocess_text)
            
            # Split data
            X = df['processed_text']
            y = df[label_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create pipeline
            self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Train model
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            self.model.fit(X_train_tfidf, y_train)
            
            # Evaluate model
            X_test_tfidf = self.vectorizer.transform(X_test)
            y_pred = self.model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            self.save_model()
            
            return {
                "status": "success",
                "accuracy": round(accuracy, 3),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "classes": list(self.model.classes_),
                "classification_report": classification_report(y_test, y_pred, output_dict=True)
            }
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {"status": "error", "message": str(e)}
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Make predictions using trained model
        
        Args:
            text (str): Input text for prediction
            
        Returns:
            Dict: Prediction results
        """
        try:
            if self.model is None or self.vectorizer is None:
                return {"error": "Model not trained. Please train the model first."}
            
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Transform and predict
            text_tfidf = self.vectorizer.transform([processed_text])
            prediction = self.model.predict(text_tfidf)[0]
            probabilities = self.model.predict_proba(text_tfidf)[0]
            
            # Get class probabilities
            class_probs = dict(zip(self.model.classes_, probabilities))
            
            return {
                "text": text,
                "prediction": prediction,
                "probabilities": {k: round(v, 3) for k, v in class_probs.items()},
                "confidence": round(max(probabilities), 3)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}
    
    def save_model(self, filename: str = "ai_model.pkl"):
        """Save trained model and vectorizer"""
        try:
            if self.model and self.vectorizer:
                model_data = {
                    'model': self.model,
                    'vectorizer': self.vectorizer
                }
                joblib.dump(model_data, filename)
                logger.info(f"Model saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filename: str = "ai_model.pkl"):
        """Load trained model and vectorizer"""
        try:
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            logger.info(f"Model loaded from {filename}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def generate_sample_data(self) -> List[Dict]:
        """Generate sample data for demonstration"""
        sample_data = [
            {"text": "I love this product! It's amazing and works perfectly.", "label": "positive"},
            {"text": "This is terrible. I hate it and it doesn't work at all.", "label": "negative"},
            {"text": "The product is okay, nothing special but it works.", "label": "neutral"},
            {"text": "Excellent quality and fast delivery. Highly recommended!", "label": "positive"},
            {"text": "Waste of money. Poor quality and slow service.", "label": "negative"},
            {"text": "It's a decent product for the price. No complaints.", "label": "neutral"},
            {"text": "Outstanding service and great customer support!", "label": "positive"},
            {"text": "Very disappointed with this purchase. Not worth it.", "label": "negative"},
            {"text": "Average product, meets basic expectations.", "label": "neutral"},
            {"text": "Fantastic experience! Will definitely buy again.", "label": "positive"}
        ]
        return sample_data
    
    def demo(self):
        """Run demonstration of AI capabilities"""
        logger.info("Starting AI Script Demo...")
        
        # 1. Sentiment Analysis Demo
        print("\n" + "="*50)
        print("SENTIMENT ANALYSIS DEMO")
        print("="*50)
        
        demo_texts = [
            "I absolutely love this new AI technology!",
            "This is the worst experience I've ever had.",
            "The weather is okay today, nothing special."
        ]
        
        for text in demo_texts:
            result = self.sentiment_analysis(text)
            print(f"\nText: {text}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Polarity: {result['polarity']}")
            print(f"Confidence: {result['confidence']}")
        
        # 2. Text Classification Demo
        print("\n" + "="*50)
        print("TEXT CLASSIFICATION DEMO")
        print("="*50)
        
        # Generate and train on sample data
        sample_data = self.generate_sample_data()
        print(f"\nTraining on {len(sample_data)} sample texts...")
        
        training_result = self.train_classifier(sample_data, "text", "label")
        if training_result["status"] == "success":
            print(f"Model trained successfully!")
            print(f"Accuracy: {training_result['accuracy']}")
            print(f"Classes: {training_result['classes']}")
            
            # Test predictions
            test_texts = [
                "This is absolutely wonderful!",
                "I'm very unhappy with this.",
                "It's just average, nothing special."
            ]
            
            print("\nTesting predictions:")
            for text in test_texts:
                prediction = self.predict(text)
                if "error" not in prediction:
                    print(f"\nText: {text}")
                    print(f"Prediction: {prediction['prediction']}")
                    print(f"Confidence: {prediction['confidence']}")
        
        print("\n" + "="*50)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*50)

def main():
    """Main function to run the AI script"""
    parser = argparse.ArgumentParser(description="AI Script - ML and NLP Toolkit")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--sentiment", type=str, help="Analyze sentiment of text")
    parser.add_argument("--predict", type=str, help="Make prediction on text")
    parser.add_argument("--train", type=str, help="Train model with JSON data file")
    parser.add_argument("--load-model", type=str, help="Load existing model")
    
    args = parser.parse_args()
    
    # Initialize AI Script
    ai = AIScript()
    
    if args.demo:
        ai.demo()
    elif args.sentiment:
        result = ai.sentiment_analysis(args.sentiment)
        print(json.dumps(result, indent=2))
    elif args.predict:
        if ai.load_model():
            result = ai.predict(args.predict)
            print(json.dumps(result, indent=2))
        else:
            print("Error: No trained model found. Please train a model first.")
    elif args.train:
        try:
            with open(args.train, 'r') as f:
                data = json.load(f)
            result = ai.train_classifier(data, "text", "label")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error loading training data: {e}")
    elif args.load_model:
        if ai.load_model(args.load_model):
            print("Model loaded successfully!")
        else:
            print("Error loading model.")
    else:
        # Default: run demo
        ai.demo()

if __name__ == "__main__":
    main()
