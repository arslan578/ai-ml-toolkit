# AI Script - Machine Learning & NLP Toolkit

A clean, easy-to-run AI script with machine learning and natural language processing capabilities. Built by Arslan Rajpoot for versatile AI applications.

## 🚀 Features

- **Sentiment Analysis**: Analyze text sentiment (positive, negative, neutral)
- **Text Classification**: Train and use ML models for text categorization
- **Natural Language Processing**: Text preprocessing, tokenization, and lemmatization
- **Model Persistence**: Save and load trained models
- **Easy Integration**: Simple API for integration with existing systems
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 📋 Requirements

- Python 3.7+
- Required packages listed in `requirements.txt`

## 🛠️ Installation

### 1. Clone or Download
```bash
# If using git
git clone <repository-url>
cd project

# Or simply download the files to your project directory
```

### 2. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv ai_env
source ai_env/bin/activate  # On Windows: ai_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python ai_script.py --demo
```

## 🎯 Quick Start

### Run Demo
```bash
python ai_script.py --demo
```

### Sentiment Analysis
```bash
python ai_script.py --sentiment "I love this AI script!"
```

### Train a Model
```bash
# Create training data file (sample_data.json)
python ai_script.py --train sample_data.json
```

### Make Predictions
```bash
python ai_script.py --predict "This is amazing!"
```

## 📖 Usage Examples

### 1. Basic Sentiment Analysis
```python
from ai_script import AIScript

ai = AIScript()
result = ai.sentiment_analysis("I absolutely love this product!")
print(result)
# Output: {'text': 'I absolutely love this product!', 'sentiment': 'Positive', 'polarity': 0.8, 'subjectivity': 0.6, 'confidence': 0.8}
```

### 2. Train Text Classification Model
```python
# Prepare your data
training_data = [
    {"text": "Great product!", "label": "positive"},
    {"text": "Terrible service", "label": "negative"},
    {"text": "It's okay", "label": "neutral"}
]

# Train the model
ai = AIScript()
result = ai.train_classifier(training_data, "text", "label")
print(f"Model accuracy: {result['accuracy']}")
```

### 3. Make Predictions
```python
# After training
prediction = ai.predict("This is wonderful!")
print(f"Prediction: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence']}")
```

## 📁 Project Structure

```
project/
├── ai_script.py          # Main AI script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── sample_data.json     # Example training data (optional)
├── ai_model.pkl         # Saved model (after training)
└── ai_script.log        # Log file (generated during execution)
```

## 🔧 Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--demo` | Run full demonstration | `python ai_script.py --demo` |
| `--sentiment <text>` | Analyze sentiment | `python ai_script.py --sentiment "Great!"` |
| `--predict <text>` | Make prediction | `python ai_script.py --predict "Amazing!"` |
| `--train <file>` | Train with JSON data | `python ai_script.py --train data.json` |
| `--load-model <file>` | Load existing model | `python ai_script.py --load-model model.pkl` |

## 📊 Training Data Format

Your training data should be in JSON format:

```json
[
    {
        "text": "I love this product!",
        "label": "positive"
    },
    {
        "text": "This is terrible.",
        "label": "negative"
    },
    {
        "text": "It's okay, nothing special.",
        "label": "neutral"
    }
]
```

## 🔍 API Reference

### AIScript Class

#### Methods

- `sentiment_analysis(text)`: Analyze sentiment of input text
- `train_classifier(data, text_column, label_column)`: Train text classification model
- `predict(text)`: Make predictions using trained model
- `save_model(filename)`: Save trained model to file
- `load_model(filename)`: Load model from file
- `preprocess_text(text)`: Preprocess text for NLP tasks

## 🚨 Troubleshooting

### Common Issues

1. **NLTK Data Download Error**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

2. **Permission Errors**
   - Make sure you have write permissions in the project directory
   - Try running with `sudo` if necessary (Linux/Mac)

3. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version: `python --version` (should be 3.7+)

4. **Model Loading Errors**
   - Ensure model file exists and is not corrupted
   - Train a new model if needed

## 📈 Performance Tips

- For large datasets, consider using more powerful ML algorithms
- Adjust `max_features` in TfidfVectorizer for better performance
- Use GPU acceleration for larger models (requires additional setup)

## 🔄 Integration Examples

### Flask API Integration
```python
from flask import Flask, request, jsonify
from ai_script import AIScript

app = Flask(__name__)
ai = AIScript()

@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    text = request.json.get('text')
    result = ai.sentiment_analysis(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

### Django Integration
```python
# views.py
from django.http import JsonResponse
from ai_script import AIScript

def sentiment_view(request):
    if request.method == 'POST':
        text = request.POST.get('text')
        ai = AIScript()
        result = ai.sentiment_analysis(text)
        return JsonResponse(result)
```

## 📝 Logging

The script automatically creates detailed logs in `ai_script.log`. Logs include:
- Training progress
- Prediction results
- Error messages
- Performance metrics

## 🤝 Support

For questions or issues:
- Check the logs in `ai_script.log`
- Review the troubleshooting section
- Ensure all dependencies are properly installed

## 📄 License

This project is created for client use. All rights reserved.

---

**Created by Arslan Rajpoot**  
Full Stack Developer & AI Engineer  
Portfolio: [aimexperts.com](https://aimexperts.com)  
GitHub: [github.com/arslan578](https://github.com/arslan578)
