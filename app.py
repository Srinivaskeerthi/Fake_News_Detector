from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load the model and vectorizer
model_path = 'model.pkl'
vectorizer_path = 'vectorizer.pkl'

# Check if model files exist
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    model_loaded = True
else:
    model_loaded = False

@app.route('/')
def home():
    return render_template('index.html', model_status=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})
    
    # Get the news text from the request
    data = request.get_json()
    news_text = data.get('text', '')
    
    if not news_text:
        return jsonify({'error': 'No text provided'})
    
    # Transform the text using the vectorizer
    text_tfidf = vectorizer.transform([news_text])
    
    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    
    # Get prediction probability
    proba = model.predict_proba(text_tfidf)[0]
    confidence = proba[1] if prediction == 1 else proba[0]
    
    result = {
        'prediction': int(prediction),
        'prediction_text': 'Fake' if prediction == 1 else 'Real',
        'confidence': float(confidence)
    }
    
    return jsonify(result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    # This endpoint is for API usage
    return predict()

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template if it doesn't exist
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        .container {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 12px;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-sizing: border-box;
            margin-top: 6px;
            margin-bottom: 16px;
            resize: vertical;
            font-family: inherit;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            display: none;
        }
        .real {
            background-color: #dff0d8;
            border: 1px solid #d6e9c6;
            color: #3c763d;
        }
        .fake {
            background-color: #f2dede;
            border: 1px solid #ebccd1;
            color: #a94442;
        }
        .error {
            background-color: #fcf8e3;
            border: 1px solid #faebcc;
            color: #8a6d3b;
        }
        .loader {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 2s linear infinite;
            margin: 20px auto;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Fake News Detector</h1>
        
        {% if not model_status %}
        <div class="result error" style="display: block;">
            <p><strong>Warning:</strong> Model not loaded. Please run the training script first:</p>
            <code>python train_model.py</code>
        </div>
        {% endif %}
        
        <p>Enter a news article to check if it's real or fake:</p>
        <textarea id="newsText" placeholder="Paste news article text here..."></textarea>
        <button onclick="predictNews()">Analyze</button>
        
        <div class="loader" id="loader"></div>
        
        <div class="result" id="result">
            <p><strong>Prediction:</strong> <span id="prediction"></span></p>
            <p><strong>Confidence:</strong> <span id="confidence"></span>%</p>
        </div>
    </div>

    <script>
        function predictNews() {
            const newsText = document.getElementById('newsText').value.trim();
            
            if (!newsText) {
                showResult('Please enter some text to analyze', 'error');
                return;
            }
            
            // Show loader
            document.getElementById('loader').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            // Send request to the server
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: newsText }),
            })
            .then(response => response.json())
            .then(data => {
                // Hide loader
                document.getElementById('loader').style.display = 'none';
                
                if (data.error) {
                    showResult(data.error, 'error');
                } else {
                    const resultDiv = document.getElementById('result');
                    document.getElementById('prediction').textContent = data.prediction_text;
                    document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(2);
                    
                    resultDiv.className = 'result ' + (data.prediction === 1 ? 'fake' : 'real');
                    resultDiv.style.display = 'block';
                }
            })
            .catch(error => {
                document.getElementById('loader').style.display = 'none';
                showResult('Error: ' + error.message, 'error');
            });
        }
        
        function showResult(message, type) {
            const resultDiv = document.getElementById('result');
            resultDiv.className = 'result ' + type;
            resultDiv.innerHTML = `<p>${message}</p>`;
            resultDiv.style.display = 'block';
        }
    </script>
</body>
</html>
            ''')
    
    # Run the Flask app
    app.run(debug=True)