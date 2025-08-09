from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

#http://127.0.0.1:5000/

app = Flask(__name__)
CORS(app)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

def load_model():
    """Load the fine-tuned model from local directory"""
    global model, tokenizer, device
    
    # Set device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model path
    model_path = r"C:\Users\Dell\Documents\fake_news_model"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    
    # Move model to device
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    print("Model loaded successfully!")

def predict_fake_news(text):
    """Predict whether the given text is fake news"""
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        raise ValueError("Model not loaded. Please ensure the model is loaded first.")
    
    # Tokenize the input text
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Assuming 0 = REAL, 1 = FAKE (adjust based on your model's label mapping)
    prediction = "FAKE" if predicted_class == 1 else "REAL"
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            "real": probabilities[0][0].item(),
            "fake": probabilities[0][1].item()
        }
    }

@app.route('/predict', methods=['POST'])
def predict():
    """
    Flask route that receives POST request with text, 
    calls the prediction function, and returns the result as JSON
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided. Please send JSON with "text" field.'
            }), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({
                'error': 'Text cannot be empty.'
            }), 400
        
        # Call the prediction function
        result = predict_fake_news(text)
        
        # Return result as JSON
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in prediction endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred during prediction.',
            'details': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else None
    })

if __name__ == '__main__':
    try:
        # Load the model when starting the application
        print("Starting fake news detection service...")
        load_model()
        
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
        exit(1) 