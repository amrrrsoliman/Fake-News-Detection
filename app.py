from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
import os
import logging

#http://127.0.0.1:5000/

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

# Model path - explicitly set the path
model_path = r"C:\Users\Dell\Documents\fake_news_model"

def load_model():
    """Load the fine-tuned model from local directory"""
    global model, tokenizer, device
    
    try:
        # Set device (GPU if available, else CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Debug: Print model path and check if it exists
        logger.info(f"Model path: {model_path}")
        logger.info(f"Model path type: {type(model_path)}")
        logger.info(f"Model path exists: {os.path.exists(model_path)}")
        
        # Check if model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at: {model_path}")
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load tokenizer and model using the global model_path
        logger.info("Loading tokenizer...")
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        logger.info("Tokenizer loaded successfully!")
        
        logger.info("Loading model...")
        # Use trust_remote_code=True and torch_dtype=torch.float32 for better compatibility
        model = RobertaForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        logger.info("Model loaded successfully!")
        
        # Move model to device
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

def predict_fake_news(text):
    """Predict whether the given text is fake news"""
    global model, tokenizer, device
    
    logger.info("Starting predict_fake_news function")
    
    if model is None or tokenizer is None:
        logger.error("Model or tokenizer is None")
        raise ValueError("Model not loaded. Please ensure the model is loaded first.")
    
    try:
        logger.info("Tokenizing input text...")
        # Tokenize the input text
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        logger.info("Text tokenized successfully")
        
        logger.info("Moving inputs to device...")
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logger.info("Inputs moved to device")
        
        logger.info("Running model prediction...")
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        logger.info(f"Prediction completed. Class: {predicted_class}, Confidence: {confidence}")
        
        # Invert the prediction due to dataset label swapping
        # If model predicts FAKE (class 1), we show REAL, and vice versa
        prediction = "REAL" if predicted_class == 1 else "FAKE"
        
        result = {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {
                "real": probabilities[0][1].item(),  # Invert probabilities too
                "fake": probabilities[0][0].item()
            }
        }
        
        logger.info(f"Returning result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e

@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('fakenewsFront.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for fake news prediction"""
    try:
        logger.info("Received prediction request")
        
        # Get JSON data from request
        data = request.get_json()
        logger.info(f"Request data: {data}")
        
        if not data or 'text' not in data:
            logger.error("No text provided in request")
            return jsonify({
                'error': 'No text provided. Please send JSON with "text" field.'
            }), 400
        
        text = data['text'].strip()
        logger.info(f"Text to analyze: {text[:100]}...")
        
        if not text:
            logger.error("Empty text provided")
            return jsonify({
                'error': 'Text cannot be empty.'
            }), 400
        
        # Make prediction
        logger.info("Starting prediction...")
        result = predict_fake_news(text)
        logger.info(f"Prediction result: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    try:
        # Load the model when starting the application
        logger.info("Starting fake news detection service...")
        load_model()
        
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        exit(1) 