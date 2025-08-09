import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    """Load the fine-tuned model from local directory"""
    try:
        # Set device (GPU if available, else CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Model path
        model_path = r"C:\Users\Dell\Documents\fake_news_model"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at: {model_path}")
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load tokenizer and model
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        
        # Move model to device
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        logger.info("Model loaded successfully!")
        return model, tokenizer, device
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

def predict_fake_news(text, model, tokenizer, device):
    """
    Predict whether the given text is fake news
    
    Args:
        text (str): Input text to analyze
        model: Loaded DistilBERT model
        tokenizer: Loaded tokenizer
        device: PyTorch device (CPU/GPU)
    
    Returns:
        dict: Prediction results with label and confidence
    """
    try:
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
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise e

def test_model():
    """Test the model with sample texts"""
    try:
        # Load model
        model, tokenizer, device = load_model()
        
        # Sample texts for testing
        test_texts = [
            "Scientists discover new species of deep-sea creatures in the Pacific Ocean",
            "BREAKING: Aliens contact Earth government in secret meeting",
            "New study shows benefits of regular exercise for mental health",
            "5G technology causes coronavirus, experts confirm",
            "NASA announces plans for Mars mission in 2030",
            "Miracle cure found for all diseases - doctors shocked"
        ]
        
        print("\n" + "="*60)
        print("FAKE NEWS DETECTION TEST")
        print("="*60)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}:")
            print(f"Text: {text}")
            
            result = predict_fake_news(text, model, tokenizer, device)
            
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Real probability: {result['probabilities']['real']:.3f}")
            print(f"Fake probability: {result['probabilities']['fake']:.3f}")
            print("-" * 40)
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_model() 