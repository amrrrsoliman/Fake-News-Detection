import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_step_by_step():
    """Test the model loading and prediction step by step"""
    try:
        print("=== STEP 1: Setting up device ===")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        print("\n=== STEP 2: Checking model path ===")
        model_path = r"C:\Users\Dell\Documents\fake_news_model"
        print(f"Model path: {model_path}")
        print(f"Path exists: {os.path.exists(model_path)}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at: {model_path}")
        
        print("\n=== STEP 3: Loading tokenizer ===")
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        print("✅ Tokenizer loaded successfully!")
        
        print("\n=== STEP 4: Loading model ===")
        model = RobertaForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        print("✅ Model loaded successfully!")
        
        # Move model to device
        model.to(device)
        model.eval()
        print(f"✅ Model moved to {device} and set to eval mode")
        
        print("\n=== STEP 5: Testing prediction ===")
        test_text = "Scientists discover new species of deep-sea creatures in the Pacific Ocean"
        print(f"Test text: {test_text}")
        
        # Tokenize
        inputs = tokenizer(
            test_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        print("✅ Text tokenized successfully")
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print("✅ Inputs moved to device")
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        print("✅ Prediction completed")
        
        # Model label mapping: 0 = FAKE, 1 = REAL
        prediction = "REAL" if predicted_class == 1 else "FAKE"
        
        # Get the correct probability for the predicted class
        predicted_probability = probabilities[0][predicted_class].item()
        
        result = {
            "prediction": prediction,
            "confidence": predicted_probability,
            "probabilities": {
                "real": probabilities[0][1].item(),  # Class 1 = REAL
                "fake": probabilities[0][0].item()   # Class 0 = FAKE
            }
        }
        
        print(f"\n=== RESULT ===")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Real probability: {result['probabilities']['real']:.3f}")
        print(f"Fake probability: {result['probabilities']['fake']:.3f}")
        
        print("\n✅ All tests passed! Model is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_step_by_step() 