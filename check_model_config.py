import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os
import json

def check_model_config():
    """Check the model configuration and label mapping"""
    model_path = r"C:\Users\Dell\Documents\fake_news_model"
    
    print("=== MODEL CONFIGURATION ANALYSIS ===")
    
    # Check if config.json exists
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        print("✅ Found config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"Model type: {config.get('model_type', 'Unknown')}")
        print(f"Architecture: {config.get('architectures', 'Unknown')}")
        print(f"Num labels: {config.get('num_labels', 'Unknown')}")
        
        # Check for label mapping
        if 'id2label' in config:
            print(f"Label mapping: {config['id2label']}")
        else:
            print("No explicit label mapping found")
            
        if 'label2id' in config:
            print(f"Reverse label mapping: {config['label2id']}")
    else:
        print("❌ config.json not found")
    
    # Load model and check
    print("\n=== LOADING MODEL ===")
    try:
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        
        print(f"Model config: {model.config}")
        print(f"Num labels: {model.config.num_labels}")
        
        if hasattr(model.config, 'id2label'):
            print(f"Model id2label: {model.config.id2label}")
        if hasattr(model.config, 'label2id'):
            print(f"Model label2id: {model.config.label2id}")
            
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    check_model_config() 