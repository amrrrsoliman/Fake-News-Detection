import requests
import json

def analyze_model_performance():
    """Analyze model performance and suggest improvements"""
    
    # Test cases with expected results
    test_cases = [
        {
            "text": "Scientists discover new species of deep-sea creatures in the Pacific Ocean",
            "expected": "REAL",
            "category": "Science News"
        },
        {
            "text": "BREAKING: Aliens contact Earth government in secret meeting",
            "expected": "FAKE",
            "category": "Conspiracy"
        },
        {
            "text": "New study shows benefits of regular exercise for mental health",
            "expected": "REAL",
            "category": "Health News"
        },
        {
            "text": "5G technology causes coronavirus, experts confirm",
            "expected": "FAKE",
            "category": "Conspiracy"
        },
        {
            "text": "NASA announces plans for Mars mission in 2030",
            "expected": "REAL",
            "category": "Space News"
        },
        {
            "text": "Miracle cure found for all diseases - doctors shocked",
            "expected": "FAKE",
            "category": "Medical Hoax"
        },
        {
            "text": "Climate change study reveals alarming temperature increases",
            "expected": "REAL",
            "category": "Environmental News"
        },
        {
            "text": "Secret government program controls weather patterns",
            "expected": "FAKE",
            "category": "Conspiracy"
        }
    ]
    
    url = "http://localhost:5000/predict"
    
    print("🔍 MODEL PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    correct_predictions = 0
    total_predictions = 0
    accuracy_scores = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['category']}")
        print(f"Text: {test_case['text'][:60]}...")
        print(f"Expected: {test_case['expected']}")
        
        try:
            response = requests.post(url, json={"text": test_case['text']})
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('prediction', 'UNKNOWN')
                accuracy = result.get('confidence', 0)
                probabilities = result.get('probabilities', {})
                
                print(f"Prediction: {prediction}")
                print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
                print(f"Real Prob: {probabilities.get('real', 0):.3f}")
                print(f"Fake Prob: {probabilities.get('fake', 0):.3f}")
                
                # Check if prediction is correct
                if prediction == test_case['expected']:
                    print("✅ CORRECT")
                    correct_predictions += 1
                else:
                    print("❌ INCORRECT")
                
                total_predictions += 1
                accuracy_scores.append(accuracy)
                
            else:
                print(f"❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 40)
    
    # Calculate overall performance
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    avg_confidence = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
    
    print(f"\n📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
    print(f"Overall Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
    print(f"Average Confidence: {avg_confidence:.3f} ({avg_confidence*100:.1f}%)")
    
    # Suggestions for improvement
    print(f"\n💡 SUGGESTIONS FOR IMPROVEMENT")
    print("=" * 60)
    
    if overall_accuracy < 0.8:
        print("❌ Model accuracy is below 80% - needs improvement")
        print("   - Consider retraining with more diverse data")
        print("   - Add more examples of each category")
        print("   - Fine-tune hyperparameters")
    
    if avg_confidence < 0.7:
        print("⚠️  Average confidence is low - model is uncertain")
        print("   - This is normal for many ML models")
        print("   - Consider adjusting confidence thresholds")
    
    print("\n🎯 RECOMMENDATIONS:")
    print("1. Retrain model with larger, more diverse dataset")
    print("2. Add more training examples for edge cases")
    print("3. Consider ensemble methods (combine multiple models)")
    print("4. Fine-tune model hyperparameters")
    print("5. Use data augmentation techniques")

if __name__ == "__main__":
    analyze_model_performance() 