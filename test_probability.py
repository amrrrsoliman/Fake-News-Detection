import requests
import json

def test_probabilities():
    """Test various texts to see probability distributions"""
    
    test_texts = [
        "Scientists discover new species of deep-sea creatures in the Pacific Ocean",
        "BREAKING: Aliens contact Earth government in secret meeting",
        "New study shows benefits of regular exercise for mental health",
        "5G technology causes coronavirus, experts confirm",
        "NASA announces plans for Mars mission in 2030",
        "Miracle cure found for all diseases - doctors shocked"
    ]
    
    url = "http://localhost:5000/predict"
    
    print("Testing probability distributions...")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text[:50]}...")
        
        try:
            response = requests.post(url, json={"text": text})
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('prediction', 'UNKNOWN')
                confidence = result.get('confidence', 0)
                probabilities = result.get('probabilities', {})
                
                print(f"Prediction: {prediction}")
                print(f"Overall Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
                print(f"Real Probability: {probabilities.get('real', 0):.3f} ({probabilities.get('real', 0)*100:.1f}%)")
                print(f"Fake Probability: {probabilities.get('fake', 0):.3f} ({probabilities.get('fake', 0)*100:.1f}%)")
                
                # Check if probabilities add up to 1
                total_prob = probabilities.get('real', 0) + probabilities.get('fake', 0)
                print(f"Total Probability: {total_prob:.3f}")
                
            else:
                print(f"❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 40)

if __name__ == "__main__":
    test_probabilities() 