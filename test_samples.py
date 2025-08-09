import requests
import json

def test_samples():
    """Test various sample texts to see model behavior"""
    
    samples = [
        {
            "text": "Scientists discover new species of deep-sea creatures in the Pacific Ocean",
            "expected": "REAL"
        },
        {
            "text": "BREAKING: Aliens contact Earth government in secret meeting",
            "expected": "FAKE"
        },
        {
            "text": "New study shows benefits of regular exercise for mental health",
            "expected": "REAL"
        },
        {
            "text": "5G technology causes coronavirus, experts confirm",
            "expected": "FAKE"
        },
        {
            "text": "NASA announces plans for Mars mission in 2030",
            "expected": "REAL"
        },
        {
            "text": "Miracle cure found for all diseases - doctors shocked",
            "expected": "FAKE"
        }
    ]
    
    url = "http://localhost:5000/predict"
    
    print("Testing various sample texts...")
    print("=" * 60)
    
    for i, sample in enumerate(samples, 1):
        print(f"\nTest {i}:")
        print(f"Text: {sample['text']}")
        print(f"Expected: {sample['expected']}")
        
        try:
            response = requests.post(url, json={"text": sample['text']})
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('prediction', 'UNKNOWN')
                confidence = result.get('confidence', 0)
                
                print(f"Prediction: {prediction}")
                print(f"Confidence: {confidence:.3f}")
                
                if prediction == sample['expected']:
                    print("✅ CORRECT")
                else:
                    print("❌ INCORRECT")
                    
            else:
                print(f"❌ API Error: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 40)

if __name__ == "__main__":
    test_samples() 