import requests
import json

def test_api():
    """Test the Flask API endpoint"""
    url = "http://localhost:5000/predict"
    
    # Test data
    test_data = {
        "text": "Scientists discover new species of deep-sea creatures in the Pacific Ocean"
    }
    
    print("Testing API endpoint...")
    print(f"URL: {url}")
    print(f"Data: {test_data}")
    
    try:
        # Send POST request
        response = requests.post(url, json=test_data)
        
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS!")
            print(f"Prediction: {result.get('prediction')}")
            print(f"Confidence: {result.get('confidence')}")
            print(f"Full Response: {json.dumps(result, indent=2)}")
        else:
            print(f"\n❌ ERROR!")
            print(f"Error Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ CONNECTION ERROR!")
        print("Make sure the Flask app is running on http://localhost:5000")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")

if __name__ == "__main__":
    test_api() 