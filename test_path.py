import os

# Test the model path
model_path = r"C:\Users\Dell\Documents\fake_news_model"

print(f"Testing model path: {model_path}")
print(f"Path exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    print("✅ Model directory found!")
    print("Contents of directory:")
    try:
        files = os.listdir(model_path)
        for file in files:
            print(f"  - {file}")
    except Exception as e:
        print(f"Error listing directory: {e}")
else:
    print("❌ Model directory not found!")
    print("Please check if the path is correct:")
    print(f"Expected path: {model_path}")
    
    # Check if the Documents folder exists
    docs_path = r"C:\Users\Dell\Documents"
    print(f"\nDocuments folder exists: {os.path.exists(docs_path)}")
    
    if os.path.exists(docs_path):
        print("Contents of Documents folder:")
        try:
            files = os.listdir(docs_path)
            for file in files:
                if "fake" in file.lower() or "news" in file.lower() or "model" in file.lower():
                    print(f"  - {file} (potential match)")
        except Exception as e:
            print(f"Error listing Documents: {e}") 