# Fake News Detection System

A web-based fake news detection system using a fine-tuned DistilBERT model with a Flask backend and modern HTML/CSS frontend.

## Features

- 🤖 **AI-Powered Detection**: Uses a fine-tuned DistilBERT model for accurate fake news detection
- 🌐 **Web Interface**: Modern, responsive web interface with real-time predictions
- 📊 **Confidence Scores**: Shows prediction confidence and probability breakdown
- ⚡ **Fast Processing**: Optimized for quick text analysis
- 📱 **Mobile Friendly**: Responsive design works on all devices

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Model Setup

Ensure your fine-tuned DistilBERT model is located at:
```
C:\Users\Dell\Documents\fake_news_model
```

The model directory should contain:
- `config.json`
- `pytorch_model.bin`
- `tokenizer.json`
- `vocab.txt`
- Other model files

### 3. Test the Model

Before running the web app, test your model:

```bash
python test_model.py
```

This will run sample predictions to verify the model is working correctly.

### 4. Run the Web Application

```bash
python app.py
```

The application will be available at: `http://localhost:5000`

## API Endpoints

### POST `/predict`
Analyze text for fake news detection.

**Request:**
```json
{
    "text": "Your news article or headline here"
}
```

**Response:**
```json
{
    "prediction": "FAKE",
    "confidence": 0.85,
    "probabilities": {
        "real": 0.15,
        "fake": 0.85
    }
}
```

### GET `/health`
Check if the model is loaded and service is running.

## Files Structure

```
fake_news_project/
├── app.py                 # Flask backend application
├── test_model.py          # Standalone model testing script
├── requirements.txt       # Python dependencies
├── fakenewsFront.html    # Frontend HTML file
└── README.md             # This file
```

## Model Configuration

The system assumes:
- **Class 0**: REAL news
- **Class 1**: FAKE news

If your model uses different label mapping, update the prediction logic in `predict_fake_news()` function.

## Troubleshooting

### Model Loading Issues
- Ensure the model path is correct
- Check that all model files are present
- Verify PyTorch and Transformers versions are compatible

### API Errors
- Check if Flask server is running
- Verify CORS is enabled for cross-origin requests
- Ensure JSON format is correct in requests

### Performance Tips
- Use GPU if available for faster inference
- Consider model quantization for production deployment
- Implement caching for repeated predictions

## Example Usage

1. **Start the server:**
   ```bash
   python app.py
   ```

2. **Open browser and go to:** `http://localhost:5000`

3. **Paste a news article or headline**

4. **Click "Check News" to get prediction**

## Testing

Run the test script to verify model functionality:
```bash
python test_model.py
```

This will test the model with sample texts and show predictions with confidence scores. 