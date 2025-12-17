# 🧠 Fake News Detector – NLP-Based Headline Classifier (Project code is uploaded in the master branch)


An AI-powered system that classifies news headlines as **real or fake** using a fine-tuned BERT transformer model. Built with a focus on **accuracy, scalability, and real-time inference**.

---

## ✨ Features

- **High Accuracy**  
  Achieves **94.97%** accuracy on validation data.

- **Real-Time Classification**  
  Flask-based web app for instant headline analysis.

- **Scalable NLP Pipeline**  
  Automated preprocessing and labeling of 20k+ headlines.

- **Transformer Fine-Tuning**  
  Custom-trained BERT model optimized for precision and recall.

- **Deployment Ready**  
  REST API endpoint for easy integration into other systems.

---

## 🛠️ Tech Stack

- Python  
- Hugging Face Transformers (BERT)  
- Flask (Backend API)  
- Scikit-learn (Metrics & utilities)  
- Pandas, NumPy (Data processing)  
- HTML/CSS/JavaScript (Frontend interface)  

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/amrrrsoliman/Fake-News-Detection.git
cd Fake-News-Detection
pip install -r requirements.txt



🤖 Model Details
Base Model: BERT (Bidirectional Encoder Representations from Transformers)

Dataset: Custom-trained on 20,000+ labeled headlines

Performance Metrics:

Accuracy: 94.97%

Optimized for precision and recall to minimize false positives/negatives

Training Features: Hyperparameter tuning, supervised fine-tuning, and validation splitting


🔌 API Endpoint
You can also use the model via a direct API call:

bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"headline": "Your news headline here"}'
Response:

json
{
  "prediction": "Real",
  "confidence": 0.92
}
📈 Future Improvements
Expand dataset with multi-language support

Implement ensemble models (BERT + RoBERTa)

Add explainable AI (XAI) features to show classification reasoning

🧾 License
This project is for educational and research purposes. Please cite appropriately if used in academic work.

Code

---

✅ This version makes it crystal clear:
- The model is in the file `Fake_News_Detection_95`.  
- That file contains both the **view link** and the **direct download link**.  
- Recruiters or collaborators can easily find and access the model without confusion.  

Would you like me to now **standardize this same “Model Location” section** for your S
