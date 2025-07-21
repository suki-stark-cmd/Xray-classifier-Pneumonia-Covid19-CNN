# � COVID-19 & Pneumonia Detection System

An AI-powered medical imaging system for detecting COVID-19 and pneumonia from chest X-ray images.

## 🎯 Features

- **3-Class Classification**: Normal, Viral Pneumonia, COVID+ detection
- **High-Performance Model**: 15.3M parameter deep learning model
- **Web Interface**: User-friendly Flask web application
- **Real-time Predictions**: Upload X-ray images and get instant results
- **Medical Guidance**: Specific advice for each condition
- **RESTful API**: Easy integration with other systems

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

### 3. Access the Web Interface
Open your browser and go to: `http://localhost:5000`

## 🎯 Model Specifications

- **Architecture**: Deep Convolutional Neural Network (15.3M parameters)
- **Input**: 224×224×3 RGB chest X-ray images
- **Output**: 3-class classification
  - 0: Normal
  - 1: Viral Pneumonia  
  - 2: COVID+
- **Model File**: `models/Covid.h5` (66MB)

## 🏗️ Project Structure

```
xray-classifier/
├── app.py                              # Flask web application
├── models/Covid.h5                     # Trained model (66MB)
├── static/                             # Web assets
├── templates/                          # HTML templates
├── Dataset/                            # Training data
├── COVID_Pneumonia_Detection.ipynb     # Documentation notebook
├── requirements.txt                    # Dependencies
└── README.md                           # Project documentation
```

## � API Usage

### Prediction Endpoint
```http
POST /predict
Content-Type: multipart/form-data

Parameters:
- file: Chest X-ray image (PNG/JPG/JPEG, max 5MB)
```

### Response Format
```json
{
  "prediction": "Normal",
  "confidence": 87.45,
  "risk_level": "High Confidence",
  "interpretation": "✅ No clear abnormalities detected",
  "medical_advice": "Continue regular health monitoring..."
}
```

### Health Check
```http
GET /health
```

## ⚠️ Medical Disclaimer

**This tool is for screening purposes only. Always consult healthcare professionals for medical diagnosis.**

### Medical Guidance by Classification:
- **Normal**: Continue regular health monitoring
- **Viral Pneumonia**: Immediate medical consultation for antiviral treatment
- **COVID+**: Immediate isolation and contact healthcare professionals

## � Technical Specifications

- **Backend**: Flask (Python)
- **AI Model**: Deep Convolutional Neural Network
- **Input**: 224×224×3 RGB chest X-ray images
- **Output**: 3-class classification with confidence scores
- **Framework**: TensorFlow/Keras
- **Parameters**: 15,305,027 total parameters

## �️ Safety Features

✅ Input validation and sanitization  
✅ File size limits (5MB max)  
✅ Secure file handling  
✅ Error handling and logging  
✅ Medical disclaimer and guidance  

---

**Built with ❤️ for medical AI research and education**
- [ ] Add batch processing capabilities

## 📁 Dataset Recommendations

For better performance, consider using these datasets:
1. **COVID-19 Radiography Database** (21K images, well-balanced)
2. **ChestX-ray14 NIH** (112K images, 14 pathologies)
3. **CheXpert** (224K images, Stanford quality)

## 🛠️ Development

- Model training: `FinalXray (1).ipynb`
- Web interface: `app.py` + `templates/`
- Model storage: `models/` directory

## 📝 License

This project is for educational and research purposes.

---
Made with ❤️ for medical AI research
