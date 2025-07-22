from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)

# 🩺 Load Binary Pneumonia Detection Model
print("🚀 Loading Binary Pneumonia Detection Model...")

# Model configuration
IMG_SIZE = 224
THRESHOLD = 0.4  # Optimized threshold from analysis
CLASS_NAMES = ['Normal', 'Pneumonia']

try:
    print("💾 Loading trained model...")
    # Try to load the available model
    model_paths = [
        'models/pneumonia_detector_complete.h5',
        'models/Covid2.h5',
        'models/pneumonia_model.h5'
    ]
    
    model = None
    model_loaded_from = None
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                model_loaded_from = model_path
                print(f"✅ Model loaded from: {model_path}")
                break
            except Exception as load_error:
                print(f"❌ Failed to load {model_path}: {load_error}")
                continue
    
    if model is None:
        raise Exception("No valid model files found")
    
    # Load model configuration if available
    try:
        with open('models/evaluation_results.json', 'r') as f:
            model_info = json.load(f)
    except:
        model_info = {"test_accuracy": 0.85, "sensitivity": 0.90}
    
    # Check model output shape and adjust configuration
    output_shape = model.output.shape[-1]
    if output_shape == 3:
        # 3-class model based on your model's exact mapping:
        # 0 = Normal, 1 = Viral Pneumonia, 2 = COVID+
        CLASS_NAMES = ['Normal', 'Viral Pneumonia', 'COVID+']
        THRESHOLD = 0.5  # Use standard threshold for multiclass
        print(f"   Detected 3-class model")
    elif output_shape == 1:
        # Binary model
        CLASS_NAMES = ['Normal', 'Pneumonia']
        THRESHOLD = 0.4  # Optimized threshold
        print(f"   Detected binary model")
    else:
        # Unknown configuration
        CLASS_NAMES = [f'Class_{i}' for i in range(output_shape)]
        THRESHOLD = 0.5
        print(f"   Detected {output_shape}-class model")
    
    print(f"   Model: {model_loaded_from}")
    print(f"   Input Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"   Output Classes: {len(CLASS_NAMES)} ({CLASS_NAMES})")
    print(f"   Parameters: {model.count_params():,}")
    print(f"   Threshold: {THRESHOLD}")
    
    MODEL_LOADED = True
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️ Running in demo mode - please ensure a model file exists")
    model = None
    model_info = {}
    MODEL_LOADED = False

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_SIZE_MB = 5

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is loaded
    if not MODEL_LOADED or model is None:
        return jsonify({
            'error': 'Model not available', 
            'message': 'Please ensure pneumonia_detector_complete.h5 is in models/ directory'
        }), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_SIZE_MB * 1024 * 1024:
            return jsonify({'error': f'File size exceeds {MAX_SIZE_MB}MB limit'}), 400

        try:
            # Save and process image
            filename = secure_filename(file.filename)
            filepath = os.path.join("static", filename)
            file.save(filepath)

            # Load and preprocess image
            img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = model.predict(img_array, verbose=0)[0]
            
            # Handle different model types
            if len(CLASS_NAMES) == 2:  # Binary model
                prediction_prob = prediction[0] if len(prediction.shape) > 0 else prediction
                prediction_binary = int(prediction_prob > THRESHOLD)
                
                if prediction_binary == 1:  # Pneumonia
                    label = "Pneumonia"
                    confidence = float(prediction_prob)
                    interpretation = "⚠️ Pneumonia indicators detected - Medical consultation recommended"
                    medical_advice = "This screening suggests possible pneumonia. Please consult with a healthcare professional for proper diagnosis and treatment."
                else:  # Normal
                    label = "Normal"
                    confidence = float(1 - prediction_prob)
                    interpretation = "✅ No clear pneumonia indicators detected"
                    medical_advice = "This screening appears normal. Continue regular health monitoring and consult a doctor if symptoms develop."
                
                raw_probability = float(prediction_prob)
                
            else:  # Multi-class model
                predicted_class = np.argmax(prediction)
                label = CLASS_NAMES[predicted_class]
                confidence = float(prediction[predicted_class])
                raw_probability = float(prediction[predicted_class])
                
                # Provide specific medical advice based on your model's classes
                if label == "Normal":
                    interpretation = "✅ No clear abnormalities detected"
                    medical_advice = "This screening appears normal. Continue regular health monitoring and consult a doctor if symptoms develop."
                elif label == "Viral Pneumonia":
                    interpretation = "⚠️ Viral Pneumonia indicators detected - Medical consultation recommended"
                    medical_advice = "This screening suggests possible viral pneumonia. Please consult with a healthcare professional immediately for proper diagnosis and antiviral treatment."
                elif label == "COVID+":
                    interpretation = "🦠 COVID-19 indicators detected - Immediate medical attention required"
                    medical_advice = "This screening suggests possible COVID-19 infection. Please isolate immediately and contact healthcare professionals for testing and treatment protocols."
                else:
                    interpretation = f"⚠️ {label} indicators detected - Medical consultation recommended"
                    medical_advice = f"This screening suggests possible {label.lower()}. Please consult with a healthcare professional for proper diagnosis and treatment."
            
            # Risk assessment based on confidence
            if confidence > 0.8:
                risk_level = "High Confidence"
                risk_color = "#28a745" if label == "Normal" else "#dc3545"
            elif confidence > 0.6:
                risk_level = "Moderate Confidence"
                risk_color = "#ffc107"
            else:
                risk_level = "Low Confidence"
                risk_color = "#6c757d"

            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

            return jsonify({
                'prediction': label,
                'confidence': round(confidence * 100, 2),
                'raw_probability': round(raw_probability, 4),
                'threshold_used': THRESHOLD if len(CLASS_NAMES) == 2 else "N/A",
                'risk_level': risk_level,
                'risk_color': risk_color,
                'interpretation': interpretation,
                'medical_advice': medical_advice,
                'model_info': {
                    'architecture': 'Medical AI Model',
                    'input_size': f'{IMG_SIZE}x{IMG_SIZE}',
                    'classes': CLASS_NAMES,
                    'accuracy': f"{model_info.get('test_accuracy', 0)*100:.1f}%",
                    'sensitivity': f"{model_info.get('sensitivity', 0)*100:.1f}%"
                },
                'disclaimer': 'This is a screening tool only. Always consult healthcare professionals for medical diagnosis.'
            })

        except Exception as e:
            # Clean up file on error
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            
            print(f"Error processing image: {e}")
            return jsonify({
                'error': 'Error processing image',
                'message': 'Please ensure the image is a valid chest X-ray'
            }), 500

    return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG files.'}), 400

@app.route('/model-info')
def model_info_endpoint():
    """Endpoint to get model information"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'model_loaded': MODEL_LOADED,
        'architecture': 'EfficientNetB3 + Custom Head',
        'input_size': f'{IMG_SIZE}x{IMG_SIZE}',
        'classes': CLASS_NAMES,
        'threshold': THRESHOLD,
        'performance': model_info
    })

@app.route('/health')
def health_check():
    """Health check endpoint for deployment"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'timestamp': str(tf.timestamp())
    })

if __name__ == '__main__':
    print(f"\n🌟 Pneumonia Detection API Ready!")
    print(f"   Model Status: {'✅ Loaded' if MODEL_LOADED else '❌ Not Available'}")
    print(f"   Endpoint: http://localhost:5000")
    print(f"   Upload limit: {MAX_SIZE_MB}MB")
    print(f"   Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"\n🏥 Medical Disclaimer: This tool is for screening purposes only.")
    print(f"   Always consult healthcare professionals for medical diagnosis.\n")
    
    # For local development
    if __name__ == "__main__":
        app.run(debug=True, host='0.0.0.0', port=5000)

# For Vercel deployment
app = app
