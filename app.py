from flask import Flask, request, jsonify
import os
"""is a security function provided by the Werkzeug library (which Flask uses internally).
It is used to sanitize uploaded filenames so they are safe to save on your server."""
from werkzeug.utils import secure_filename
from utils.image_utils import allowed_file
from model import load_model, predict_image

UPLOAD_FOLDER = "uploads"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model once at startup
#Loading a TensorFlow model is slow.
#we do NOT want to reload it every time someone calls /predict.
#So the model is loaded only one time when the server starts.
#If loading fails, model = None, and the API will respond with an error.
try:
    model = load_model()
except Exception as e:
    model = None
    print("ERROR: Model failed to load:", e)

#this is the home route
@app.route('/')
def home():
    return jsonify({"message": "Cat vs Dog API running"})

#this is for prediction
@app.route('/predict', methods=['POST'])
def predict():
    #checking if the model is uploaded
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    #checking files
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    #checking if the file is empty
    if file.filename == '':
        return jsonify({"error": "File has no name"}), 400

    #checking the extension
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: jpg, jpeg, png"}), 400

    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # File size check
    if os.path.getsize(filepath) > 5 * 1024 * 1024: #convert 5mb to bytes
        return jsonify({"error": "File too large (max 5MB)"}), 400

    # Prediction
    try:
        label, confidence = predict_image(model, filepath)
    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

    return jsonify({
        "prediction": label,
        "confidence": confidence
    })


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(host="0.0.0.0", port=5000)
