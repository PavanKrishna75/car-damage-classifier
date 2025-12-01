
🚗 Car Damage Classification System

A full-stack machine learning application that classifies car damage types from uploaded images using a fine-tuned EfficientNet model. 

This project includes:

- A FastAPI backend for inference
- A TypeScript frontend for uploading images
- A full training pipeline built from scratch


🧠 Model Overview
- Architecture: EfficientNetB0
- Task: 6-class car damage classification
- Training Accuracy (Final Epoch): ~76.6%
- Best Validation Accuracy: ~82.7%
- Validation Accuracy (Final Epoch): ~82.3%
- Loss Function: Categorical Cross-Entropy
- Training Duration: 10 epochs
- Model Checkpoint: Saved automatically at best validation accuracy
- File Used for Inference: efficientnet_best_6labels.keras

⚙️ Tech Stack

Backend:
- FastAPI
- Uvicorn
- TensorFlow / Keras
- Pillow
- Python-Multipart

Frontend:
- TypeScript
- Vite

DeepLearning:
- EfficientNetB0
- Softmax classification head
- GPU-optimized training loop
- Data Augmentation

ML / Data:
- Pandas
- Scikit-Learn
- Matplotlib

🚀 Running the Application

1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

2. Install dependencies
pip install -r src/requirements.txt

3. Start the FastAPI backend
uvicorn api.app:app --reload

🖥️ Running the Frontend (TypeScript + Vite)

1. Navigate into the frontend folder
cd frontend

2. Install dependencies
npm install

3. Start the frontend
npm run dev

The frontend automatically interacts with the FastAPI backend.
No manual API calls or port numbers are required from the user.

The app opens in the browser and allows:
- Single or multiple image uploads
- Automatic POST request to backend
- Display of predictions with confidence scores


📊 Training Summary

Training Accuracy (Final Epoch): ~76.6%
Best Validation Accuracy: ~82.7%
Validation Accuracy (Final Epoch): ~82.3%
Training Loss: Decreased steadily
Best Validation Loss: ~1.17

🔮 Future Improvements

- Add batch image prediction
- Improve accuracy with more data augmentation
- Add Docker deployment
- Deploy to AWS / Render
- Add confusion matrix visualization


Author

Pavan Krishna Kottachalla
Car Damage Classification System
