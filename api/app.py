from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from src.infer import DamageClassifier


app = FastAPI(title="Car Damage Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = None


@app.on_event("startup")
def load_model():
    global classifier
    classifier = DamageClassifier()
    print("Model loaded")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if classifier is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)

    content = await file.read()

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    label, confidence = classifier.predict(image)
    return {"label": label, "confidence": confidence}
