import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import random

app = FastAPI(title="Rock Paper Scissors ML Backend", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    model_data = joblib.load("model.joblib")
    model = model_data["model"]
    labels = model_data["labels"] # ['Paper', 'Scissor', 'Stone']
    accuracy = model_data.get("accuracy", 0.9887)
    print(f"Loaded model successfully. Accuracy: {accuracy*100:.2f}%. Labels: {labels}")
except Exception as e:
    print("Error loading model.joblib:", e)
    model = None
    labels = ['Paper', 'Scissor', 'Stone']
    accuracy = 0.9887

# Map raw label to standard game names
LABEL_MAP = {
    "Stone": "Rock",
    "Paper": "Paper",
    "Scissor": "Scissors"
}

REVERSE_LABEL_MAP = {
    "Rock": "Stone",
    "Paper": "Paper",
    "Scissors": "Scissor"
}

def normalize_landmarks(raw_landmarks: List[float]) -> np.ndarray:
    """Normalize 42 floats (21 pairs of x,y) relative to wrist."""
    if len(raw_landmarks) != 42:
        raise ValueError(f"Expected 42 landmark values, got {len(raw_landmarks)}")
    
    pts = np.array(raw_landmarks, dtype=np.float32).reshape(21, 2)
    wrist = pts[0].copy()
    
    # Center relative to wrist
    pts_centered = pts - wrist
    
    # Scale normalization
    norm = np.linalg.norm(pts_centered)
    if norm > 0:
        pts_centered = pts_centered / norm
        
    return pts_centered.flatten().reshape(1, -1)

class PredictRequest(BaseModel):
    landmarks: List[float]  # 42 float values: x0, y0, x1, y1 ... x20, y20

class PlayRequest(BaseModel):
    landmarks: Optional[List[float]] = None
    player_move: Optional[str] = None  # "Rock", "Paper", or "Scissors"
    difficulty: Optional[str] = "normal"  # "normal", "smart", "impossible"

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "accuracy": accuracy
    }

@app.get("/api/info")
def model_info():
    return {
        "dataset_samples": 45050,
        "categories": ["Rock", "Paper", "Scissors"],
        "raw_categories": labels,
        "model_type": "Multi-Layer Perceptron Neural Network / Random Forest",
        "accuracy": accuracy,
        "input_features": 42,
        "landmark_points": 21
    }

@app.post("/api/predict")
def predict_gesture(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="ML model is not loaded")
    
    try:
        features = normalize_landmarks(req.landmarks)
        probs = model.predict_proba(features)[0]
        pred_idx = np.argmax(probs)
        raw_label = labels[pred_idx]
        game_move = LABEL_MAP.get(raw_label, raw_label)
        
        prob_dict = {}
        for idx, lbl in enumerate(labels):
            move_name = LABEL_MAP.get(lbl, lbl)
            prob_dict[move_name] = float(probs[idx])
            
        return {
            "prediction": game_move,
            "raw_label": raw_label,
            "confidence": float(probs[pred_idx]),
            "probabilities": prob_dict
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/play")
def play_round(req: PlayRequest):
    player_move = req.player_move
    confidence = 1.0
    probabilities = {}

    if req.landmarks:
        pred_res = predict_gesture(PredictRequest(landmarks=req.landmarks))
        player_move = pred_res["prediction"]
        confidence = pred_res["confidence"]
        probabilities = pred_res["probabilities"]

    if not player_move or player_move not in ["Rock", "Paper", "Scissors"]:
        raise HTTPException(status_code=400, detail="Invalid player move or landmarks missing")

    # Determine AI move based on difficulty
    moves = ["Rock", "Paper", "Scissors"]
    counter_moves = {
        "Rock": "Paper",
        "Paper": "Scissors",
        "Scissors": "Rock"
    }
    
    if req.difficulty == "impossible":
        ai_move = counter_moves[player_move]
    elif req.difficulty == "smart":
        # 60% chance to counter player, 40% random
        if random.random() < 0.6:
            ai_move = counter_moves[player_move]
        else:
            ai_move = random.choice(moves)
    else:
        ai_move = random.choice(moves)

    # Determine winner
    if player_move == ai_move:
        result = "draw"
        message = "It's a Tie!"
    elif counter_moves[player_move] == ai_move:
        result = "lose"
        message = f"{ai_move} beats {player_move}. AI Wins!"
    else:
        result = "win"
        message = f"{player_move} beats {ai_move}. You Win!"

    return {
        "player_move": player_move,
        "ai_move": ai_move,
        "result": result,
        "message": message,
        "confidence": confidence,
        "probabilities": probabilities
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
