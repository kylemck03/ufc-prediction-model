from venv import logger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from fastapi.middleware.cors import CORSMiddleware
import logging


app = FastAPI()

#Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FighterStats(BaseModel):
    f_1: int
    num_rounds: int
    title_fight: int
    weight_class: int
    gender: int
    fighter_height_cm: float
    fighter_weight_lbs: float
    fighter_reach_cm: float
    fighter_stance: float
    fighter_w: int
    fighter_l: int
    fighter_d: int
    fighter_dob: float
    avg_ctrl_time: float
    avg_reversals: float
    avg_submission_att: float
    avg_takedown_succ: float
    avg_takedown_att: float
    avg_sig_strikes_att: float
    avg_total_strikes_succ: float
    avg_total_strikes_att: float
    avg_knockdowns: float
    avg_finish_time: float

class PredictionRequest(BaseModel):
    fighter1: FighterStats
    fighter2: FighterStats

class PredictionResponse(BaseModel):
    winner: str
    probability: float
    fighter1_probability: float
    fighter2_probability: float

class FightPredictor(nn.Module):
    def __init__(self):
        super(FightPredictor, self).__init__()
        self.fc1 = nn.Linear(23, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
# Load model and scaler
model = FightPredictor()
model.load_state_dict(torch.load('best_test_model(7).pth', map_location=torch.device('cpu')))
model.eval()
scaler = joblib.load('scaler.joblib')

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    print(f"Received request data: {request}")

    try:
        # Log the extracted data
        print("Processing fighter data...")
        fighter1_data = request.fighter1
        fighter2_data = request.fighter2
        print(f"Fighter 1 data: {fighter1_data}")
        print(f"Fighter 2 data: {fighter2_data}")

        # Add logs before and after model prediction
        logger.info("Making prediction...")
        # prediction logic...

        # Convert fighter stats to numpy arrays
        fighter1_data = np.array([[
            request.fighter1.f_1,
            request.fighter1.num_rounds,
            request.fighter1.title_fight,
            request.fighter1.weight_class,
            request.fighter1.gender,
            request.fighter1.fighter_height_cm,
            request.fighter1.fighter_weight_lbs,
            request.fighter1.fighter_reach_cm,
            request.fighter1.fighter_stance,
            request.fighter1.fighter_w,
            request.fighter1.fighter_l,
            request.fighter1.fighter_d,
            request.fighter1.fighter_dob,
            request.fighter1.avg_ctrl_time,
            request.fighter1.avg_reversals,
            request.fighter1.avg_submission_att,
            request.fighter1.avg_takedown_succ,
            request.fighter1.avg_takedown_att,
            request.fighter1.avg_sig_strikes_att,
            request.fighter1.avg_total_strikes_succ,
            request.fighter1.avg_total_strikes_att,
            request.fighter1.avg_knockdowns,
            request.fighter1.avg_finish_time
        ]])

        fighter2_data = np.array([[
            request.fighter2.f_1,
            request.fighter2.num_rounds,
            request.fighter2.title_fight,
            request.fighter2.weight_class,
            request.fighter2.gender,
            request.fighter2.fighter_height_cm,
            request.fighter2.fighter_weight_lbs,
            request.fighter2.fighter_reach_cm,
            request.fighter2.fighter_stance,
            request.fighter2.fighter_w,
            request.fighter2.fighter_l,
            request.fighter2.fighter_d,
            request.fighter2.fighter_dob,
            request.fighter2.avg_ctrl_time,
            request.fighter2.avg_reversals,
            request.fighter2.avg_submission_att,
            request.fighter2.avg_takedown_succ,
            request.fighter2.avg_takedown_att,
            request.fighter2.avg_sig_strikes_att,
            request.fighter2.avg_total_strikes_succ,
            request.fighter2.avg_total_strikes_att,
            request.fighter2.avg_knockdowns,
            request.fighter2.avg_finish_time
        ]])
        
        # Convert Numpy arrays to PyTorch tensors
        fighter1_scaled = scaler.transform(fighter1_data)
        fighter2_scaled = scaler.transform(fighter2_data)
        
        # Convert to PyTorch tensors
        fighter1_tensor = torch.Tensor(fighter1_scaled).float()
        fighter2_tensor = torch.Tensor(fighter2_scaled).float()
        
        # Get predictions
        with torch.no_grad():
            pred1 = model(fighter1_tensor)
            pred2 = model(fighter2_tensor)
            prob1 = torch.nn.functional.softmax(pred1, dim=1)[:, 1]
            prob2 = torch.nn.functional.softmax(pred2, dim=1)[:, 1]

        print(f"Fighter 1 probability BEFORE FLOAT: {prob1}")
        print(f"Fighter 2 probability BEFORE FLOAT: {prob2}")
        
        # Compare probabilities
        fighter1_prob = float(prob1.item())
        fighter2_prob = float(prob2.item())
        
        winner = "Fighter 1" if fighter1_prob > fighter2_prob else "Fighter 2"
        winning_prob = max(fighter1_prob, fighter2_prob)
        
        return PredictionResponse(
            winner=winner,
            probability=round(winning_prob * 100, 2),
            fighter1_probability=fighter1_prob * 100,
            fighter2_probability=fighter1_prob * 100
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)