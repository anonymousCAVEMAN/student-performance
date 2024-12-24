from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from src.pipelines.predict_pipeline import PredictPipeline

app = FastAPI()

# Define Pydantic model
class LoanApplication(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int

@app.post("/predict")
async def predict_loan_application(input_data: LoanApplication):
    try:
        # Convert Pydantic model to DataFrame
        pred_df = pd.DataFrame([input_data.dict()])
        
        # Run prediction pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return {"prediction": results[0]}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
