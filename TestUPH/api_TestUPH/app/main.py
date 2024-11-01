from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle
import os
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the model during startup
model_path = os.path.join("model", "rndf_regression_model.pkl")
if not os.path.exists(model_path):
    raise HTTPException(status_code=500, detail="Model file not found")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Define the input data schema
class NewData(BaseModel):
    TESTER_ID: str
    handler_id: str
    product_no: str
    QTY_IN: int
    QTY_OUT: int

# Define the API endpoint
@app.post("/predict/")
async def predict_uph(data: NewData):
    # Convert input data to DataFrame
    new_data = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(new_data)
    
    # Return the prediction
    return {"prediction": prediction.tolist()}

# Example of predicting UPH for new data
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)