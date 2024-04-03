from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

class ScoringItem(BaseModel):
    gender: int
    age: int
    year: int
    income: float
    incedence: int
    cgpa: float

# Load model files
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)
    
with open('depression.pkl','rb') as f:
    depression = pickle.load(f)

with open('anxiety.pkl','rb') as f:
    anxiety = pickle.load(f)

with open('adhd.pkl','rb') as f:
    adhd = pickle.load(f)

with open('doctor.pkl','rb') as f:
    doctor = pickle.load(f)

# Create prediction function
def predict(item):
    # Ensure consistency in feature names
    feature_mapping = {
        'gender': 'gender 0=female,1=male',
        'age': 'Age',
        'incedence': 'Major Incidence',
        'income': 'What is your Family income?',
        'year': 'Your current year of Study'
    }
    
    # Transform feature names in the input item
    transformed_item = {feature_mapping.get(k, k): v for k, v in item.dict().items()}
    
    # Create DataFrame from the transformed item
    df = pd.DataFrame([transformed_item])
    
    # Scale the input data
    testdf = scaler.transform(df)
    
    # Make predictions
    depression_score = depression.predict_proba(testdf)
    anxiety_score = anxiety.predict_proba(testdf)
    adhd_score = adhd.predict_proba(testdf)
    doctor_score = doctor.predict_proba(testdf)
    
    return {
        'depression': depression_score[0][1],
        'anxiety': anxiety_score[0][1],
        'adhd': adhd_score[0][1],
        'doctor': doctor_score[0][1]
    }

# Define route
@app.post('/api/predict')
async def predict_endpoint(item: ScoringItem):
    return predict(item)
