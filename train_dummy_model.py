import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import os

def create_and_save_model():
    np.random.seed(42)
    # Generate 1000 synthetic samples
    n_samples = 1000
    
    # 13 Features based on typical heart disease dataset
    data = {
        'age': np.random.randint(29, 78, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.normal(131, 17, n_samples),
        'chol': np.random.normal(246, 51, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.normal(149, 22, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.exponential(1.0, n_samples),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(1, 4, n_samples)
    }
    
    # Target variable (0 = Low Risk, 1 = High Risk)
    # Give some logical correlation
    target = (
        (data['age'] > 55).astype(int) + 
        (data['cp'] > 1).astype(int) + 
        (data['thalach'] < 130).astype(int) + 
        (data['exang'] == 1).astype(int) + 
        (data['oldpeak'] > 1.5).astype(int)
    )
    # If 3 or more risk factors, label as 1 (high risk)
    y = (target >= 2).astype(int)
    
    X = pd.DataFrame(data)
    
    # Create pipeline with scaler and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train
    pipeline.fit(X, y)
    
    # Save the pipeline
    model_path = os.path.join(os.path.dirname(__file__), 'heart_disease_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
        
    print(f"Successfully created and saved model to {model_path}")

if __name__ == "__main__":
    create_and_save_model()
