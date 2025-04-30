import os
import pickle
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from .forms import PredictionForm


def load_model():
    """Load the trained model from the file"""
    model_path = os.path.join(os.path.dirname(__file__), 'ml_models', 'best_lung_cancer_model.pkl')
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except (FileNotFoundError, IOError):
        # Return None if the model file is not found
        return None


def predict_lung_cancer_risk(data_dict):
    """Make prediction using the loaded model"""
    # Load the model
    model_data = load_model()
    if not model_data:
        return None, None
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    # Process the input data
    processed_dict = data_dict.copy()
    
    # Convert gender if it's a string
    if 'GENDER' in processed_dict and isinstance(processed_dict['GENDER'], str):
        gender_mapping = {'M': 1, 'F': 0, 'MALE': 1, 'FEMALE': 0, 'Male': 1, 'Female': 0, 'm': 1, 'f': 0}
        processed_dict['GENDER'] = gender_mapping.get(processed_dict['GENDER'], processed_dict['GENDER'])
    
    # Handle spaces in column names
    rename_dict = {}
    for key in list(processed_dict.keys()):
        new_key = key.strip().replace(" ", "_")
        if new_key != key:
            rename_dict[key] = new_key
    
    # Rename keys with spaces
    for old_key, new_key in rename_dict.items():
        processed_dict[new_key] = processed_dict.pop(old_key)
    
    # Create a DataFrame for the input data
    input_df = pd.DataFrame([processed_dict])
    
    # Ensure the input has all required features
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Keep only the features used during training
    input_df = input_df[feature_names]
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    return prediction, probability


def index(request):
    """View for the home page with the prediction form"""
    form = PredictionForm()
    return render(request, 'predictor/index.html', {'form': form})


def predict(request):
    """Handle the prediction request and return results"""
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Process the valid form data
            data = {
                'GENDER': form.cleaned_data['gender'],
                'AGE': form.cleaned_data['age'],
                'SMOKING': int(form.cleaned_data['smoking']),
                'YELLOW_FINGERS': int(form.cleaned_data['yellow_fingers']),
                'ANXIETY': int(form.cleaned_data['anxiety']),
                'PEER_PRESSURE': int(form.cleaned_data['peer_pressure']),
                'CHRONIC DISEASE': int(form.cleaned_data['chronic_disease']),
                'FATIGUE': int(form.cleaned_data['fatigue']),
                'ALLERGY': int(form.cleaned_data['allergy']),
                'WHEEZING': int(form.cleaned_data['wheezing']),
                'ALCOHOL CONSUMING': int(form.cleaned_data['alcohol_consuming']),
                'COUGHING': int(form.cleaned_data['coughing']),
                'SHORTNESS OF BREATH': int(form.cleaned_data['shortness_of_breath']),
                'SWALLOWING DIFFICULTY': int(form.cleaned_data['swallowing_difficulty']),
                'CHEST PAIN': int(form.cleaned_data['chest_pain'])
            }
            
            # Make prediction
            prediction, probability = predict_lung_cancer_risk(data)
            
            if prediction is None:
                result = {
                    'error': 'Model not found. Please train the model first.'
                }
            else:
                result = {
                    'prediction': int(prediction),
                    'probability': float(probability),
                    'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
                }
            
            return JsonResponse(result)
        else:
            return JsonResponse({'error': 'Invalid form data'}, status=400)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)
