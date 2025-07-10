#!/usr/bin/env python3
"""
Test script to verify model loading and prediction functionality
"""

import joblib
import pandas as pd
import os
import numpy as np

def test_model_loading():
    """Test if the model and encoders load correctly"""
    print("Testing model loading...")
    
    MODEL_DIR = 'track_prediction_model'
    
    try:
        # Load the model
        model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
        print("✓ Model loaded successfully")
        
        # Load label encoders
        encoders = {}
        encoder_files = [
            'le_Gender.pkl', 'le_Faculty.pkl', 'le_Certification_Field.pkl',
            'le_Certification_Soutce.pkl', 'le_Company_Type.pkl', 'le_Jop_Category.pkl',
            'le_Jop_Title.pkl', 'le_Experience_Level.pkl'
        ]
        
        for encoder_file in encoder_files:
            feature_name = encoder_file.replace('le_', '').replace('.pkl', '')
            encoders[feature_name] = joblib.load(os.path.join(MODEL_DIR, encoder_file))
            print(f"✓ Encoder for {feature_name} loaded successfully")
        
        # Load selected features
        with open(os.path.join(MODEL_DIR, 'selected_features.txt'), 'r') as f:
            selected_features = [line.strip() for line in f.readlines()]
        print(f"✓ Selected features loaded: {selected_features}")
        
        return model, encoders, selected_features
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None, None

def test_prediction(model, encoders, selected_features):
    """Test making a prediction with sample data"""
    print("\nTesting prediction...")
    
    try:
        # Create sample input data
        sample_data = pd.DataFrame([{
            'Gender': 'M',
            'Faculty': 'Computer Science',
            'Certification_Field': 'Machine Learning',
            'Certification_Soutce': 'Coursera',
            'Company_Type': 'Local',
            'Jop_Category': 'Machine Learning',
            'Experience_Level': 'Entry Level'
        }])
        
        # Add derived features
        sample_data['Job_Cert_Alignment'] = (
            sample_data['Certification_Field'].str.contains('Development|Engineering|Science|Analysis', case=False, na=False)
        ).astype(int)
        
        sample_data['Is_Egypt'] = 1
        
        # Encode categorical features
        encoded_data = pd.DataFrame()
        
        for feature in selected_features:
            if feature in sample_data.columns and feature in encoders:
                encoded_data[f'{feature}_encoded'] = encoders[feature].transform(sample_data[feature])
            else:
                if feature == 'Job_Cert_Alignment':
                    encoded_data[f'{feature}_encoded'] = sample_data[feature]
                elif feature == 'Is_Egypt':
                    encoded_data[f'{feature}_encoded'] = sample_data[feature]
        
        # Ensure correct column order
        feature_columns = [f'{feature}_encoded' for feature in selected_features]
        X = encoded_data[feature_columns]
        
        # Make prediction
        prediction_encoded = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        
        # Get predicted job title
        predicted_job_title = encoders['Jop_Title'].inverse_transform([prediction_encoded])[0]
        confidence = prediction_proba[prediction_encoded] * 100
        
        print(f"✓ Prediction successful!")
        print(f"  Sample input: Male, Computer Science, Machine Learning, Coursera, Local, Machine Learning, Entry Level")
        print(f"  Predicted job title: {predicted_job_title}")
        print(f"  Confidence: {confidence:.2f}%")
        
        # Show top 3 predictions
        top_3_indices = np.argsort(prediction_proba)[-3:][::-1]
        print(f"  Top 3 predictions:")
        for i, idx in enumerate(top_3_indices):
            job_title = encoders['Jop_Title'].inverse_transform([idx])[0]
            prob = prediction_proba[idx] * 100
            print(f"    {i+1}. {job_title}: {prob:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Error making prediction: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("JOB TITLE PREDICTION MODEL TEST")
    print("=" * 50)
    
    # Test model loading
    model, encoders, selected_features = test_model_loading()
    
    if model is not None:
        # Test prediction
        success = test_prediction(model, encoders, selected_features)
        
        if success:
            print("\n" + "=" * 50)
            print("✓ ALL TESTS PASSED!")
            print("The model is ready to use in the Flask application.")
            print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print("✗ PREDICTION TEST FAILED!")
            print("Please check the model files and try again.")
            print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("✗ MODEL LOADING FAILED!")
        print("Please ensure all model files are present in the track_prediction_model directory.")
        print("=" * 50)

if __name__ == "__main__":
    main() 