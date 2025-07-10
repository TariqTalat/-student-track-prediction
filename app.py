from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

MODEL_DIR = 'track_prediction_model'

# Load model, encoders, and feature selector
model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
feature_selector = joblib.load(os.path.join(MODEL_DIR, 'feature_selector.pkl'))
le_target = joblib.load(os.path.join(MODEL_DIR, 'le_target.pkl'))

# Load all feature encoders
encoder_files = [
    'le_Gender.pkl', 'le_Faculty.pkl', 'le_Certification_Field.pkl',
    'le_Certification_Soutce.pkl', 'le_Company_Type.pkl', 'le_Company_location.pkl',
    'le_Jop_Category.pkl', 'le_Jop_Title.pkl', 'le_Experience_Level.pkl'
]
encoders = {}
for encoder_file in encoder_files:
    feature_name = encoder_file.replace('le_', '').replace('.pkl', '')
    encoders[feature_name] = joblib.load(os.path.join(MODEL_DIR, encoder_file))

# Load selected features
with open(os.path.join(MODEL_DIR, 'selected_features.txt'), 'r') as f:
    selected_features = [line.strip() for line in f.readlines() if line.strip()]

# Dropdown options (extracted from data or hardcoded for demo)
DROPDOWN_OPTIONS = {
    'Gender': ['M', 'F'],
    'Faculty': ['Computer Science', 'Engineering', 'Commerce', 'Medicine', 'Pharmacy', 'Arts', 'Science', 'Law'],
    'Certification_Field': [
        'Backend Development', 'Database Developmnet', 'Graphic Design', 'Network', 'Blockchain',
        'Data Engineering', 'Machine Learning', 'Deep learning', 'NLP', 'Database Management',
        'Software Engineering', 'Frontend Development', 'Full Stack Development', 'Data Analysis',
        'Web Development', 'Computer Vision', 'Data Science', 'AI', 'Mobile Development',
        'Cybersecurity', 'DevOps'
    ],
    'Certification_Soutce': ['LinkedIn Learning', 'DataCamp', 'Codecademy', 'Coursera', 'Pluralsight', 'Udacity', 'edX'],
    'Company_Type': ['Local', 'International', 'National', 'Multinational'],
    'Company_location': ['Egypt', 'USA', 'KSA', 'England', 'Tunisa', 'Farance', 'Canda', 'UEA', 'Morocco'],
    'Jop_Category': [
        'Cybersecurity', 'Machine Learning', 'Web Development', 'Data Analysis', 'Video Editing',
        'UI/UX Design', 'Content Writing', 'Digital Marketing', 'Mobile App Development', 'Graphic Design'
    ],
    'Jop_Title': [
        'Backend Developer', 'Cloud Engineer', 'Data Analyst', 'Data Scientist', 'Database Administrator',
        'DevOps Engineer', 'Frontend Developer', 'Full Stack Developer', 'IT Support Specialist',
        'Machine Learning Engineer', 'Mobile Developer', 'QA Engineer', 'Security Analyst',
        'Software Engineer', 'System Administrator'
    ]
}

@app.route('/')
def index():
    return render_template('index.html', options=DROPDOWN_OPTIONS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # User input features
        user_input = {
            'Gender': data['gender'],
            'Faculty': data['faculty'],
            'Certification_Field': data['certification_field'],
            'Certification_Soutce': data['certification_source'],
            'Company_Type': data['company_type'],
            'Company_location': data['company_location'],
            'Jop_Category': data['job_category'],
            'Jop_Title': data['job_title']
        }
        
        # Engineered features as per the notebook
        # Cert_Faculty_Match: (Certification_Field contains tech keywords AND Faculty contains Computer/Engineering)
        cert_tech_keywords = ['Computer', 'Data', 'AI', 'ML', 'Backend', 'Frontend', 'Database', 
                             'Development', 'Engineering', 'Science', 'Analysis']
        faculty_tech_keywords = ['Computer', 'Engineering']
        
        user_input['Cert_Faculty_Match'] = int(
            any(keyword in user_input['Certification_Field'] for keyword in cert_tech_keywords) and
            any(keyword in user_input['Faculty'] for keyword in faculty_tech_keywords)
        )
        
        # Job_Cert_Alignment: (Jop_Category == Certification_Field)
        user_input['Job_Cert_Alignment'] = int(user_input['Jop_Category'] == user_input['Certification_Field'])
        
        # Is_Egypt: (Company_location == 'Egypt')
        user_input['Is_Egypt'] = int(user_input['Company_location'] == 'Egypt')
        
        # Experience_Level: derived from Jop_Title
        title = user_input['Jop_Title'].lower()
        if any(x in title for x in ['junior', 'assistant', 'entry']):
            user_input['Experience_Level'] = 'Junior'
        elif any(x in title for x in ['senior', 'lead', 'manager', 'specialist']):
            user_input['Experience_Level'] = 'Senior'
        elif any(x in title for x in ['developer', 'analyst', 'engineer']):
            user_input['Experience_Level'] = 'Mid'
        else:
            user_input['Experience_Level'] = 'Junior'  # Default
        
        # Prepare DataFrame with all required features
        input_df = pd.DataFrame([user_input])
        
        # Encode categorical features
        for col in input_df.columns:
            if col in encoders and input_df[col].dtype == 'object':
                try:
                    input_df[col] = encoders[col].transform(input_df[col].astype(str))
                except Exception as e:
                    print(f"Warning: Error encoding {col}: {e}")
                    input_df[col] = 0  # Unknown category fallback
        
        # Ensure ALL original features are present and in correct order
        feature_columns = [
            'Gender', 'Faculty', 'Certification_Field', 'Certification_Soutce',
            'Company_Type', 'Company_location', 'Jop_Category', 'Jop_Title',
            'Cert_Faculty_Match', 'Job_Cert_Alignment', 'Is_Egypt', 'Experience_Level'
        ]
        for feature in feature_columns:
            if feature not in input_df.columns:
                input_df[feature] = 0
        input_df = input_df[feature_columns]
        
        # Feature selection
        input_selected = feature_selector.transform(input_df)
        
        # Predict
        pred_encoded = model.predict(input_selected)[0]
        pred_proba = model.predict_proba(input_selected)[0]
        pred_track = le_target.inverse_transform([pred_encoded])[0]
        
        # Top 3 predictions
        top3_idx = np.argsort(pred_proba)[-3:][::-1]
        top3_names = []
        for idx in top3_idx:
            try:
                name = le_target.inverse_transform([idx])[0]
            except Exception as e:
                print(f'Error mapping idx {idx}: {e}')
                name = f'Class_{idx}'
            top3_names.append(name)
        print('le_target.classes_:', le_target.classes_)
        print('Top3 idx:', top3_idx)
        print('Top3 names:', top3_names)
        top3 = [
            {
                'track': name,
                'probability': round(100 * pred_proba[idx], 2)
            }
            for idx, name in zip(top3_idx, top3_names)
        ]
        
        confidence = round(100 * pred_proba[pred_encoded], 2)
        
        return jsonify({
            'success': True,
            'predicted_track': pred_track,
            'confidence': confidence,
            'top_3_predictions': top3,
            'input_data': data
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/options')
def get_options():
    return jsonify(DROPDOWN_OPTIONS)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 