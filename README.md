# Job Title Prediction System

An AI-powered web application that predicts job titles based on user profile information using machine learning.

## Features

- **Modern UI**: Beautiful, responsive web interface with gradient backgrounds and smooth animations
- **Real-time Prediction**: Instant job title predictions with confidence scores
- **Top 3 Predictions**: Shows the top 3 most likely job titles with probabilities
- **Form Validation**: Client-side validation with visual feedback
- **Error Handling**: Comprehensive error handling and user feedback
- **Mobile Responsive**: Works perfectly on desktop, tablet, and mobile devices

## Model Information

The application uses a **Random Forest Classifier** trained on a dataset of 1,903 student profiles with the following features:

- **Gender**: Male/Female
- **Faculty**: Academic background (Computer Science, Engineering, etc.)
- **Certification Field**: Area of certification (Backend Development, Data Science, etc.)
- **Certification Source**: Platform where certification was obtained
- **Company Type**: Type of company (Local, International, National, Multinational)
- **Job Category**: Category of job (Cybersecurity, Machine Learning, etc.)
- **Experience Level**: Professional experience level
- **Job-Certification Alignment**: Derived feature indicating alignment between job and certification
- **Is_Egypt**: Geographic indicator

## Predicted Job Titles

The model can predict the following job titles:
- Backend Developer
- Cloud Engineer
- Data Analyst
- Data Scientist
- Database Administrator
- DevOps Engineer
- Frontend Developer
- Full Stack Developer
- IT Support Specialist
- Machine Learning Engineer
- Mobile Developer
- QA Engineer
- Security Analyst
- Software Engineer
- System Administrator

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the model files are in place**:
   - `track_prediction_model/best_model.pkl` - The trained model
   - `track_prediction_model/le_*.pkl` - Label encoders for each feature
   - `track_prediction_model/selected_features.txt` - List of features used by the model

## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Fill in the form** with your profile information:
   - Select your gender
   - Choose your faculty/educational background
   - Pick your certification field
   - Select the certification source/platform
   - Choose the company type you're targeting
   - Select your desired job category
   - Pick your experience level

4. **Click "Predict Job Title"** to get your AI-powered career recommendation

## API Endpoints

### GET `/`
- **Description**: Main application page
- **Response**: HTML page with the prediction form

### POST `/predict`
- **Description**: Make a job title prediction
- **Request Body**: JSON with user profile data
- **Response**: JSON with prediction results including:
  - Predicted job title
  - Confidence score
  - Top 3 predictions with probabilities

### GET `/api/options`
- **Description**: Get available dropdown options
- **Response**: JSON with all available options for each field

## Technical Architecture

- **Backend**: Flask web framework
- **Frontend**: HTML5, CSS3, JavaScript with Bootstrap 5
- **Machine Learning**: Scikit-learn Random Forest Classifier
- **Data Processing**: Pandas for data manipulation
- **Model Persistence**: Joblib for model serialization

## Model Performance

- **Accuracy**: ~70% on test data
- **Best performing job titles**:
  - IT Support Specialist: 96% precision/recall
  - Frontend Developer: 90% precision, 96% recall
  - DevOps Engineer: 68% precision, 77% recall

## File Structure

```
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── templates/
│   └── index.html                 # Main HTML template
└── track_prediction_model/        # Model files
    ├── best_model.pkl             # Trained model
    ├── le_*.pkl                   # Label encoders
    └── selected_features.txt      # Feature list
```

## Error Handling

The application includes comprehensive error handling for:
- Missing or invalid form data
- Unseen categories in the model
- Model loading errors
- Network/API errors

## Security Features

- Input validation and sanitization
- CSRF protection (Flask built-in)
- Secure model loading
- Error message sanitization

## Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and demonstration purposes.

## Support

For issues or questions, please check the error logs or contact the development team. 