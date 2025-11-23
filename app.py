from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React Native app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and preprocessors
model_artifacts = None
label_encoders = None

# ==================== HELPER FUNCTIONS ====================

def prepare_features(df):
    """Feature engineering pipeline"""
    
    # Time-based features
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
    df['Is_Night'] = df['Hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)
    df['Is_Weekend'] = pd.to_datetime(df['Date']).dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
    
    # Amount-based features
    df['Amount_Log'] = np.log1p(df['amount'])
    df['Is_High_Amount'] = (df['amount'] > 5000).astype(int)  # Fixed threshold
    df['Is_Round_Amount'] = (df['amount'] % 100 == 0).astype(int)
    
    # Velocity features
    df['High_Frequency'] = (df['Transaction_Frequency'] > 5).astype(int)  # Fixed threshold
    df['Abnormal_Deviation'] = (df['Transaction_Amount_Deviation'] > 50).astype(int)  # Fixed threshold
    
    # Risk Score
    df['Risk_Score'] = (
        df['Is_Night'] * 0.2 +
        df['Is_High_Amount'] * 0.3 +
        df['High_Frequency'] * 0.25 +
        df['Abnormal_Deviation'] * 0.25
    )
    
    return df

def load_model():
    """Load the trained model and artifacts"""
    global model_artifacts, label_encoders
    
    try:
        # Check if model file exists
        model_path = 'fraud_detection_model_one.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_artifacts = pickle.load(f)
                logger.info("Model loaded successfully from file")
        else:
            logger.warning("Model file not found, using mock predictions")
            model_artifacts = None
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_artifacts = None

def get_mock_prediction(transaction_data):
    """Generate mock prediction when model is not available"""
    amount = float(transaction_data.get('amount', 0))
    frequency = int(transaction_data.get('Transaction_Frequency', 1))
    deviation = float(transaction_data.get('Transaction_Amount_Deviation', 0))
    
    # Simple mock logic based on risk factors
    fraud_probability = 0.1
    
    # Amount-based risk
    if amount > 5000:
        fraud_probability += 0.3
    elif amount > 2000:
        fraud_probability += 0.15
    
    # Frequency-based risk
    if frequency > 5:
        fraud_probability += 0.2
    elif frequency > 3:
        fraud_probability += 0.1
    
    # Deviation-based risk
    if deviation > 50:
        fraud_probability += 0.15
    
    # Transaction type risk
    if transaction_data.get('Transaction_Type') == 'Online':
        fraud_probability += 0.1
    
    # Channel risk
    if transaction_data.get('Transaction_Channel') == 'Web':
        fraud_probability += 0.05
    
    # Time risk (if provided)
    try:
        hour = pd.to_datetime(transaction_data.get('Time', '12:00:00'), format='%H:%M:%S').hour
        if hour < 6 or hour > 22:
            fraud_probability += 0.1
    except:
        pass
    
    # Cap the probability
    fraud_probability = min(fraud_probability, 0.95)
    
    # Determine risk level
    if fraud_probability >= 0.7:
        risk_level = 'HIGH'
    elif fraud_probability >= 0.3:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'
    
    return {
        'fraud_prediction': 1 if fraud_probability > 0.5 else 0,
        'fraud_probability': fraud_probability,
        'risk_level': risk_level
    }

# ==================== API ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_artifacts is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Get request data
        data = request.json
        logger.info(f"Received prediction request for amount: ${data.get('amount', 0)}")
        
        # Use mock prediction if model not loaded
        if model_artifacts is None:
            logger.warning("Using mock prediction as model is not loaded")
            result = get_mock_prediction(data)
            return jsonify(result)
        
        # Create DataFrame from input
        df = pd.DataFrame([data])
        
        # Apply feature engineering
        df = prepare_features(df)
        
        # Handle categorical encoding
        categorical_features = ['Transaction_Type', 'Payment_Gateway', 'Transaction_Status', 
                              'Device_OS', 'Merchant_Category', 'Transaction_Channel',
                              'Transaction_City', 'Transaction_State']
        
        for col in categorical_features:
            if col in df.columns and col in model_artifacts['label_encoders']:
                try:
                    le = model_artifacts['label_encoders'][col]
                    # Handle unknown categories
                    df[f'{col}_Encoded'] = df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
                except:
                    df[f'{col}_Encoded'] = 0
            else:
                df[f'{col}_Encoded'] = 0
        
        # Select features
        feature_cols = model_artifacts['feature_cols']
        
        # Ensure all required features exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        X = df[feature_cols]
        
        # Scale features
        X_scaled = model_artifacts['scaler'].transform(X)
        
        # Make prediction
        prediction = model_artifacts['model'].predict(X_scaled)[0]
        probability = model_artifacts['model'].predict_proba(X_scaled)[0, 1]
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = 'HIGH'
        elif probability >= 0.3:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        result = {
            'fraud_prediction': int(prediction),
            'fraud_probability': float(probability),
            'risk_level': risk_level
        }
        
        logger.info(f"Prediction result: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        # Return mock prediction on error
        return jsonify(get_mock_prediction(request.json))


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,
        debug=False  # Set to False in production
    )
