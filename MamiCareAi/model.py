import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class PreeclampsiaPredictor:
    """Lightweight preeclampsia risk prediction model"""
    
    def __init__(self, model_path='models/preeclampsia_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.label_encoders = {}
        self.feature_names = [
            'systolic_bp', 'diastolic_bp', 'urine_protein_encoded', 
            'maternal_age', 'gravida', 'parity', 'history_preeclampsia_encoded',
            'gestational_age', 'symptom_count', 'high_risk_symptoms'
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained model and encoders"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.label_encoders = model_data['encoders']
                logging.info("Model loaded successfully")
            else:
                logging.warning(f"Model file not found at {self.model_path}. Please run train.py first.")
                # Create a simple rule-based fallback
                self.model = None
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            self.model = None
    
    def encode_features(self, patient_data):
        """Encode categorical features and extract symptom information"""
        features = {}
        
        # Direct numerical features
        features['systolic_bp'] = patient_data['systolic_bp']
        features['diastolic_bp'] = patient_data['diastolic_bp']
        features['maternal_age'] = patient_data['maternal_age']
        features['gravida'] = patient_data['gravida']
        features['parity'] = patient_data['parity']
        features['gestational_age'] = patient_data['gestational_age']
        
        # Encode urine protein
        protein_mapping = {'negative': 0, 'trace': 1, '1+': 2, '2+': 3, '3+': 4}
        features['urine_protein_encoded'] = protein_mapping.get(patient_data['urine_protein'], 0)
        
        # Encode history of preeclampsia
        features['history_preeclampsia_encoded'] = 1 if patient_data['history_preeclampsia'].lower() == 'yes' else 0
        
        # Process symptoms
        symptoms = [s.strip().lower() for s in patient_data['symptoms'].split(',') if s.strip()]
        high_risk_symptoms = ['headache', 'vision problems', 'epigastric pain', 'decreased fetal movement']
        
        features['symptom_count'] = len(symptoms)
        features['high_risk_symptoms'] = sum(1 for symptom in symptoms if symptom in high_risk_symptoms)
        
        return features
    
    def rule_based_prediction(self, features):
        """Simple rule-based prediction as fallback"""
        risk_score = 0
        
        # Blood pressure criteria
        if features['systolic_bp'] >= 160 or features['diastolic_bp'] >= 110:
            risk_score += 3  # Severe hypertension
        elif features['systolic_bp'] >= 140 or features['diastolic_bp'] >= 90:
            risk_score += 2  # Mild hypertension
        
        # Proteinuria
        if features['urine_protein_encoded'] >= 3:  # 2+ or 3+
            risk_score += 3
        elif features['urine_protein_encoded'] >= 2:  # 1+
            risk_score += 2
        elif features['urine_protein_encoded'] >= 1:  # trace
            risk_score += 1
        
        # History of preeclampsia
        if features['history_preeclampsia_encoded']:
            risk_score += 2
        
        # Age factors
        if features['maternal_age'] >= 40 or features['maternal_age'] < 18:
            risk_score += 1
        
        # High-risk symptoms
        risk_score += features['high_risk_symptoms']
        
        # Gestational age (later pregnancy = higher risk)
        if features['gestational_age'] >= 34:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 6:
            return 'High', 0.85
        elif risk_score >= 3:
            return 'Moderate', 0.70
        else:
            return 'Low', 0.60
    
    def predict(self, patient_data):
        """Make prediction for a single patient"""
        try:
            # Encode features
            features = self.encode_features(patient_data)
            
            if self.model is not None:
                # Use trained model
                feature_vector = np.array([[features[name] for name in self.feature_names]])
                prediction = self.model.predict(feature_vector)[0]
                probabilities = self.model.predict_proba(feature_vector)[0]
                confidence = max(probabilities)
                
                # Map numerical prediction to risk level
                risk_levels = ['Low', 'Moderate', 'High']
                risk_level = risk_levels[prediction]
                
                return risk_level, round(confidence, 3)
            else:
                # Use rule-based fallback
                return self.rule_based_prediction(features)
                
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            # Return conservative high-risk prediction on error
            return 'High', 0.5

    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            return sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return []
