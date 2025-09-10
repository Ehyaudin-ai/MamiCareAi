import os
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)

def generate_synthetic_dataset(n_samples=1000):
    """Generate synthetic preeclampsia dataset based on medical literature"""
    np.random.seed(42)  # For reproducibility
    
    data = []
    
    for i in range(n_samples):
        # Generate base patient characteristics
        maternal_age = np.random.normal(28, 6)  # Mean age 28, std 6
        maternal_age = max(16, min(45, int(maternal_age)))  # Clamp to realistic range
        
        gravida = np.random.poisson(2) + 1  # Number of pregnancies
        parity = max(0, gravida - np.random.poisson(1) - 1)  # Number of prior deliveries
        
        gestational_age = np.random.normal(32, 8)  # Gestational age in weeks
        gestational_age = max(20, min(42, int(gestational_age)))
        
        # History of preeclampsia (10% baseline rate)
        history_preeclampsia = np.random.choice(['yes', 'no'], p=[0.1, 0.9])
        
        # Generate blood pressure based on risk factors
        base_systolic = 110 + np.random.normal(0, 10)
        base_diastolic = 70 + np.random.normal(0, 8)
        
        # Risk modifiers
        age_risk = 0
        if maternal_age < 18 or maternal_age > 35:
            age_risk = 15
        
        history_risk = 20 if history_preeclampsia == 'yes' else 0
        late_pregnancy_risk = max(0, (gestational_age - 28) * 0.5)
        
        total_bp_risk = age_risk + history_risk + late_pregnancy_risk
        
        systolic_bp = base_systolic + total_bp_risk + np.random.normal(0, 5)
        diastolic_bp = base_diastolic + total_bp_risk * 0.6 + np.random.normal(0, 3)
        
        systolic_bp = max(80, min(200, int(systolic_bp)))
        diastolic_bp = max(50, min(130, int(diastolic_bp)))
        
        # Generate proteinuria based on BP and other risk factors
        protein_risk = 0
        if systolic_bp >= 140 or diastolic_bp >= 90:
            protein_risk += 0.3
        if history_preeclampsia == 'yes':
            protein_risk += 0.2
        if maternal_age > 35:
            protein_risk += 0.1
        
        protein_levels = ['negative', 'trace', '1+', '2+', '3+']
        protein_probs = [0.6 - protein_risk, 0.2, 0.1 + protein_risk/3, 0.05 + protein_risk/3, 0.05 + protein_risk/3]
        protein_probs = [max(0, p) for p in protein_probs]
        protein_probs = [p / sum(protein_probs) for p in protein_probs]  # Normalize
        
        urine_protein = np.random.choice(protein_levels, p=protein_probs)
        
        # Generate symptoms
        symptom_risk = 0
        if systolic_bp >= 140 or diastolic_bp >= 90:
            symptom_risk += 0.3
        if urine_protein in ['2+', '3+']:
            symptom_risk += 0.2
        if history_preeclampsia == 'yes':
            symptom_risk += 0.1
        
        possible_symptoms = ['headache', 'vision problems', 'swelling', 'epigastric pain', 'decreased fetal movement']
        symptoms = []
        
        for symptom in possible_symptoms:
            if np.random.random() < symptom_risk * 0.4:  # Adjust probability
                symptoms.append(symptom)
        
        symptoms_str = ', '.join(symptoms)
        
        # Determine risk level based on clinical criteria
        risk_score = 0
        
        # BP criteria
        if systolic_bp >= 160 or diastolic_bp >= 110:
            risk_score += 3
        elif systolic_bp >= 140 or diastolic_bp >= 90:
            risk_score += 2
        
        # Proteinuria
        protein_scores = {'negative': 0, 'trace': 1, '1+': 2, '2+': 3, '3+': 4}
        risk_score += protein_scores[urine_protein]
        
        # History
        if history_preeclampsia == 'yes':
            risk_score += 2
        
        # Age
        if maternal_age < 18 or maternal_age > 35:
            risk_score += 1
        
        # Symptoms
        high_risk_symptoms = ['headache', 'vision problems', 'epigastric pain', 'decreased fetal movement']
        risk_score += sum(1 for symptom in symptoms if symptom in high_risk_symptoms)
        
        # Assign risk level
        if risk_score >= 6:
            risk_level = 'High'
        elif risk_score >= 3:
            risk_level = 'Moderate'
        else:
            risk_level = 'Low'
        
        # Add some noise to make it more realistic
        if np.random.random() < 0.05:  # 5% chance of mis-classification
            risk_levels = ['Low', 'Moderate', 'High']
            current_idx = risk_levels.index(risk_level)
            if current_idx > 0 and np.random.random() < 0.5:
                risk_level = risk_levels[current_idx - 1]
            elif current_idx < 2:
                risk_level = risk_levels[current_idx + 1]
        
        data.append({
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'urine_protein': urine_protein,
            'maternal_age': maternal_age,
            'gravida': gravida,
            'parity': parity,
            'history_preeclampsia': history_preeclampsia,
            'gestational_age': gestational_age,
            'symptoms': symptoms_str,
            'risk_level': risk_level
        })
    
    return pd.DataFrame(data)

def prepare_features(df):
    """Prepare features for machine learning"""
    # Create a copy
    df_processed = df.copy()
    
    # Encode categorical variables
    protein_mapping = {'negative': 0, 'trace': 1, '1+': 2, '2+': 3, '3+': 4}
    df_processed['urine_protein_encoded'] = df_processed['urine_protein'].map(protein_mapping)
    
    df_processed['history_preeclampsia_encoded'] = (df_processed['history_preeclampsia'] == 'yes').astype(int)
    
    # Process symptoms
    def count_symptoms(symptoms_str):
        if pd.isna(symptoms_str) or symptoms_str == '':
            return 0
        return len([s.strip() for s in symptoms_str.split(',') if s.strip()])
    
    def count_high_risk_symptoms(symptoms_str):
        if pd.isna(symptoms_str) or symptoms_str == '':
            return 0
        symptoms = [s.strip().lower() for s in symptoms_str.split(',') if s.strip()]
        high_risk = ['headache', 'vision problems', 'epigastric pain', 'decreased fetal movement']
        return sum(1 for symptom in symptoms if symptom in high_risk)
    
    df_processed['symptom_count'] = df_processed['symptoms'].apply(count_symptoms)
    df_processed['high_risk_symptoms'] = df_processed['symptoms'].apply(count_high_risk_symptoms)
    
    # Select features for model
    feature_columns = [
        'systolic_bp', 'diastolic_bp', 'urine_protein_encoded', 
        'maternal_age', 'gravida', 'parity', 'history_preeclampsia_encoded',
        'gestational_age', 'symptom_count', 'high_risk_symptoms'
    ]
    
    X = df_processed[feature_columns]
    
    # Encode target variable
    le_target = LabelEncoder()
    y = le_target.fit_transform(df_processed['risk_level'])
    
    return X, y, le_target, feature_columns

def train_model():
    """Train the preeclampsia prediction model"""
    logging.info("Generating synthetic dataset...")
    
    # Generate dataset
    df = generate_synthetic_dataset(n_samples=2000)
    
    # Save dataset
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_dataset.csv', index=False)
    logging.info("Dataset saved to data/sample_dataset.csv")
    
    # Prepare features
    X, y, le_target, feature_columns = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    logging.info("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logging.info(f"Model accuracy: {accuracy:.3f}")
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred, target_names=le_target.classes_))
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logging.info("\nFeature Importance:")
    logging.info(importance.to_string(index=False))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_data = {
        'model': model,
        'encoders': {'target': le_target},
        'feature_columns': feature_columns,
        'accuracy': accuracy
    }
    
    joblib.dump(model_data, 'models/preeclampsia_model.joblib')
    logging.info("Model saved to models/preeclampsia_model.joblib")
    
    return model, le_target, feature_columns

if __name__ == "__main__":
    logging.info("Starting MAMICARE AI model training...")
    train_model()
    logging.info("Training completed successfully!")
