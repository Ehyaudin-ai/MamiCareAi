import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from model import PreeclampsiaPredictor

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "mamicare-ai-secret-key")

# Initialize the predictor
predictor = PreeclampsiaPredictor()

@app.route('/')
def index():
    """Main form page for patient data input"""
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Process form data and make prediction"""
    try:
        # Extract form data
        patient_data = {
            'systolic_bp': float(request.form.get('systolic_bp', 0)),
            'diastolic_bp': float(request.form.get('diastolic_bp', 0)),
            'urine_protein': request.form.get('urine_protein', 'negative'),
            'maternal_age': int(request.form.get('maternal_age', 0)),
            'gravida': int(request.form.get('gravida', 0)),
            'parity': int(request.form.get('parity', 0)),
            'history_preeclampsia': request.form.get('history_preeclampsia', 'no'),
            'gestational_age': int(request.form.get('gestational_age', 0)),
            'symptoms': request.form.get('symptoms', '').strip()
        }
        
        # Validate required fields
        if not all([
            patient_data['systolic_bp'] > 0,
            patient_data['diastolic_bp'] > 0,
            patient_data['maternal_age'] > 0,
            patient_data['gestational_age'] > 0
        ]):
            flash('Please fill in all required fields with valid values.', 'error')
            return redirect(url_for('index'))
        
        # Make prediction
        risk_level, confidence = predictor.predict(patient_data)
        
        # Generate recommendations
        recommendations = generate_recommendations(risk_level, patient_data['history_preeclampsia'])
        
        # Save case to CSV
        save_case_to_csv(patient_data, risk_level, confidence)
        
        # Prepare result data
        result_data = {
            'patient_data': patient_data,
            'risk_level': risk_level,
            'confidence': confidence,
            'recommendations': recommendations,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return render_template('result.html', **result_data)
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        flash(f'An error occurred during prediction: {str(e)}', 'error')
        return redirect(url_for('index'))

def generate_recommendations(risk_level, history_preeclampsia):
    """Generate medical recommendations based on risk level and history"""
    recommendations = []
    
    if risk_level == 'High':
        recommendations.append("High risk — urgent referral needed coupled with loading magnesium sulfate (MgSo4). Call or transfer to nearest facility.")
    elif risk_level == 'Moderate':
        recommendations.append("Moderate risk — contact health worker and schedule assessment within 24 hours.")
    else:  # Low risk
        next_appointment = datetime.now().strftime('%Y-%m-%d')
        recommendations.append(f"Low risk — continue routine care and next appointment on {next_appointment}.")
    
    # Add aspirin recommendation if history of preeclampsia
    if history_preeclampsia.lower() == 'yes':
        recommendations.append("Recommend low-dose aspirin per protocol; confirm with clinician.")
    
    return recommendations

def save_case_to_csv(patient_data, risk_level, confidence):
    """Save case data to local CSV file with timestamp"""
    try:
        # Prepare data for CSV
        case_data = {
            'timestamp': datetime.now().isoformat(),
            'systolic_bp': patient_data['systolic_bp'],
            'diastolic_bp': patient_data['diastolic_bp'],
            'urine_protein': patient_data['urine_protein'],
            'maternal_age': patient_data['maternal_age'],
            'gravida': patient_data['gravida'],
            'parity': patient_data['parity'],
            'history_preeclampsia': patient_data['history_preeclampsia'],
            'gestational_age': patient_data['gestational_age'],
            'symptoms': patient_data['symptoms'],
            'predicted_risk': risk_level,
            'confidence': confidence
        }
        
        # Create DataFrame
        df = pd.DataFrame([case_data])
        
        # Check if file exists and append or create new
        csv_file = 'data/case_records.csv'
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        if os.path.exists(csv_file):
            # Append to existing file
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            # Create new file with header
            df.to_csv(csv_file, mode='w', header=True, index=False)
            
        logging.info(f"Case saved to {csv_file}")
        
    except Exception as e:
        logging.error(f"Error saving case to CSV: {str(e)}")

@app.route('/cases')
def view_cases():
    """View saved cases (optional feature for clinic staff)"""
    try:
        csv_file = 'data/case_records.csv'
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            cases = df.to_dict('records')
            return render_template('cases.html', cases=cases)
        else:
            flash('No cases found.', 'info')
            return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error loading cases: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/demo')
def demo_dashboard():
    """Demonstration dashboard for examiners and audiences"""
    try:
        # Load presentation materials
        import json
        
        # Load case studies
        with open('demo_materials/case_studies.json', 'r') as f:
            case_studies = json.load(f)
            
        # Load metrics
        with open('demo_materials/presentation_metrics.json', 'r') as f:
            metrics = json.load(f)
            
        # Load validation data
        with open('demo_materials/research_validation.json', 'r') as f:
            validation = json.load(f)
            
        # Load real dataset
        real_df = pd.read_csv('data/real_datasets/maternal_health_dataset.csv')
        sample_cases = real_df.head(10).to_dict('records')
        
        return render_template('demo_dashboard.html', 
                             case_studies=case_studies,
                             metrics=metrics,
                             validation=validation,
                             sample_cases=sample_cases,
                             total_patients=len(real_df))
    except Exception as e:
        logging.error(f"Demo dashboard error: {str(e)}")
        flash(f'Error loading demonstration materials: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/dataset')
def dataset_api():
    """API endpoint for dataset information"""
    try:
        # Load real dataset statistics
        real_df = pd.read_csv('data/real_datasets/maternal_health_dataset.csv')
        
        stats = {
            'total_patients': len(real_df),
            'risk_distribution': real_df['risk_level'].value_counts().to_dict(),
            'age_stats': {
                'mean': real_df['maternal_age'].mean(),
                'min': real_df['maternal_age'].min(),
                'max': real_df['maternal_age'].max()
            },
            'bp_stats': {
                'mean_systolic': real_df['systolic_bp'].mean(),
                'mean_diastolic': real_df['diastolic_bp'].mean(),
                'hypertension_rate': ((real_df['systolic_bp'] >= 140) | (real_df['diastolic_bp'] >= 90)).sum() / len(real_df)
            },
            'proteinuria_distribution': real_df['urine_protein'].value_counts().to_dict(),
            'data_quality': {
                'completeness': (real_df.notna().sum() / len(real_df)).to_dict(),
                'date_range': f"{real_df['timestamp'].min()[:10]} to {real_df['timestamp'].max()[:10]}"
            }
        }
        
        return stats
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
