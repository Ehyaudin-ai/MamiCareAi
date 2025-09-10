# MAMICARE AI - Preeclampsia Screening Tool

An offline-capable Flask web application for preeclampsia risk screening in rural clinics using lightweight machine learning.

## Overview

MAMICARE AI is designed specifically for low-resource healthcare settings where internet connectivity may be unreliable. The application uses a trained machine learning model to assess preeclampsia risk based on simple clinical inputs and provides evidence-based recommendations for patient care.

## Features

- **Offline Capability**: Runs entirely locally without internet dependency
- **Simple Interface**: Mobile-friendly, text-first design optimized for rural clinic use
- **AI-Powered Risk Assessment**: Uses lightweight Random Forest classifier
- **Clinical Decision Support**: Provides risk-appropriate recommendations
- **Case Management**: Automatic saving of assessments to local CSV files
- **Emergency Protocols**: Clear guidance for high-risk cases

## Risk Categories

- **Low Risk**: Continue routine care
- **Moderate Risk**: Contact health worker within 24 hours
- **High Risk**: Urgent referral with magnesium sulfate preparation

## Installation & Setup

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Quick Start

1. **Install Dependencies**
   ```bash
   pip install flask pandas scikit-learn joblib numpy
   ```

2. **Train the Model**
   ```bash
   python train.py
   ```
   This generates a synthetic dataset and trains the machine learning model.

3. **Run the Application**
   ```bash
   python app.py
   ```
   The application will be available at `http://localhost:5000`

## Project Structure

```
mamicare-ai/
├── app.py                  # Main Flask application
├── main.py                 # Entry point for deployment
├── model.py                # ML model wrapper and prediction logic
├── train.py                # Model training script
├── templates/
│   ├── form.html          # Patient assessment form
│   ├── result.html        # Risk assessment results
│   └── cases.html         # Case history viewer
├── static/
│   └── style.css          # Custom styling
├── data/
│   ├── sample_dataset.csv # Generated training data
│   └── case_records.csv   # Saved patient assessments
├── models/
│   └── preeclampsia_model.joblib # Trained ML model
└── README.md

```

## Model Performance

- **Algorithm**: Random Forest Classifier
- **Training Accuracy**: 91.5%
- **Key Features** (by importance):
  1. Urine protein level (29.3%)
  2. Systolic blood pressure (15.6%)
  3. Diastolic blood pressure (13.6%)
  4. History of preeclampsia (11.2%)
  5. Symptom count (8.7%)

## Usage

### Assessment Workflow

1. **Patient Data Entry**: Fill out the comprehensive assessment form with:
   - Blood pressure readings (systolic/diastolic)
   - Urine protein level (dipstick results)
   - Maternal age and pregnancy history
   - Current gestational age
   - Present symptoms

2. **Risk Assessment**: The AI model analyzes input data and provides:
   - Risk level (Low/Moderate/High)
   - Confidence percentage
   - Clinical recommendations
   - Emergency protocols (if high risk)

3. **Case Management**: Each assessment is automatically saved with:
   - Timestamp for tracking
   - Complete patient data
   - Prediction results
   - Downloadable case records

## Clinical Decision Support

### Risk Categories & Actions

**🟢 LOW RISK**
- Continue routine prenatal care
- Next scheduled appointment
- Routine monitoring

**🟡 MODERATE RISK**
- Contact health worker within 24 hours
- Enhanced monitoring
- Consider additional testing

**🔴 HIGH RISK**
- **URGENT**: Immediate referral required
- Prepare magnesium sulfate (MgSO₄)
- Contact receiving facility
- Emergency transport preparation

### Special Recommendations

- **History of Preeclampsia**: Automatic low-dose aspirin recommendation
- **Emergency Protocols**: Detailed high-risk management guidelines
- **Transport Preparation**: Pre-referral stabilization checklist

## Technical Features

### Offline Capability
- No internet required during operation
- Local model inference
- CSV-based data storage
- Self-contained dependencies

### Mobile Optimization
- Responsive Bootstrap design
- Touch-friendly interface
- Text-first design for low-bandwidth
- Print-ready results

### Data Security
- Local data storage only
- No external API calls
- Patient privacy protection
- HIPAA-compliant design

## Deployment Options

### Local Clinic Setup
```bash
# Quick deployment
python train.py    # One-time setup
python app.py     # Start application
```

### Production Deployment
```bash
# Using Gunicorn (recommended)
gunicorn --bind 0.0.0.0:5000 --reuse-port main:app
```

## Customization

### Adjusting Risk Thresholds
Edit `model.py` to modify the rule-based fallback criteria:
```python
# Blood pressure thresholds
if systolic_bp >= 160 or diastolic_bp >= 110:
    risk_score += 3  # Severe hypertension
```

### Adding New Features
1. Update the form in `templates/form.html`
2. Modify feature encoding in `model.py`
3. Retrain model with `train.py`

### Localization
- Translate templates for local language
- Adjust clinical guidelines for regional protocols
- Modify recommendation text in `app.py`

## Validation & Testing

The model was trained on synthetic data based on:
- WHO preeclampsia guidelines
- ACOG clinical recommendations
- Evidence-based risk factors
- Rural clinic workflow requirements

**Note**: For production use, validate with local clinical data and adjust thresholds according to regional protocols.

## Support & Maintenance

### Monitoring Model Performance
- Review case records in `data/case_records.csv`
- Track prediction confidence levels
- Collect feedback from healthcare providers

### Regular Updates
- Retrain model with actual clinical outcomes
- Update risk thresholds based on local data
- Refresh recommendations per latest guidelines

## License & Disclaimer

This tool is designed for clinical decision support only and should not replace professional medical judgment. Always consult with qualified healthcare providers for final treatment decisions.

## Contributing

For improvements or adaptations:
1. Fork the repository
2. Implement changes
3. Test with synthetic data
4. Submit pull request with documentation

---

**MAMICARE AI** - Bridging the gap in rural maternal healthcare through accessible AI technology.
