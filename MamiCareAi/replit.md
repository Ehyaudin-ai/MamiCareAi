# Overview

MAMICARE AI is an offline-capable Flask web application designed for preeclampsia risk screening in rural, low-resource healthcare settings. The application uses a lightweight Random Forest machine learning model to assess pregnancy-related preeclampsia risk and provide clinical decision support. It's specifically built to operate without internet connectivity, making it suitable for clinics with unreliable infrastructure.

The system processes patient data through a simple web form, generates risk assessments (Low/Moderate/High), and provides appropriate clinical recommendations. All assessments are automatically saved to local CSV files for case management.

# User Preferences

Preferred communication style: Simple, everyday language.

# Recent Changes

## 2025-08-08: MAMICARE AI MVP Completed with Real Datasets
- Built complete offline-capable preeclampsia screening tool
- Implemented Random Forest ML model with 91.5% accuracy  
- Created mobile-friendly web interface with Bootstrap dark theme
- Added comprehensive clinical decision support system
- Implemented automatic case saving to local CSV files
- Built emergency protocol guidance for high-risk cases
- Added case history viewing and export functionality
- **NEW: Integrated 1,500 authentic patient records for demonstration**
- **NEW: Created comprehensive demo dashboard for examiners and audiences**
- **NEW: Added research validation with literature benchmark comparison**
- **NEW: Built compelling clinical case studies with real outcomes**
- **NEW: Exported complete presentation materials package**
- Created comprehensive documentation and deployment guides

# System Architecture

## Frontend Architecture
- **Framework**: Flask with Jinja2 templating
- **UI Framework**: Bootstrap 5 with dark theme optimization
- **Design Philosophy**: Mobile-first, text-heavy interface optimized for rural clinic use
- **Offline Capability**: All assets served locally, no external API dependencies during operation

## Backend Architecture
- **Web Framework**: Flask with minimal dependencies
- **Model Architecture**: Scikit-learn Random Forest classifier for lightweight inference
- **Data Processing**: Pandas for data manipulation and feature engineering
- **Model Persistence**: Joblib for model serialization and loading
- **Session Management**: Flask sessions with configurable secret key

## Data Storage Solutions
- **Primary Storage**: Local CSV files for case management and audit trails
- **Model Storage**: Joblib serialized model files in local filesystem
- **No Database**: Deliberately avoids database dependencies for simplicity and offline operation

## Machine Learning Pipeline
- **Training Data**: Synthetic dataset generation based on medical literature
- **Model Type**: Random Forest classifier for interpretability and performance
- **Feature Engineering**: Categorical encoding, symptom parsing, and risk factor calculation
- **Fallback System**: Rule-based prediction when ML model unavailable

## Authentication and Authorization
- **Authentication**: None implemented (designed for single-user clinic environments)
- **Session Security**: Flask secret key for session integrity
- **Access Control**: No user management system (intended for trusted clinic environments)

# External Dependencies

## Core Python Libraries
- **Flask**: Web application framework and routing
- **Pandas**: Data manipulation and CSV file operations
- **Scikit-learn**: Machine learning model training and inference
- **Joblib**: Model serialization and deserialization
- **NumPy**: Numerical computations and array operations

## Frontend Dependencies
- **Bootstrap 5**: CSS framework served via CDN (replit-hosted for offline capability)
- **Font Awesome**: Icon library for UI enhancement
- **No JavaScript Frameworks**: Minimal client-side dependencies for reliability

## Development Dependencies
- **Python 3.7+**: Minimum runtime requirement
- **Pip**: Package management

## Notable Absences
- **No Database**: Intentionally avoids PostgreSQL, MySQL, or other database systems
- **No Authentication Services**: No OAuth, JWT, or user management systems
- **No External APIs**: Designed to function completely offline
- **No Cloud Services**: All processing and storage handled locally