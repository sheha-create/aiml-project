"""
Food Label Analysis System Configuration
Setup guide for complete deployment
"""

PROJECT_INFO = {
    'name': 'Food Label Analysis System',
    'version': '1.0.0',
    'description': 'AI-driven food label analysis for diabetic and hypertension patients',
    'author': 'AI/ML Development Team',
    'date_created': 'January 2026',
}

# ===================== SYSTEM FEATURES =====================

SYSTEM_FEATURES = {
    'ocr': {
        'name': 'OCR Engine',
        'description': 'Extracts nutrition facts and ingredients using EasyOCR + Tesseract',
        'languages': ['English', 'Hindi'],
        'supported_formats': ['PNG', 'JPG', 'PDF'],
        'target_accuracy': '85%+',
    },
    'clinical_metrics': {
        'name': 'Clinical Metrics Computation',
        'metrics': [
            'Glycemic Index (GI)',
            'Glycemic Load (GL)',
            'Sodium Load (%)',
            'Nutrient Density Score',
            'Hidden Sugars Detection',
        ],
        'target_precision': 'High (±15 cal, ±1.5g sugar)',
    },
    'classification': {
        'name': 'Personalized Food Classification',
        'classes': ['SUITABLE', 'MODERATE', 'AVOID'],
        'considers': [
            'Diabetes type (Type 1, Type 2, Gestational)',
            'Hypertension severity (Normal to Crisis)',
            'Age and activity level',
            'Individual medical thresholds',
        ],
        'target_f1_score': '0.75+',
    },
    'fraud_detection': {
        'name': 'Fraud Detection',
        'detects': [
            'Unrealistic nutrition claims',
            'Serving size manipulation',
            'Missing allergen information',
            'Ingredient-nutrition inconsistency',
        ],
        'validation': 'Market range comparison',
    },
    'substitution': {
        'name': 'Substitution Engine',
        'provides': [
            'Healthier alternatives',
            'Health improvement metrics',
            'Nutritional similarity matching',
            'Personalized suggestions',
        ],
    },
    'meal_simulation': {
        'name': 'Meal Simulation',
        'features': [
            'Multi-food impact analysis',
            'Aggregated metrics',
            'Safety scoring',
            'Real-time recommendations',
        ],
    },
    'compliance': {
        'name': 'Compliance Tracking',
        'features': [
            'Daily food logging',
            'Weekly compliance reports',
            'Caregiver notifications',
            'Improvement tracking',
        ],
    },
}

# ===================== INSTALLATION CHECKLIST =====================

INSTALLATION_STEPS = [
    {
        'step': 1,
        'name': 'Environment Setup',
        'commands': [
            'git clone <repository-url>',
            'cd food_label_analyzer',
            'python -m venv venv',
            'source venv/bin/activate  # Linux/Mac',
            'venv\\Scripts\\activate  # Windows',
        ],
    },
    {
        'step': 2,
        'name': 'Install Python Dependencies',
        'commands': [
            'pip install -r requirements.txt',
            'pip install pytest pytest-cov  # For testing',
        ],
    },
    {
        'step': 3,
        'name': 'Install System Dependencies (OCR)',
        'commands': [
            'sudo apt-get install tesseract-ocr libtesseract-dev  # Linux',
            'brew install tesseract  # macOS',
            '# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki',
        ],
    },
    {
        'step': 4,
        'name': 'Verify Installation',
        'commands': [
            'python -c "import pytesseract; print(\'Tesseract OK\')"',
            'python -c "import cv2; print(\'OpenCV OK\')"',
            'python -c "import easyocr; print(\'EasyOCR OK\')"',
        ],
    },
    {
        'step': 5,
        'name': 'Download Pre-trained Models',
        'commands': [
            'python notebooks/download_models.py  # If applicable',
        ],
    },
]

# ===================== QUICK START GUIDE =====================

QUICK_START_DEMO = """
# Quick Start Example: Analyzing a Food Label

from src.config import UserProfile, DiabetesType, HypertensionSeverity
from src.ocr_engine.label_ocr import NutritionLabelOCR, NutritionFactsParser
from src.clinical_metrics.metrics_calculator import ClinicalMetricsComputer
from src.classification.classifier import FoodClassifier

# Step 1: Create User Profile
user_profile = UserProfile(
    user_id="DEMO001",
    age=52,
    gender="male",
    weight_kg=85,
    height_cm=175,
    has_diabetes=True,
    diabetes_type=DiabetesType.TYPE_2,
    hypertension_severity=HypertensionSeverity.STAGE_1,
    max_daily_sugar_g=25,
    max_daily_sodium_mg=2000,
)

# Step 2: Extract Text from Food Label Image
ocr_engine = NutritionLabelOCR()
ocr_result = ocr_engine.extract_from_label("food_label.jpg")
print(f"OCR Confidence: {ocr_result.confidence:.1%}")

# Step 3: Parse Nutrition Facts
parser = NutritionFactsParser()
nutrition = parser.parse_nutrition_facts(ocr_result.raw_text)
print(f"Sugars: {nutrition.sugars_g}g, Sodium: {nutrition.sodium_mg}mg")

# Step 4: Compute Clinical Metrics
metrics_computer = ClinicalMetricsComputer(user_profile)
metrics = metrics_computer.compute_metrics(nutrition, "Cookies", user_profile)
print(f"Glycemic Load: {metrics.glycemic_load:.1f}")

# Step 5: Classify Food for User
classifier = FoodClassifier()
food_item = FoodItem(
    item_id="food_123",
    name="Cookies",
    brand="BrandX",
    category="snacks",
    nutrition_facts=nutrition,
    ingredients=IngredientsList(),
    clinical_metrics=metrics,
    classification=FoodClassification.SUITABLE,
)
classification, confidence, explanation = classifier.classify_food(food_item, user_profile)
print(f"Classification: {classification.value} ({confidence:.0%})")
print(f"Explanation: {explanation}")
"""

# ===================== API ENDPOINT SUMMARY =====================

API_ENDPOINTS = [
    {
        'method': 'POST',
        'endpoint': '/api/users/register',
        'description': 'Register new user with medical profile',
        'auth': 'Public',
    },
    {
        'method': 'POST',
        'endpoint': '/api/foods/analyze-label',
        'description': 'Analyze food label image',
        'auth': 'Required (user_id)',
    },
    {
        'method': 'POST',
        'endpoint': '/api/substitutions/recommend',
        'description': 'Get healthier food alternatives',
        'auth': 'Required',
    },
    {
        'method': 'POST',
        'endpoint': '/api/meals/simulate',
        'description': 'Evaluate multi-food meal impact',
        'auth': 'Required',
    },
    {
        'method': 'GET',
        'endpoint': '/api/users/{user_id}/weekly-report',
        'description': 'Generate weekly compliance report',
        'auth': 'Required',
    },
    {
        'method': 'GET',
        'endpoint': '/api/metrics/ocr-accuracy',
        'description': 'Get OCR accuracy metrics',
        'auth': 'Public',
    },
    {
        'method': 'GET',
        'endpoint': '/api/metrics/fraud-detection',
        'description': 'Get fraud detection statistics',
        'auth': 'Public',
    },
]

# ===================== DEPLOYMENT CHECKLIST =====================

DEPLOYMENT_CHECKLIST = {
    'development': [
        '✓ Install all dependencies',
        '✓ Run demo scripts',
        '✓ Test OCR on sample images',
        '✓ Verify classification accuracy',
    ],
    'testing': [
        '✓ Run evaluation_metrics.py',
        '✓ Check OCR accuracy >85%',
        '✓ Verify F1 score >0.75',
        '✓ Test API endpoints',
    ],
    'staging': [
        '✓ Deploy with Docker',
        '✓ Setup database (PostgreSQL)',
        '✓ Configure authentication (JWT)',
        '✓ Enable logging and monitoring',
    ],
    'production': [
        '✓ HIPAA compliance check',
        '✓ Data encryption at rest',
        '✓ SSL/TLS for APIs',
        '✓ Backup and disaster recovery',
        '✓ Audit logging',
        '✓ Rate limiting',
    ],
}

# ===================== PERFORMANCE TARGETS =====================

PERFORMANCE_TARGETS = {
    'ocr': {
        'extraction_success_rate': '>85%',
        'average_confidence': '0.85-0.95',
        'nutrient_extraction_accuracy': {
            'calories': '±15 kcal',
            'sugar': '±1.5g',
            'sodium': '±100mg',
        },
        'processing_time_per_image': '<5 seconds',
    },
    'classification': {
        'macro_f1_score': '≥0.75',
        'suitable_f1': '≥0.80',
        'moderate_f1': '≥0.72',
        'avoid_f1': '≥0.75',
        'response_time': '<1 second',
    },
    'compliance_improvement': {
        'sugar_reduction': '15-20%',
        'sodium_reduction': '12-18%',
        'user_engagement': '>70%',
    },
}

# ===================== TROUBLESHOOTING GUIDE =====================

TROUBLESHOOTING = {
    'ocr_low_accuracy': [
        'Ensure image resolution ≥300 DPI',
        'Check lighting and contrast',
        'Try preprocessing image manually',
        'Verify Tesseract installation',
    ],
    'classification_errors': [
        'Check user profile completeness',
        'Verify medical thresholds',
        'Review training data balance',
        'Test with known foods first',
    ],
    'api_timeout': [
        'Check system resources',
        'Verify database connection',
        'Test with smaller datasets',
        'Check network connectivity',
    ],
    'fraud_detection_false_positives': [
        'Review market range database',
        'Adjust confidence thresholds',
        'Check ingredient parsing',
        'Validate serving size ranges',
    ],
}

# ===================== CONFIGURATION =====================

CONFIG = {
    'app_name': PROJECT_INFO['name'],
    'version': PROJECT_INFO['version'],
    'debug': False,  # Set to True for development
    'workers': 4,
    'port': 8000,
    'host': '0.0.0.0',
    'ocr': {
        'use_easyocr': True,
        'use_tesseract': True,
        'preprocessing': True,
    },
    'database': {
        'type': 'sqlite',  # Production: postgresql
        'path': 'food_labels.db',
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    },
}

# ===================== MEDICAL COMPLIANCE =====================

MEDICAL_COMPLIANCE = {
    'regulations': [
        'FDA - Food Labeling Guide',
        'ADA - Diabetes Management Guidelines',
        'ACC/AHA - Hypertension Guidelines',
        'HIPAA - Patient Data Protection',
    ],
    'disclaimers': [
        'System is for informational purposes only',
        'Not a substitute for professional medical advice',
        'Users must consult healthcare providers',
        'Recommendations are not medical prescriptions',
    ],
    'data_protection': [
        'End-to-end encryption',
        'HIPAA-compliant audit logging',
        'PII data minimization',
        'Regular security audits',
    ],
}

if __name__ == '__main__':
    print("Food Label Analysis System - Configuration")
    print(f"Version: {PROJECT_INFO['version']}")
    print(f"Features: {len(SYSTEM_FEATURES)} major components")
    print(f"API Endpoints: {len(API_ENDPOINTS)}")
    print("\nFor complete setup, run:")
    print("  python notebooks/demo_usage.py")
    print("\nTo start API server, run:")
    print("  python -m uvicorn api.main:app --reload")
