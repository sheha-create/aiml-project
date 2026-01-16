"""
REST API for Food Label Analysis System
Provides endpoints for OCR, classification, recommendations, and meal planning
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
from datetime import datetime
import uuid

# Import system modules
from src.config import (
    UserProfile, DiabetesType, HypertensionSeverity, ActivityLevel, Gender,
    FoodItem, NutritionFacts, IngredientsList, ClinicalMetrics, FoodClassification
)
from src.ocr_engine.label_ocr import NutritionLabelOCR, NutritionFactsParser, LabelNormalizer
from src.clinical_metrics.metrics_calculator import ClinicalMetricsComputer
from src.classification.classifier import FoodClassifier
from src.fraud_detection.fraud_detector import FraudDetectionEngine
from src.substitution_engine.recommender import SubstitutionRecommendationEngine
from src.meal_simulation.simulator import MealSimulator
from src.compliance_tracking.tracker import DailyConsumptionTracker, ComplianceReportGenerator


# ===================== REQUEST/RESPONSE MODELS =====================

class UserProfileRequest(BaseModel):
    """User profile creation request"""
    user_id: str
    age: int
    gender: str  # "male", "female", "other"
    weight_kg: float
    height_cm: float
    diabetes_type: Optional[str] = None  # type_1, type_2, gestational
    has_diabetes: bool = False
    hypertension_severity: str = "normal"
    activity_level: str = "lightly_active"
    max_daily_sugar_g: Optional[float] = None
    max_daily_sodium_mg: Optional[float] = None
    max_daily_calories: Optional[float] = None
    caregiver_email: Optional[str] = None


class OCRAnalysisRequest(BaseModel):
    """OCR analysis request"""
    user_id: str
    food_name: str
    category: str
    detect_fraud: bool = True


class OCRAnalysisResponse(BaseModel):
    """OCR analysis response"""
    status: str
    food_id: str
    ocr_confidence: float
    nutrition_facts: Dict
    ingredients: Dict
    classification: str
    classification_confidence: float
    clinical_metrics: Dict
    fraud_analysis: Optional[Dict] = None
    explanation: str


class MealSimulationRequest(BaseModel):
    """Meal simulation request"""
    user_id: str
    meal_name: str
    food_ids: List[str]


class SubstitutionRequest(BaseModel):
    """Substitution recommendation request"""
    user_id: str
    food_id: str


# ===================== FASTAPI APPLICATION =====================

app = FastAPI(
    title="Food Label Analysis System",
    description="AI-driven food label analysis for diabetic and hypertension patients",
    version="1.0.0"
)

# Global storage (in production, use database)
users: Dict[str, UserProfile] = {}
foods: Dict[str, FoodItem] = {}
trackers: Dict[str, DailyConsumptionTracker] = {}

# Initialize engines
ocr_engine = NutritionLabelOCR()
ocr_parser = NutritionFactsParser()
label_normalizer = LabelNormalizer()
classifier = FoodClassifier()
fraud_engine = FraudDetectionEngine()
substitution_engine = SubstitutionRecommendationEngine()
meal_simulator = MealSimulator()


# ===================== API ENDPOINTS =====================

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "system": "Food Label Analysis System",
        "version": "1.0.0"
    }


@app.post("/api/users/register")
async def register_user(profile: UserProfileRequest):
    """
    Register a new user with medical profile
    
    Args:
        profile: User profile information
        
    Returns:
        Created user profile
    """
    try:
        # Parse enums
        gender = Gender[profile.gender.upper()]
        hypertension = HypertensionSeverity[profile.hypertension_severity.upper()]
        activity = ActivityLevel[profile.activity_level.upper()]
        
        diabetes_type = None
        if profile.has_diabetes and profile.diabetes_type:
            diabetes_type = DiabetesType[profile.diabetes_type.upper()]
        
        # Create user profile
        user = UserProfile(
            user_id=profile.user_id,
            age=profile.age,
            gender=gender,
            weight_kg=profile.weight_kg,
            height_cm=profile.height_cm,
            diabetes_type=diabetes_type,
            has_diabetes=profile.has_diabetes,
            hypertension_severity=hypertension,
            activity_level=activity,
            max_daily_sugar_g=profile.max_daily_sugar_g,
            max_daily_sodium_mg=profile.max_daily_sodium_mg,
            max_daily_calories=profile.max_daily_calories,
            caregiver_email=profile.caregiver_email,
        )
        
        users[profile.user_id] = user
        trackers[profile.user_id] = DailyConsumptionTracker(profile.user_id)
        
        return {
            "status": "success",
            "user_id": user.user_id,
            "message": "User profile created successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/foods/analyze-label")
async def analyze_food_label(request: OCRAnalysisRequest, file: UploadFile = File(...)):
    """
    Analyze food label from image using OCR
    
    Args:
        request: Analysis request parameters
        file: Image file of food label
        
    Returns:
        Analyzed food item with classification and metrics
    """
    try:
        # Validate user exists
        if request.user_id not in users:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_profile = users[request.user_id]
        
        # Save uploaded image temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # OCR extraction
            ocr_result = ocr_engine.extract_from_label(tmp_path)
            
            if not ocr_result.raw_text:
                raise HTTPException(status_code=400, detail="Could not extract text from label")
            
            # Parse nutrition facts
            nutrition = ocr_parser.parse_nutrition_facts(
                ocr_result.detected_regions.get('nutrition_facts', ocr_result.raw_text)
            )
            
            # Normalize to 100g
            nutrition = label_normalizer.normalize_nutrition_facts(nutrition, 100)
            
            # Parse ingredients
            ingredients = ocr_parser.parse_ingredients(
                ocr_result.detected_regions.get('ingredients', '')
            )
            
            # Compute clinical metrics
            metrics_computer = ClinicalMetricsComputer(user_profile)
            clinical_metrics = metrics_computer.compute_metrics(
                nutrition, request.food_name, user_profile
            )
            
            # Classify food
            food_item = FoodItem(
                item_id=f"food_{uuid.uuid4().hex[:8]}",
                name=request.food_name,
                brand="",
                category=request.category,
                nutrition_facts=nutrition,
                ingredients=ingredients,
                clinical_metrics=clinical_metrics,
                classification=FoodClassification.SUITABLE,  # Placeholder
                image_path=tmp_path,
                ocr_confidence=ocr_result.confidence,
            )
            
            classification, confidence, explanation = classifier.classify_food(
                food_item, user_profile
            )
            
            food_item.classification = classification
            food_item.classification_score = confidence
            food_item.explanation = explanation
            
            # Fraud detection
            fraud_result = None
            if request.detect_fraud:
                fraud_result = fraud_engine.analyze_for_fraud(food_item, request.category)
            
            # Store food item
            foods[food_item.item_id] = food_item
            
            # Log consumption
            if user_profile.user_id in trackers:
                trackers[user_profile.user_id].log_food(food_item)
            
            # Prepare response
            response = OCRAnalysisResponse(
                status="success",
                food_id=food_item.item_id,
                ocr_confidence=ocr_result.confidence,
                nutrition_facts={
                    'serving_size_g': nutrition.serving_size_grams,
                    'calories': nutrition.calories,
                    'carbs_g': nutrition.total_carbs_g,
                    'sugars_g': nutrition.sugars_g,
                    'sodium_mg': nutrition.sodium_mg,
                    'fiber_g': nutrition.dietary_fiber_g,
                    'protein_g': nutrition.protein_g,
                },
                ingredients={
                    'ingredients': ingredients.ingredients,
                    'sugar_indicators': ingredients.sugar_ingredients,
                    'sodium_indicators': ingredients.sodium_ingredients,
                    'allergens': ingredients.allergens,
                },
                classification=classification.value,
                classification_confidence=confidence,
                clinical_metrics={
                    'glycemic_index': clinical_metrics.glycemic_index,
                    'glycemic_load': clinical_metrics.glycemic_load,
                    'sodium_load_percent': clinical_metrics.sodium_load_percentage,
                    'sugar_load_percent': clinical_metrics.sugar_load_percentage,
                    'nutrient_density_score': clinical_metrics.nutrient_density_score,
                    'risk_factors': clinical_metrics.risk_factors,
                },
                fraud_analysis=None if not fraud_result else {
                    'fraud_confidence': fraud_result.fraud_confidence,
                    'flags': fraud_result.fraud_flags,
                    'details': fraud_result.details,
                },
                explanation=explanation,
            )
            
            return response
        
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/substitutions/recommend")
async def get_substitutions(request: SubstitutionRequest):
    """
    Get healthier food substitutions
    
    Args:
        request: User and food IDs
        
    Returns:
        List of recommended substitutes
    """
    try:
        if request.user_id not in users:
            raise HTTPException(status_code=404, detail="User not found")
        
        if request.food_id not in foods:
            raise HTTPException(status_code=404, detail="Food not found")
        
        user_profile = users[request.user_id]
        original_food = foods[request.food_id]
        
        # Find substitutes
        recommendation = substitution_engine.find_substitutes(
            original_food, user_profile, max_suggestions=5
        )
        
        return {
            "status": "success",
            "original_food_id": request.food_id,
            "substitutes": [
                {
                    "food_id": sub.item_id,
                    "name": sub.name,
                    "brand": sub.brand,
                    "classification": sub.classification.value,
                    "sugar_reduction_percent": recommendation.sugar_reduction_percent,
                    "sodium_reduction_percent": recommendation.sodium_reduction_percent,
                    "calorie_reduction_percent": recommendation.calorie_reduction_percent,
                } for sub in recommendation.substitute_items
            ],
            "reasoning": recommendation.reasoning,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/meals/simulate")
async def simulate_meal(request: MealSimulationRequest):
    """
    Simulate multi-food meal impact
    
    Args:
        request: User ID and food IDs in meal
        
    Returns:
        Meal analysis with safety score and recommendations
    """
    try:
        if request.user_id not in users:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_profile = users[request.user_id]
        
        # Get food items
        meal_foods = []
        for food_id in request.food_ids:
            if food_id not in foods:
                raise HTTPException(status_code=404, detail=f"Food {food_id} not found")
            meal_foods.append(foods[food_id])
        
        # Simulate meal
        meal_result = meal_simulator.simulate_meal(
            meal_foods, request.meal_name, user_profile
        )
        
        return {
            "status": "success",
            "meal_id": meal_result.meal_id,
            "meal_name": request.meal_name,
            "totals": {
                "calories": meal_result.total_calories,
                "carbs_g": meal_result.total_carbs_g,
                "sugars_g": meal_result.total_sugars_g,
                "sodium_mg": meal_result.total_sodium_mg,
                "fiber_g": meal_result.total_fiber_g,
                "protein_g": meal_result.total_protein_g,
            },
            "estimated_glycemic_load": meal_result.estimated_glycemic_load,
            "safety_score": meal_result.meal_safety_score,
            "classification": meal_result.meal_classification.value,
            "recommendations": meal_result.recommendations,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users/{user_id}/weekly-report")
async def get_weekly_compliance_report(user_id: str):
    """
    Generate weekly compliance report for caregiver
    
    Args:
        user_id: User ID
        
    Returns:
        Compliance report with metrics and recommendations
    """
    try:
        if user_id not in users:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_profile = users[user_id]
        tracker = trackers.get(user_id)
        
        if not tracker:
            raise HTTPException(status_code=400, detail="No consumption data for user")
        
        # Generate report
        generator = ComplianceReportGenerator(tracker)
        report = generator.generate_weekly_report(user_profile)
        
        return {
            "status": "success",
            "user_id": user_id,
            "report_period": f"{report.report_start_date.strftime('%Y-%m-%d')} to {report.report_end_date.strftime('%Y-%m-%d')}",
            "compliance_percentage": report.compliance_percentage,
            "total_meals_logged": report.total_meals_logged,
            "average_daily_sugar_g": report.average_daily_sugar_g,
            "average_daily_sodium_mg": report.average_daily_sodium_mg,
            "average_daily_calories": report.average_daily_calories,
            "sugar_threshold_violations": report.sugar_threshold_violations,
            "sodium_threshold_violations": report.sodium_threshold_violations,
            "summary": report.summary_text,
            "recommendations": report.recommendations,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/ocr-accuracy")
async def get_ocr_accuracy_metrics():
    """Get OCR accuracy metrics"""
    metrics = ocr_engine.get_ocr_accuracy()
    return {
        "status": "success",
        "ocr_metrics": metrics,
    }


@app.get("/api/metrics/fraud-detection")
async def get_fraud_detection_metrics():
    """Get fraud detection metrics"""
    metrics = fraud_engine.get_fraud_metrics()
    return {
        "status": "success",
        "fraud_metrics": metrics,
    }
