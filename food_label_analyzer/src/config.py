"""
Configuration and data models for the Food Label Analysis System
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# ===================== ENUMS =====================

class DiabetesType(Enum):
    TYPE_1 = "type_1"
    TYPE_2 = "type_2"
    GESTATIONAL = "gestational"
    PREDIABETIC = "prediabetic"

class HypertensionSeverity(Enum):
    NORMAL = "normal"                    # SBP < 120, DBP < 80
    ELEVATED = "elevated"                # SBP 120-129, DBP < 80
    STAGE_1 = "stage_1"                  # SBP 130-139, DBP 80-89
    STAGE_2 = "stage_2"                  # SBP >= 140, DBP >= 90
    CRISIS = "crisis"                    # SBP > 180, DBP > 120

class ActivityLevel(Enum):
    SEDENTARY = "sedentary"              # Little or no exercise
    LIGHTLY_ACTIVE = "lightly_active"    # 1-3 days/week
    MODERATELY_ACTIVE = "moderately_active"  # 3-5 days/week
    VERY_ACTIVE = "very_active"          # 6-7 days/week
    EXTREMELY_ACTIVE = "extremely_active"  # Physical job + exercise

class FoodClassification(Enum):
    SUITABLE = "suitable"                # Safe for regular consumption
    MODERATE = "moderate"                # Acceptable in controlled portions
    AVOID = "avoid"                      # Not recommended

class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"

class UserGoal(Enum):
    """User-selected health goals for food recommendations"""
    LOW_SUGAR = "low_sugar"
    LOW_SODIUM = "low_sodium"
    LOW_CALORIES = "low_calories"
    HIGH_PROTEIN = "high_protein"
    HIGH_FIBER = "high_fiber"
    BALANCED = "balanced"

class DietaryPreference(Enum):
    """User dietary preferences"""
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    HALAL = "halal"
    KOSHER = "kosher"
    GLUTEN_FREE = "gluten_free"
    NO_PREFERENCE = "no_preference"

class CuisinePreference(Enum):
    """User preferred cuisines"""
    INDIAN = "indian"
    WESTERN = "western"
    ASIAN = "asian"
    MEDITERRANEAN = "mediterranean"
    MIXED = "mixed"

# ===================== DATA MODELS =====================

@dataclass
class NutritionFacts:
    """Standardized nutrition information extracted from food labels"""
    serving_size_grams: float
    serving_size_unit: str
    servings_per_container: Optional[float] = None
    
    # Core nutrients
    calories: float = 0.0
    total_fat_g: float = 0.0
    saturated_fat_g: float = 0.0
    trans_fat_g: float = 0.0
    polyunsaturated_fat_g: float = 0.0
    monounsaturated_fat_g: float = 0.0
    
    cholesterol_mg: float = 0.0
    sodium_mg: float = 0.0
    
    total_carbs_g: float = 0.0
    dietary_fiber_g: float = 0.0
    sugars_g: float = 0.0
    added_sugars_g: Optional[float] = None
    sugar_alcohols_g: Optional[float] = None
    
    protein_g: float = 0.0
    
    # Minerals and vitamins
    potassium_mg: Optional[float] = None
    calcium_mg: Optional[float] = None
    iron_mg: Optional[float] = None
    magnesium_mg: Optional[float] = None
    
    # Hidden sugars (calculated)
    hidden_sugars_g: float = 0.0
    
    # Source tracking
    extraction_confidence: float = 1.0
    raw_extraction: Optional[str] = None


@dataclass
class IngredientsList:
    """Standardized ingredients information"""
    ingredients: List[str] = field(default_factory=list)
    allergens: List[str] = field(default_factory=list)
    
    # Sugar indicators
    sugar_ingredients: List[str] = field(default_factory=list)
    artificial_sweeteners: List[str] = field(default_factory=list)
    
    # Sodium indicators
    sodium_ingredients: List[str] = field(default_factory=list)
    
    # Preservatives and additives
    additives: List[str] = field(default_factory=list)
    
    # Extraction metadata
    raw_ingredients_text: Optional[str] = None
    extraction_confidence: float = 1.0


@dataclass
class UserProfile:
    """User-specific medical and lifestyle information"""
    user_id: str
    
    # Demographics
    age: int
    gender: Gender
    weight_kg: float
    height_cm: float
    
    # Medical conditions
    diabetes_type: Optional[DiabetesType] = None
    has_diabetes: bool = False
    hba1c_percentage: Optional[float] = None  # Last recorded
    
    hypertension_severity: HypertensionSeverity = HypertensionSeverity.NORMAL
    systolic_bp: Optional[int] = None        # Last recorded
    diastolic_bp: Optional[int] = None       # Last recorded
    
    # Medications and thresholds
    on_diabetes_medication: bool = False
    on_blood_pressure_medication: bool = False
    insulin_dependent: bool = False
    
    # Lifestyle
    activity_level: ActivityLevel = ActivityLevel.LIGHTLY_ACTIVE
    
    # User health goals and preferences
    primary_goal: UserGoal = UserGoal.BALANCED
    secondary_goals: List[UserGoal] = field(default_factory=list)
    dietary_preference: DietaryPreference = DietaryPreference.NO_PREFERENCE
    cuisine_preference: CuisinePreference = CuisinePreference.MIXED
    
    # Dietary preferences
    cultural_preferences: List[str] = field(default_factory=list)  # e.g., ["Indian", "Vegetarian"]
    allergies: List[str] = field(default_factory=list)
    
    # Medical thresholds (daily recommendations)
    max_daily_sugar_g: Optional[float] = None
    max_daily_sodium_mg: Optional[float] = None
    max_daily_calories: Optional[float] = None
    carb_tolerance_g: Optional[float] = None
    
    # Caregiver contact
    caregiver_email: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ClinicalMetrics:
    """Computed clinical metrics for food items"""
    glycemic_index: Optional[float] = None      # 0-100 scale
    glycemic_load: Optional[float] = None       # Per serving
    sodium_load_percentage: float = 0.0         # % of daily allowance
    sugar_load_percentage: float = 0.0          # % of daily allowance
    caloric_percentage: float = 0.0             # % of daily allowance
    
    # Macronutrient ratios
    carb_to_fiber_ratio: float = 0.0
    sugar_per_100cal: float = 0.0               # Hidden sugar indicator
    sodium_per_100cal: float = 0.0
    
    # Nutrient density score (0-100, higher better)
    nutrient_density_score: float = 0.0
    
    # Medical reasoning
    reasoning_text: str = ""
    risk_factors: List[str] = field(default_factory=list)


@dataclass
class FoodItem:
    """Complete food item with extracted and computed data"""
    item_id: str
    name: str
    brand: str
    category: str
    
    # Extracted data
    nutrition_facts: NutritionFacts
    ingredients: IngredientsList
    
    # Computed metrics
    clinical_metrics: ClinicalMetrics
    
    # Classification and recommendations
    classification: FoodClassification
    classification_score: float = 0.0  # 0-1, confidence
    
    # Explanation
    explanation: str = ""
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    extracted_at: datetime = field(default_factory=datetime.now)
    image_path: Optional[str] = None
    ocr_confidence: float = 1.0


@dataclass
class SubstitutionRecommendation:
    """Alternative food recommendation"""
    original_item_id: str
    substitute_items: List[FoodItem]  # Ranked by similarity and safety
    
    # Comparison metrics
    sugar_reduction_percent: float = 0.0
    sodium_reduction_percent: float = 0.0
    calorie_reduction_percent: float = 0.0
    
    reasoning: str = ""


@dataclass
class FraudDetectionResult:
    """Fraud detection findings"""
    item_id: str
    fraud_flags: List[str] = field(default_factory=list)
    
    # Specific fraud indicators
    unrealistic_nutrition_claim: bool = False
    serving_size_manipulation: bool = False
    missing_ingredient_allergen: bool = False
    inconsistent_nutrition: bool = False
    
    fraud_confidence: float = 0.0  # 0-1
    details: str = ""
    market_range: Optional[Dict] = None  # Comparable products' ranges


@dataclass
class MealSimulationResult:
    """Multi-food meal impact analysis"""
    meal_id: str
    foods: List[FoodItem]
    
    # Aggregated metrics
    total_calories: float = 0.0
    total_carbs_g: float = 0.0
    total_sugars_g: float = 0.0
    total_sodium_mg: float = 0.0
    total_fiber_g: float = 0.0
    total_protein_g: float = 0.0
    
    # Clinical impact
    estimated_glycemic_load: float = 0.0
    sodium_load_percentage: float = 0.0
    
    # Overall classification
    meal_classification: FoodClassification = FoodClassification.MODERATE
    meal_safety_score: float = 0.0  # 0-100
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ComplianceReport:
    """Weekly compliance summary for caregivers"""
    user_id: str
    report_start_date: datetime
    report_end_date: datetime
    
    # Tracking metrics
    total_meals_logged: int = 0
    compliant_meals_count: int = 0
    compliance_percentage: float = 0.0
    
    # Aggregated consumption
    average_daily_sugar_g: float = 0.0
    average_daily_sodium_mg: float = 0.0
    average_daily_calories: float = 0.0
    
    # Violations
    sugar_threshold_violations: int = 0
    sodium_threshold_violations: int = 0
    
    # Summary
    summary_text: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    generated_at: datetime = field(default_factory=datetime.now)


# ===================== CONSTANTS =====================

# Medical thresholds for daily intake (WHO/ADA recommendations)
MEDICAL_THRESHOLDS = {
    "daily_sugar_grams": 25,              # ADA recommendation for added sugars
    "daily_sodium_mg": 2300,              # WHO recommendation
    "daily_calories": 2000,               # General reference
    "max_glycemic_load": 100,             # Per day
}

# Glycemic index ranges
GI_RANGES = {
    "low": (0, 55),
    "medium": (55, 70),
    "high": (70, 100),
}

# Sodium alert thresholds per serving
SODIUM_ALERTS = {
    "high": 400,      # mg per serving
    "medium": 200,
    "low": 0,
}

# Sugar alert thresholds per serving
SUGAR_ALERTS = {
    "high": 12,       # grams per serving
    "medium": 6,
    "low": 0,
}

# Hidden sugar indicators (keywords in ingredients)
HIDDEN_SUGAR_KEYWORDS = [
    "sugar", "glucose", "fructose", "sucrose", "dextrose", "maltose",
    "honey", "syrup", "molasses", "fruit juice concentrate", "cane juice",
    "agave", "stevia", "aspartame", "sorbitol", "xylitol", "erythritol",
    "sugar alcohols", "concentrated fruit juice", "malt", "barley",
]

# Sodium-rich ingredients
HIGH_SODIUM_KEYWORDS = [
    "salt", "sodium", "brine", "cured", "salted", "soy sauce",
    "teriyaki", "worcestershire", "sauerkraut", "pickled",
    "monosodium glutamate", "msg", "preserved", "smoked",
]
