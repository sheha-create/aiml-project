"""
Comprehensive Demo and Usage Guide for Food Label Analysis System
Demonstrates all major features with real-world examples
"""
import json
from datetime import datetime, timedelta
from src.config import (
    UserProfile, DiabetesType, HypertensionSeverity, ActivityLevel, Gender,
    FoodItem, NutritionFacts, IngredientsList, ClinicalMetrics, FoodClassification
)
from src.clinical_metrics.metrics_calculator import ClinicalMetricsComputer
from src.classification.classifier import FoodClassifier
from src.fraud_detection.fraud_detector import FraudDetectionEngine
from src.substitution_engine.recommender import SubstitutionRecommendationEngine
from src.meal_simulation.simulator import MealSimulator
from src.compliance_tracking.tracker import DailyConsumptionTracker, ComplianceReportGenerator


# ===================== DEMO 1: USER PROFILE CREATION =====================

def demo_user_profile_creation():
    """
    Demonstrates creating user profiles with medical information
    """
    print("\\n" + "="*70)
    print("DEMO 1: USER PROFILE CREATION FOR DIABETIC PATIENT")
    print("="*70 + "\\n")
    
    # Create a Type 2 Diabetic with Hypertension
    user_profile = UserProfile(
        user_id="USER001",
        age=52,
        gender=Gender.MALE,
        weight_kg=85,
        height_cm=175,
        diabetes_type=DiabetesType.TYPE_2,
        has_diabetes=True,
        hba1c_percentage=7.5,
        hypertension_severity=HypertensionSeverity.STAGE_1,
        systolic_bp=138,
        diastolic_bp=87,
        activity_level=ActivityLevel.MODERATELY_ACTIVE,
        max_daily_sugar_g=25,
        max_daily_sodium_mg=2000,
        max_daily_calories=2000,
        carb_tolerance_g=45,
        caregiver_email="doctor@hospital.com",
    )
    
    print(f"User Profile Created:")
    print(f"  ID: {user_profile.user_id}")
    print(f"  Age: {user_profile.age}, Gender: {user_profile.gender.value}")
    print(f"  Diabetes: {user_profile.diabetes_type.value} (HbA1c: {user_profile.hba1c_percentage}%)")
    print(f"  Hypertension: {user_profile.hypertension_severity.value} (BP: {user_profile.systolic_bp}/{user_profile.diastolic_bp})")
    print(f"  Daily Limits:")
    print(f"    - Sugar: {user_profile.max_daily_sugar_g}g")
    print(f"    - Sodium: {user_profile.max_daily_sodium_mg}mg")
    print(f"    - Calories: {user_profile.max_daily_calories}")
    print(f"    - Carbs per meal: {user_profile.carb_tolerance_g}g")
    
    return user_profile


# ===================== DEMO 2: CLINICAL METRICS COMPUTATION =====================

def demo_clinical_metrics():
    """
    Demonstrates clinical metrics computation for a food item
    """
    print("\\n" + "="*70)
    print("DEMO 2: CLINICAL METRICS COMPUTATION")
    print("="*70 + "\\n")
    
    user_profile = UserProfile(
        user_id="USER002",
        age=45, gender=Gender.FEMALE,
        weight_kg=70, height_cm=165,
        has_diabetes=True,
        diabetes_type=DiabetesType.TYPE_2,
        hypertension_severity=HypertensionSeverity.ELEVATED,
    )
    
    # Create a food item (cookies)
    nutrition = NutritionFacts(
        serving_size_grams=30,
        serving_size_unit="cookies (2)",
        calories=150,
        total_carbs_g=20,
        dietary_fiber_g=1,
        sugars_g=10,
        sodium_mg=180,
        protein_g=2,
    )
    
    print("Food Item: Cookies (2 pieces)")
    print(f"Nutrition Facts (per 30g serving):")
    print(f"  Calories: {nutrition.calories}")
    print(f"  Carbs: {nutrition.total_carbs_g}g")
    print(f"  Sugar: {nutrition.sugars_g}g")
    print(f"  Sodium: {nutrition.sodium_mg}mg")
    print(f"  Fiber: {nutrition.dietary_fiber_g}g")
    
    # Compute metrics
    computer = ClinicalMetricsComputer(user_profile)
    metrics = computer.compute_metrics(nutrition, "Cookies", user_profile)
    
    print(f"\\nClinical Metrics Computed:")
    print(f"  Glycemic Index: {metrics.glycemic_index}")
    print(f"  Glycemic Load: {metrics.glycemic_load}")
    print(f"  Sugar Load: {metrics.sugar_load_percentage:.1f}% of daily allowance")
    print(f"  Sodium Load: {metrics.sodium_load_percentage:.1f}% of daily allowance")
    print(f"  Nutrient Density Score: {metrics.nutrient_density_score:.1f}/100")
    print(f"\\nRisk Factors Identified:")
    for risk in metrics.risk_factors:
        print(f"  ⚠️  {risk}")
    
    return metrics


# ===================== DEMO 3: FOOD CLASSIFICATION =====================

def demo_food_classification():
    """
    Demonstrates food classification based on user profile
    """
    print("\\n" + "="*70)
    print("DEMO 3: PERSONALIZED FOOD CLASSIFICATION")
    print("="*70 + "\\n")
    
    user_profile = UserProfile(
        user_id="USER003",
        age=55, gender=Gender.MALE,
        weight_kg=92, height_cm=180,
        has_diabetes=True,
        diabetes_type=DiabetesType.TYPE_2,
        hypertension_severity=HypertensionSeverity.STAGE_2,
        max_daily_sugar_g=25,
        max_daily_sodium_mg=1500,
    )
    
    # Example foods to classify
    foods_to_test = [
        ("Whole Wheat Bread", NutritionFacts(50, "g", 130, 25, 4, 3, 380, 5)),
        ("White Bread", NutritionFacts(50, "g", 150, 28, 1.5, 5, 450, 4)),
        ("Fried Chips", NutritionFacts(28, "g", 150, 15, 2, 0, 180, 2)),
    ]
    
    classifier = FoodClassifier()
    
    print(f"Classifying foods for patient with Type 2 Diabetes + Stage 2 Hypertension\\n")
    
    for food_name, nutrition in foods_to_test:
        computer = ClinicalMetricsComputer(user_profile)
        metrics = computer.compute_metrics(nutrition, food_name, user_profile)
        
        food_item = FoodItem(
            item_id=f"food_{food_name.replace(' ', '_')}",
            name=food_name,
            brand="Demo",
            category="Bread" if "Bread" in food_name else "Snacks",
            nutrition_facts=nutrition,
            ingredients=IngredientsList(),
            clinical_metrics=metrics,
            classification=FoodClassification.SUITABLE,
        )
        
        classification, confidence, explanation = classifier.classify_food(food_item, user_profile)
        
        print(f"{food_name}:")
        print(f"  Classification: {classification.value.upper()}")
        print(f"  Confidence: {confidence:.0%}")
        print(f"  Explanation: {explanation[:150]}...")
        print()


# ===================== DEMO 4: FRAUD DETECTION =====================

def demo_fraud_detection():
    """
    Demonstrates fraud detection for unrealistic nutrition claims
    """
    print("\\n" + "="*70)
    print("DEMO 4: FRAUD DETECTION - UNREALISTIC CLAIMS")
    print("="*70 + "\\n")
    
    fraud_engine = FraudDetectionEngine()
    
    # Suspicious food item with unrealistic claims
    suspicious_nutrition = NutritionFacts(
        serving_size_grams=8,  # Suspiciously small
        calories=50,           # Unrealistically low for cookies
        total_carbs_g=2,
        dietary_fiber_g=5,     # Too high for cookies
        sugars_g=0,            # Unlikely for cookies
        sodium_mg=10,          # Too low for packaged food
        protein_g=3,
    )
    
    ingredients = IngredientsList(
        ingredients=["Sugar", "Flour", "Butter", "Eggs"],
    )
    
    suspicious_food = FoodItem(
        item_id="fraud_test_1",
        name="Miracle Diet Cookies",
        brand="SuspiciousBrand",
        category="cookies",
        nutrition_facts=suspicious_nutrition,
        ingredients=ingredients,
        clinical_metrics=ClinicalMetrics(),
        classification=FoodClassification.SUITABLE,
    )
    
    fraud_result = fraud_engine.analyze_for_fraud(suspicious_food, "cookies")
    
    print(f"Food Item: {suspicious_food.name}")
    print(f"Fraud Confidence: {fraud_result.fraud_confidence:.0%}")
    print(f"\\nDetected Issues:")
    for flag in fraud_result.fraud_flags:
        print(f"  🚨 {flag}")
    
    if fraud_result.fraud_confidence > 0.3:
        print("\\n⚠️  FRAUD WARNING: This product should be investigated by authorities!")


# ===================== DEMO 5: SUBSTITUTION RECOMMENDATIONS =====================

def demo_substitution_recommendations():
    """
    Demonstrates food substitution recommendations
    """
    print("\\n" + "="*70)
    print("DEMO 5: HEALTHIER SUBSTITUTION RECOMMENDATIONS")
    print("="*70 + "\\n")
    
    user_profile = UserProfile(
        user_id="USER005",
        age=48, gender=Gender.MALE,
        weight_kg=88, height_cm=178,
        has_diabetes=True,
        diabetes_type=DiabetesType.TYPE_2,
        max_daily_sugar_g=25,
    )
    
    # Original unhealthy food
    original_nutrition = NutritionFacts(
        serving_size_grams=250,
        calories=280,
        total_carbs_g=45,
        dietary_fiber_g=1,
        sugars_g=22,
        sodium_mg=800,
        protein_g=8,
    )
    
    original_food = FoodItem(
        item_id="food_soft_drink",
        name="Regular Cola",
        brand="PopCo",
        category="beverages",
        nutrition_facts=original_nutrition,
        ingredients=IngredientsList(),
        clinical_metrics=ClinicalMetrics(),
        classification=FoodClassification.AVOID,
    )
    
    # Healthier alternatives
    alternative1_nutrition = NutritionFacts(
        serving_size_grams=250,
        calories=25,
        total_carbs_g=6,
        dietary_fiber_g=0,
        sugars_g=1,
        sodium_mg=50,
        protein_g=0,
    )
    
    alternative1 = FoodItem(
        item_id="food_diet_cola",
        name="Diet Cola",
        brand="PopCo",
        category="beverages",
        nutrition_facts=alternative1_nutrition,
        ingredients=IngredientsList(),
        clinical_metrics=ClinicalMetrics(),
        classification=FoodClassification.SUITABLE,
    )
    
    # Build recommendation engine
    engine = SubstitutionRecommendationEngine([alternative1])
    
    print(f"Original Food: {original_food.name}")
    print(f"  Sugar: {original_nutrition.sugars_g}g")
    print(f"  Sodium: {original_nutrition.sodium_mg}mg")
    print(f"  Calories: {original_nutrition.calories}")
    
    recommendation = engine.find_substitutes(original_food, user_profile, max_suggestions=1)
    
    if recommendation.substitute_items:
        substitute = recommendation.substitute_items[0]
        print(f"\\nRecommended Alternative: {substitute.name}")
        print(f"  Sugar: {substitute.nutrition_facts.sugars_g}g ({recommendation.sugar_reduction_percent:+.0f}%)")
        print(f"  Sodium: {substitute.nutrition_facts.sodium_mg}mg ({recommendation.sodium_reduction_percent:+.0f}%)")
        print(f"  Calories: {substitute.nutrition_facts.calories} ({recommendation.calorie_reduction_percent:+.0f}%)")
        print(f"\\nReasoning: {recommendation.reasoning[:200]}...")


# ===================== DEMO 6: MEAL SIMULATION =====================

def demo_meal_simulation():
    """
    Demonstrates multi-food meal impact simulation
    """
    print("\\n" + "="*70)
    print("DEMO 6: MEAL SIMULATION AND IMPACT ANALYSIS")
    print("="*70 + "\\n")
    
    user_profile = UserProfile(
        user_id="USER006",
        age=50, gender=Gender.FEMALE,
        weight_kg=75, height_cm=162,
        has_diabetes=True,
        diabetes_type=DiabetesType.TYPE_2,
        carb_tolerance_g=45,
        max_daily_sugar_g=25,
        max_daily_sodium_mg=2300,
    )
    
    # Create meal items
    rice_nutrition = NutritionFacts(150, "g", 200, 42, 1, 2, 300, 4)
    chicken_nutrition = NutritionFacts(100, "g", 165, 0, 0, 0, 75, 31)
    veggies_nutrition = NutritionFacts(100, "g", 25, 5, 2, 1, 50, 2)
    
    rice_food = FoodItem("rice", "Brown Rice", "Generic", "Grains",
                        rice_nutrition, IngredientsList(), ClinicalMetrics(), FoodClassification.MODERATE)
    chicken_food = FoodItem("chicken", "Grilled Chicken", "Generic", "Protein",
                           chicken_nutrition, IngredientsList(), ClinicalMetrics(), FoodClassification.SUITABLE)
    veggies_food = FoodItem("veggies", "Steamed Broccoli", "Generic", "Vegetables",
                           veggies_nutrition, IngredientsList(), ClinicalMetrics(), FoodClassification.SUITABLE)
    
    # Simulate meal
    simulator = MealSimulator(user_profile)
    meal = simulator.simulate_meal([rice_food, chicken_food, veggies_food], "Lunch", user_profile)
    
    print(f"Meal: {[f.name for f in meal.foods]}")
    print(f"\\nNutrition Totals:")
    print(f"  Calories: {meal.total_calories:.0f}")
    print(f"  Carbs: {meal.total_carbs_g:.0f}g (carb target: {user_profile.carb_tolerance_g}g)")
    print(f"  Sugar: {meal.total_sugars_g:.1f}g")
    print(f"  Sodium: {meal.total_sodium_mg:.0f}mg")
    print(f"  Fiber: {meal.total_fiber_g:.1f}g")
    print(f"  Protein: {meal.total_protein_g:.0f}g")
    
    print(f"\\nMeal Assessment:")
    print(f"  Estimated Glycemic Load: {meal.estimated_glycemic_load:.1f}")
    print(f"  Safety Score: {meal.meal_safety_score:.0f}/100")
    print(f"  Classification: {meal.meal_classification.value}")
    
    if meal.recommendations:
        print(f"\\nRecommendations:")
        for rec in meal.recommendations:
            print(f"  • {rec}")


# ===================== DEMO 7: COMPLIANCE TRACKING =====================

def demo_compliance_tracking():
    """
    Demonstrates compliance tracking and caregiver reporting
    """
    print("\\n" + "="*70)
    print("DEMO 7: WEEKLY COMPLIANCE TRACKING FOR CAREGIVERS")
    print("="*70 + "\\n")
    
    user_profile = UserProfile(
        user_id="USER007",
        age=58, gender=Gender.MALE,
        weight_kg=95, height_cm=180,
        has_diabetes=True,
        diabetes_type=DiabetesType.TYPE_2,
        hypertension_severity=HypertensionSeverity.STAGE_1,
        max_daily_sugar_g=25,
        max_daily_sodium_mg=2000,
        caregiver_email="doctor@clinic.com",
    )
    
    # Create tracker
    tracker = DailyConsumptionTracker(user_profile.user_id)
    
    # Simulate 7 days of consumption
    for day_offset in range(7):
        date = datetime.now() - timedelta(days=day_offset)
        
        # Create sample foods for the day
        if day_offset % 2 == 0:
            # Compliant day
            sugar_per_day = 18
            sodium_per_day = 1600
        else:
            # Non-compliant day
            sugar_per_day = 32
            sodium_per_day = 2800
        
        daily_nutrition = NutritionFacts(
            serving_size_grams=100,
            calories=int(600 + day_offset * 50),
            total_carbs_g=80,
            dietary_fiber_g=5,
            sugars_g=sugar_per_day / 2,  # Two meals logged per day
            sodium_mg=sodium_per_day / 2,
            protein_g=25,
        )
        
        food = FoodItem(
            f"food_day_{day_offset}",
            f"Daily Meal {day_offset}",
            "HomeCooked",
            "MixedMeal",
            daily_nutrition,
            IngredientsList(),
            ClinicalMetrics(),
            FoodClassification.MODERATE,
        )
        
        tracker.log_food(food, date)
    
    # Generate weekly report
    generator = ComplianceReportGenerator(tracker)
    report = generator.generate_weekly_report(user_profile)
    
    print(f"Weekly Compliance Report")
    print(f"Period: {report.report_start_date.strftime('%Y-%m-%d')} to {report.report_end_date.strftime('%Y-%m-%d')}")
    print(f"\\nMetrics:")
    print(f"  Compliance Percentage: {report.compliance_percentage:.0f}%")
    print(f"  Meals Logged: {report.total_meals_logged}")
    print(f"  Average Daily Sugar: {report.average_daily_sugar_g:.1f}g (target: 25g)")
    print(f"  Average Daily Sodium: {report.average_daily_sodium_mg:.0f}mg (target: 2000mg)")
    print(f"\\nViolations:")
    print(f"  Sugar Threshold Exceeded: {report.sugar_threshold_violations} days")
    print(f"  Sodium Threshold Exceeded: {report.sodium_threshold_violations} days")
    print(f"\\nTop Recommendations:")
    for i, rec in enumerate(report.recommendations[:3], 1):
        print(f"  {i}. {rec}")


# ===================== MAIN DEMO RUNNER =====================

def run_all_demos():
    """Run all system demos"""
    print("\\n" + "="*70)
    print("FOOD LABEL ANALYSIS SYSTEM - COMPREHENSIVE DEMO")
    print("For Diabetic and Hypertension Patients")
    print("="*70)
    
    demo_user_profile_creation()
    demo_clinical_metrics()
    demo_food_classification()
    demo_fraud_detection()
    demo_substitution_recommendations()
    demo_meal_simulation()
    demo_compliance_tracking()
    
    print("\\n" + "="*70)
    print("ALL DEMOS COMPLETED SUCCESSFULLY")
    print("="*70 + "\\n")


if __name__ == "__main__":
    run_all_demos()
