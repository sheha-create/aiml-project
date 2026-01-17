import streamlit as st
import tempfile
import os
import sys
from pathlib import Path

# SET PAGE CONFIG FIRST (required by Streamlit)
st.set_page_config(page_title="Food Label Analyzer", layout="wide")



from food_label_analyzer.src.data_loader import get_data_loader

# Load data
@st.cache_resource
def load_data():
    return get_data_loader()

data_loader = load_data()

st.title("Food Label Analysis — UI")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Analyze Label", "Substitutions", "Meal Substitutions", "Meal Simulator", "Weekly Report"])



# Helper: attempt imports from project modules
def try_imports():
    imports = {}
    try:
        from food_label_analyzer.src.ocr_engine.label_ocr import NutritionLabelOCR
        imports['ocr'] = NutritionLabelOCR
    except Exception:
        imports['ocr'] = None
    try:
        from food_label_analyzer.src.classification.classifier import FoodClassifier
        imports['classifier'] = FoodClassifier
    except Exception:
        imports['classifier'] = None
    try:
        from food_label_analyzer.src.substitution_engine.recommender import SubstitutionRecommendationEngine
        imports['recommender'] = SubstitutionRecommendationEngine
    except Exception:
        imports['recommender'] = None
    try:
        from food_label_analyzer.src.meal_simulation.simulator import MealSimulator
        imports['simulator'] = MealSimulator
    except Exception:
        imports['simulator'] = None
    try:
        from food_label_analyzer.src.compliance_tracking.tracker import DailyConsumptionTracker
        imports['tracker'] = DailyConsumptionTracker
    except Exception:
        imports['tracker'] = None
    return imports

IMPORTS = try_imports()

if page == "Home":
    st.header("Welcome to Food Label Analyzer")
    st.write("Comprehensive AI-driven food analysis for diabetic and hypertension patients.")
    st.markdown("---")
    
    # Show available data
    col1, col2, col3 = st.columns(3)
    with col1:
        foods = data_loader.get_all_foods()
        st.metric("Foods in Database", len(foods))
    with col2:
        users = data_loader.get_sample_users()
        st.metric("Sample User Profiles", len(users))
    with col3:
        gi_foods = len(data_loader.gi_db) if data_loader.gi_db is not None else 0
        st.metric("GI Reference Data", gi_foods)
    
    st.markdown("---")
    
    with st.expander("📊 Sample Foods Available"):
        foods = data_loader.get_all_foods()[:20]
        for food in foods:
            st.write(f"**{food.food_name}** | {food.calories} cal | {food.carbs_g}g carbs | {food.sugars_g}g sugar")
    
    with st.expander("👥 Sample Users"):
        users_df = data_loader.user_profiles_db
        if users_df is not None and len(users_df) > 0:
            st.dataframe(users_df[['user_id', 'name', 'age', 'has_diabetes', 'diabetes_type', 'hypertension_severity']])
    
    st.info("Use the sidebar to navigate features. The system includes 60+ foods, 50+ GI records, and 10 sample user profiles for testing.")

if page == "Analyze Label":
    st.header("📸 Analyze a Food Label with OCR")
    st.write("Extract nutrition facts from food package labels using AI-powered Optical Character Recognition (OCR).")
    
    st.divider()
    
    # Instructions section
    with st.expander("📋 **How to Use - Instructions & Requirements**", expanded=True):
        st.subheader("Step-by-Step Guide:")
        st.markdown("""
        1. **Prepare Your Image**
           - Take a clear photo of the **Nutrition Facts** panel on the food package
           - Ensure the text is readable (no blurriness or shadows)
           
        2. **Upload the Image**
           - Click "Upload a label image" below
           - Select PNG, JPG, JPEG, or TIFF format
           
        3. **Review Results**
           - The system will extract nutrition information
           - Review the extracted values and edit if needed
           - Results will include calories, sugars, sodium, fiber, and more
        
        4. **Get Personalized Classification** (Optional)
           - Enter your User ID or leave as "demo_user"
           - If you have diabetes or hypertension, the system will provide specific guidance
        """)
        
        st.subheader("Image Requirements:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **✅ What Works Best:**
            - **Resolution**: 300 DPI or higher
            - **Size**: Less than 5 MB
            - **Content**: Focus on Nutrition Facts label
            - **Condition**: Well-lit, no shadows
            - **Angle**: Straight-on (90 degrees)
            - **Text**: Clearly visible black text on white background
            - **Format**: PNG, JPG, JPEG, or TIFF
            - **Language**: English or Hindi supported
            """)
        
        with col2:
            st.markdown("""
            **❌ What to Avoid:**
            - Blurry or out-of-focus photos
            - Photos taken at an angle
            - Very small or compressed images
            - Images with glare or shadows
            - Photos of the entire package
            - Handwritten labels
            - Very old/faded text
            - Files larger than 5 MB
            - Unsupported formats (GIF, BMP, WebP)
            """)
        
        st.subheader("Supported Information:")
        st.markdown("""
        The OCR system can extract:
        - **Calories** per serving
        - **Macronutrients**: Carbs, Protein, Fat (saturated, trans, unsaturated)
        - **Sugars**: Total sugars and added sugars
        - **Fiber**: Dietary fiber
        - **Sodium**: In milligrams
        - **Cholesterol**: In milligrams
        - **Ingredients**: List of all ingredients
        - **Allergens**: Common allergen warnings
        """)
        
        st.subheader("Tips for Best Results:")
        st.markdown("""
        1. Use **natural lighting** - avoid camera flash
        2. **Position the label flat** on a surface
        3. Ensure the **entire label is visible** in the frame
        4. Avoid tilting the camera
        5. Use **high-quality camera** (smartphone camera is fine)
        6. **Wait 3-5 seconds** for OCR processing
        7. **Review extracted values** - manual correction available
        """)
    
    st.divider()
    
    uploaded = st.file_uploader("Upload a label image", type=["png", "jpg", "jpeg", "tiff"])  
    user_profile = st.expander("User profile (optional)")
    with user_profile:
        user_id = st.text_input("User ID", value="demo_user")
    if uploaded:
        st.success(f"✓ Image uploaded: {uploaded.name}")
        
        # Display the uploaded image
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(uploaded, caption="Uploaded Nutrition Label", use_column_width=True)
        with col2:
            st.write("**File Info:**")
            st.write(f"- Name: {uploaded.name}")
            st.write(f"- Size: {uploaded.size / 1024:.1f} KB")
            st.write(f"- Type: {uploaded.type}")
        
        st.divider()
        
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(uploaded.getbuffer())
        tfile.flush()
        tpath = tfile.name

        # OCR Processing
        st.subheader("🔍 OCR Processing")
        
        ocr_cls = IMPORTS.get('ocr')
        if ocr_cls is None:
            st.error("❌ OCR module not available. Ensure project modules are importable and dependencies are installed.")
            st.write("**Path:** ", tpath)
        else:
            with st.spinner("⏳ Extracting nutrition information from label..."):
                try:
                    ocr = ocr_cls()
                    ocr_result = ocr.extract_from_label(tpath)
                    st.success("✓ OCR extraction completed!")
                    
                    st.subheader("📊 Extracted Nutrition Information")
                    
                    # Convert OCRResult to dictionary for easier handling
                    from food_label_analyzer.src.ocr_engine.label_ocr import NutritionFactsParser
                    parser = NutritionFactsParser()
                    nutrition_text = ocr_result.detected_regions.get('nutrition_facts', ocr_result.raw_text)
                    nutrition_facts = parser.parse_nutrition_facts(nutrition_text)
                    ingredients_obj = parser.parse_ingredients(ocr_result.detected_regions.get('ingredients', ''))
                    
                    # Build parsed dict from extracted nutrition facts
                    parsed = {
                        'calories': nutrition_facts.calories if nutrition_facts.calories else 0,
                        'total_carbs_g': nutrition_facts.total_carbs_g if nutrition_facts.total_carbs_g else 0,
                        'sugars_g': nutrition_facts.sugars_g if nutrition_facts.sugars_g else 0,
                        'protein_g': nutrition_facts.protein_g if nutrition_facts.protein_g else 0,
                        'total_fat_g': nutrition_facts.total_fat_g if nutrition_facts.total_fat_g else 0,
                        'sodium_mg': nutrition_facts.sodium_mg if nutrition_facts.sodium_mg else 0,
                        'dietary_fiber_g': nutrition_facts.dietary_fiber_g if nutrition_facts.dietary_fiber_g else 0,
                        'cholesterol_mg': nutrition_facts.cholesterol_mg if nutrition_facts.cholesterol_mg else 0,
                        'ingredients': ingredients_obj.ingredients if ingredients_obj else [],
                        'allergens': ingredients_obj.allergens if ingredients_obj else [],
                        'raw_text': ocr_result.raw_text,
                        'confidence': ocr_result.confidence,
                    }
                    
                    # Display results in readable format
                    if isinstance(parsed, dict):
                        # Count available nutrients
                        tracked_nutrients = ['calories', 'total_carbs_g', 'sugars_g', 'protein_g', 'total_fat_g', 'sodium_mg', 'dietary_fiber_g', 'cholesterol_mg']
                        available_nutrients = sum(1 for key in tracked_nutrients if parsed.get(key, 0) > 0)
                        total_nutrients = len(tracked_nutrients)
                        missing_nutrients = [key.replace('_g', '').replace('_mg', '').replace('_', ' ').title() for key in tracked_nutrients if parsed.get(key, 0) == 0]
                        
                        if available_nutrients < total_nutrients:
                            st.info(f"📊 Partial data: {available_nutrients}/{total_nutrients} nutrients found. Missing: {', '.join(missing_nutrients[:3])}{'...' if len(missing_nutrients) > 3 else ''}. This is normal - the app analyzes available data.")
                        
                        # Nutrition facts section
                        if 'nutrition_facts' in parsed or isinstance(parsed, dict):
                            cols = st.columns(3)
                            nutrition_items = [
                                ('calories', 'Calories', ''),
                                ('total_carbs_g', 'Carbs', 'g'),
                                ('sugars_g', 'Sugars', 'g'),
                                ('protein_g', 'Protein', 'g'),
                                ('total_fat_g', 'Total Fat', 'g'),
                                ('sodium_mg', 'Sodium', 'mg'),
                                ('dietary_fiber_g', 'Fiber', 'g'),
                                ('cholesterol_mg', 'Cholesterol', 'mg'),
                            ]
                            
                            col_idx = 0
                            for key, label, unit in nutrition_items:
                                value = parsed.get(key, 0)
                                # Only display if value is greater than 0
                                if value and value > 0:
                                    col = cols[col_idx % 3]
                                    if unit:
                                        col.metric(label, f"{value:.1f} {unit}")
                                    else:
                                        col.metric(label, f"{value:.0f}")
                                    col_idx += 1
                            
                            # Show note if some nutrients are missing
                            if col_idx < len(nutrition_items):
                                st.caption(f"ℹ️ Showing {col_idx}/{len(nutrition_items)} extracted nutrients. Other values were not detected in the label.")
                        
                        st.divider()
                        
                        # Ingredients section
                        if 'ingredients' in parsed and parsed['ingredients']:
                            with st.expander("🥄 Ingredients List", expanded=False):
                                if isinstance(parsed['ingredients'], list):
                                    for ing in parsed['ingredients']:
                                        st.write(f"• {ing}")
                                else:
                                    st.write(parsed['ingredients'])
                        
                        # Allergens section
                        if 'allergens' in parsed and parsed['allergens']:
                            with st.expander("⚠️ Allergen Information", expanded=False):
                                if isinstance(parsed['allergens'], list):
                                    for allergen in parsed['allergens']:
                                        st.warning(f"⚠️ Contains: {allergen}")
                                else:
                                    st.warning(f"⚠️ Contains: {parsed['allergens']}")
                        
                        st.divider()
                        
                        # Raw JSON for developers
                        with st.expander("💻 Raw OCR Data (JSON)", expanded=False):
                            st.json(parsed)
                    else:
                        st.json(parsed)
                    
                except Exception as e:
                    st.error(f"❌ OCR processing failed: {str(e)}")
                    st.info("**Troubleshooting:**\n- Ensure the image is clear and well-lit\n- Try a different angle or lighting\n- Check that the nutrition label is the main focus of the image")

        # Classification (only if OCR succeeded)
        if 'parsed' in locals():
            classifier_cls = IMPORTS.get('classifier')
            if classifier_cls is None:
                st.warning("Classifier module not available.")
            else:
                st.info("✓ Classification available (optional)")

        # cleanup temp file
        try:
            os.unlink(tpath)
        except Exception:
            pass

if page == "Substitutions":
    st.header("🔄 Smart Food Substitutions")
    st.write("Find healthier alternatives tailored to your health goals and medical profile.")
    
    # Get list of available foods
    foods = data_loader.get_all_foods()
    food_names = {f.food_name: f for f in foods}
    
    # User goal selection
    st.subheader("Your Health Goals")
    col1, col2 = st.columns(2)
    
    with col1:
        from food_label_analyzer.src.config import UserGoal
        selected_goal = st.selectbox(
            "Primary Health Goal:",
            options=[g.value for g in UserGoal],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    with col2:
        has_diabetes = st.checkbox("I have Diabetes")
        has_hypertension = st.checkbox("I have Hypertension")
    
    st.divider()
    
    # Food selection
    selected_food_name = st.selectbox("Select a food to find alternatives:", list(food_names.keys()))
    
    if st.button("Find Smart Substitutes"):
        original_food = food_names[selected_food_name]
        
        # Create user profile for recommendation
        from food_label_analyzer.src.config import UserProfile, UserGoal, Gender, HypertensionSeverity
        
        hypertension_severity = HypertensionSeverity.STAGE_1 if has_hypertension else HypertensionSeverity.NORMAL
        user = UserProfile(
            user_id="demo_user",
            age=50,
            gender=Gender.MALE,
            weight_kg=80,
            height_cm=175,
            has_diabetes=has_diabetes,
            hypertension_severity=hypertension_severity,
            primary_goal=UserGoal(selected_goal)
        )
        
        # Get substitutions using advanced engine
        from food_label_analyzer.src.substitution_engine.advanced_recommender import AdvancedSubstitutionEngine
        engine = AdvancedSubstitutionEngine()
        substitutes = engine.find_substitutes(original_food, user, top_n=5)
        
        # Display original food
        st.subheader(f"📌 Original: {original_food.food_name}")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Calories", f"{original_food.calories:.0f}")
        with col2:
            st.metric("Sugar", f"{original_food.sugars_g:.1f}g")
        with col3:
            st.metric("Sodium", f"{original_food.sodium_mg:.0f}mg")
        with col4:
            st.metric("Fiber", f"{original_food.fiber_g:.1f}g")
        with col5:
            health_score = engine.calculate_health_score(original_food, user)
            st.metric("Health Score", f"{health_score:.0f}/100")
        
        st.divider()
        
        # Display substitutes
        if substitutes:
            st.subheader("✅ Top Substitutes")
            
            for idx, sub in enumerate(substitutes, 1):
                with st.expander(f"#{idx} {sub.food_name} (Score: {sub.total_score:.2f})", expanded=(idx == 1)):
                    # Badges
                    badge_text = " ".join(sub.badges) if sub.badges else "No special badges"
                    st.write(f"**Badges:** {badge_text}")
                    
                    # Nutritional comparison
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Nutritional Deltas:**")
                        delta_text = f"""
- Sugar: {sub.sugar_delta_g:+.1f}g {('⬇️ reduction' if sub.sugar_delta_g > 0 else '⬆️ increase')}
- Net Carbs: {sub.net_carbs_delta_g:+.1f}g
- GI: {sub.gi_delta:+.0f} points
- Sodium: {sub.sodium_delta_mg:+.0f}mg
- Calories: {sub.calorie_delta:+.0f}
                        """
                        st.markdown(delta_text)
                    
                    with col2:
                        st.write("**Substitute Stats:**")
                        stats_text = f"""
- Calories: {sub.food_name} has **{sub.food_name}** cal
- Sugar: **{sub.food_name}g**
- Sodium: **{sub.food_name}mg**
- Fiber: **{sub.food_name}g**
                        """
                        st.metric("Substitute Calories", f"{original_food.calories - sub.calorie_delta:.0f}")
                        st.metric("Substitute Health Score", f"{sub.health_score_substitute:.0f}/100")
                    
                    with col3:
                        st.write("**Health Improvement:**")
                        improvement = sub.health_score_substitute - sub.health_score_original
                        st.metric("Score Improvement", f"{improvement:+.0f}", delta=f"{improvement:.0f}")
                        st.write(f"**Why?** {sub.reasoning}")
                    
                    # Component scores breakdown
                    st.write("**Recommendation Scores:**")
                    score_cols = st.columns(6)
                    with score_cols[0]:
                        st.write(f"Sugar: {sub.sugar_score:.2f}")
                    with score_cols[1]:
                        st.write(f"Net Carbs: {sub.net_carbs_score:.2f}")
                    with score_cols[2]:
                        st.write(f"GI: {sub.gi_score:.2f}")
                    with score_cols[3]:
                        st.write(f"Sodium: {sub.sodium_score:.2f}")
                    with score_cols[4]:
                        st.write(f"Calories: {sub.calorie_score:.2f}")
                    with score_cols[5]:
                        st.write(f"Similarity: {sub.similarity_score:.2f}")
        else:
            st.info("No suitable substitutes found in similar food categories.")

if page == "Meal Substitutions":
    st.header("🍽️ Meal-Level Substitutions")
    st.write("Optimize entire meals by substituting individual foods based on your health goals.")
    
    from food_label_analyzer.src.config import UserGoal, Gender, HypertensionSeverity, UserProfile
    from food_label_analyzer.src.substitution_engine.meal_substitution import MealContextSimulator, MealType
    
    foods = data_loader.get_all_foods()
    food_names = {f.food_name: f for f in foods}
    
    # Health profile
    st.subheader("Your Health Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_goal = st.selectbox("Primary Goal:", options=[g.value for g in UserGoal], format_func=lambda x: x.replace('_', ' ').title(), key="meal_goal")
    with col2:
        has_diabetes = st.checkbox("I have Diabetes", key="meal_diabetes")
    with col3:
        has_hypertension = st.checkbox("I have Hypertension", key="meal_hypertension")
    
    # Meal selection
    st.subheader("Build Your Meal")
    
    meal_type = st.selectbox("Select meal type:", [m.value for m in MealType], format_func=lambda x: x.title())
    selected_foods = st.multiselect("Select foods in this meal:", list(food_names.keys()), key="meal_foods")
    
    if st.button("Optimize Meal"):
        if selected_foods:
            # Create user profile
            hypertension_severity = HypertensionSeverity.STAGE_1 if has_hypertension else HypertensionSeverity.NORMAL
            user = UserProfile(
                user_id="demo_user",
                age=50,
                gender=Gender.MALE,
                weight_kg=80,
                height_cm=175,
                has_diabetes=has_diabetes,
                hypertension_severity=hypertension_severity,
                primary_goal=UserGoal(selected_goal)
            )
            
            # Get food objects
            meal_foods = [food_names[name] for name in selected_foods]
            
            # Simulate substitutions
            simulator = MealContextSimulator()
            result = simulator.simulate_meal_substitution(
                MealType(meal_type), meal_foods, user
            )
            
            st.divider()
            
            # Display comparison
            st.subheader("📊 Meal Comparison")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Original Meal**")
                st.write(f"Calories: {result.original_calories:.0f}")
                st.write(f"Sugar: {result.original_sugars:.1f}g")
                st.write(f"Sodium: {result.original_sodium:.0f}mg")
                st.write(f"Fiber: {result.original_fiber:.1f}g")
                st.write(f"Health Score: {result.original_health_score:.0f}/100")
            
            with col2:
                st.write("**Optimized Meal**")
                st.write(f"Calories: {result.substituted_calories:.0f}")
                st.write(f"Sugar: {result.substituted_sugars:.1f}g")
                st.write(f"Sodium: {result.substituted_sodium:.0f}mg")
                st.write(f"Fiber: {result.substituted_fiber:.1f}g")
                st.write(f"Health Score: {result.substituted_health_score:.0f}/100")
            
            with col3:
                st.write("**Improvements**")
                st.metric("Calories", f"{result.calorie_reduction_pct:+.0f}%")
                st.metric("Sugar", f"{result.sugar_reduction_pct:+.0f}%")
                st.metric("Sodium", f"{result.sodium_reduction_pct:+.0f}%")
                st.metric("Fiber", f"{result.fiber_increase_pct:+.0f}%")
                st.metric("Health Score", f"{result.health_improvement:+.0f}", delta=f"{result.health_improvement:.0f}")
            
            st.divider()
            
            # Display substitutions
            st.subheader("🔄 Food Substitutions")
            for original_food_id, substitute_score in result.substitutions.items():
                original_food = next(f for f in meal_foods if f.food_id == original_food_id)
                with st.expander(f"{original_food.food_name} → {substitute_score.food_name}"):
                    st.write(f"**Why:** {substitute_score.reasoning}")
                    st.write(f"**Badges:** {' '.join(substitute_score.badges)}")
                    st.write(f"- Sugar: {original_food.sugars_g:.1f}g → {original_food.sugars_g - substitute_score.sugar_delta_g:.1f}g ({substitute_score.sugar_delta_g:+.1f}g)")
                    st.write(f"- Sodium: {original_food.sodium_mg:.0f}mg → {original_food.sodium_mg - substitute_score.sodium_delta_mg:.0f}mg ({substitute_score.sodium_delta_mg:+.0f}mg)")
                    st.write(f"- Calories: {original_food.calories:.0f} → {original_food.calories - substitute_score.calorie_delta:.0f} ({substitute_score.calorie_delta:+.0f})")
        else:
            st.info("Select at least one food to optimize.")

if page == "Meal Simulator":
    st.header("Meal Simulation")
    st.write("Build a meal and see the combined nutritional impact.")
    
    foods = data_loader.get_all_foods()
    food_names = [f.food_name for f in foods]
    
    meal_foods = st.multiselect("Select foods for your meal:", food_names)
    
    if st.button("Analyze Meal"):
        if meal_foods:
            # Calculate totals
            total_calories = 0
            total_sugars = 0
            total_sodium = 0
            total_fiber = 0
            total_carbs = 0
            meal_details = []
            
            for food_name in meal_foods:
                matching = [f for f in foods if f.food_name == food_name]
                if matching:
                    food = matching[0]
                    total_calories += food.calories
                    total_sugars += food.sugars_g
                    total_sodium += food.sodium_mg
                    total_fiber += food.fiber_g
                    total_carbs += food.carbs_g
                    meal_details.append({
                        'Food': food.food_name,
                        'Calories': food.calories,
                        'Sugars (g)': food.sugars_g,
                        'Sodium (mg)': food.sodium_mg,
                        'Fiber (g)': food.fiber_g
                    })
            
            # Display results
            st.subheader("Meal Totals:")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Calories", f"{total_calories:.0f}")
            with col2:
                st.metric("Sugars", f"{total_sugars:.1f}g")
            with col3:
                st.metric("Sodium", f"{total_sodium:.0f}mg")
            with col4:
                st.metric("Fiber", f"{total_fiber:.1f}g")
            
            # Food breakdown
            st.subheader("Food Breakdown:")
            st.dataframe(meal_details)
            
            # Safety assessment
            st.subheader("Safety Assessment:")
            sugar_ok = total_sugars <= 25
            sodium_ok = total_sodium <= 2300
            fiber_ok = total_fiber >= 10
            
            if sugar_ok:
                st.success(f"✓ Sugar intake within limits ({total_sugars:.1f}g ≤ 25g)")
            else:
                st.warning(f"⚠ Sugar intake high ({total_sugars:.1f}g > 25g)")
            
            if sodium_ok:
                st.success(f"✓ Sodium intake within limits ({total_sodium:.0f}mg ≤ 2300mg)")
            else:
                st.warning(f"⚠ Sodium intake high ({total_sodium:.0f}mg > 2300mg)")
            
            if fiber_ok:
                st.success(f"✓ Fiber intake adequate ({total_fiber:.1f}g ≥ 10g)")
            else:
                st.info(f"ℹ Fiber intake could be higher ({total_fiber:.1f}g < 10g)")
        else:
            st.info("Select at least one food to analyze.")

if page == "Weekly Report":
    st.header("Weekly Compliance Report")
    st.write("View compliance metrics for sample users.")
    
    users = data_loader.get_sample_users()
    
    if users:
        selected_user = st.selectbox("Select a user:", users)
        
        if st.button("Generate Report"):
            user_profile = data_loader.get_user_profile(selected_user)
            if user_profile:
                st.subheader(f"Report for {user_profile.get('name', selected_user)}")
                
                # Display user info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**Age:** {user_profile.get('age')}")
                with col2:
                    st.write(f"**Weight:** {user_profile.get('weight_kg')} kg")
                with col3:
                    st.write(f"**Diabetes:** {user_profile.get('has_diabetes')}")
                with col4:
                    st.write(f"**Type:** {user_profile.get('diabetes_type', 'N/A')}")
                
                st.markdown("---")
                
                # Get consumption data
                consumption_log = data_loader.consumption_log_db
                user_consumption = consumption_log[consumption_log['user_id'] == selected_user]
                
                if len(user_consumption) > 0:
                    st.subheader("Weekly Summary:")
                    
                    # Aggregate metrics
                    total_calories = user_consumption['calories'].sum()
                    total_sugars = user_consumption['sugars_g'].sum()
                    total_sodium = user_consumption['sodium_mg'].sum()
                    total_fiber = user_consumption['fiber_g'].sum()
                    meal_count = len(user_consumption)
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total Calories", f"{total_calories:.0f}")
                    with col2:
                        st.metric("Total Sugar", f"{total_sugars:.1f}g")
                    with col3:
                        st.metric("Total Sodium", f"{total_sodium:.0f}mg")
                    with col4:
                        st.metric("Total Fiber", f"{total_fiber:.1f}g")
                    with col5:
                        st.metric("Meals Logged", meal_count)
                    
                    st.markdown("---")
                    
                    # Compliance analysis
                    st.subheader("Compliance Analysis:")
                    
                    sugar_daily_target = 25 * 2  # 2 days of data
                    sodium_daily_target = 2300 * 2
                    
                    sugar_compliance = (1 - min(total_sugars / max(sugar_daily_target, 1), 1)) * 100
                    sodium_compliance = (1 - min(total_sodium / max(sodium_daily_target, 1), 1)) * 100
                    
                    st.write(f"**Sugar Compliance:** {sugar_compliance:.1f}%")
                    st.progress(min(sugar_compliance / 100, 1))
                    
                    st.write(f"**Sodium Compliance:** {sodium_compliance:.1f}%")
                    st.progress(min(sodium_compliance / 100, 1))
                    
                    st.markdown("---")
                    
                    # Recommendations
                    st.subheader("Recommendations:")
                    if total_sugars > sugar_daily_target:
                        st.warning(f"⚠ Sugar intake is {total_sugars - sugar_daily_target:.0f}g above target. Reduce sugary foods.")
                    else:
                        st.success("✓ Sugar intake is within acceptable range.")
                    
                    if total_sodium > sodium_daily_target:
                        st.warning(f"⚠ Sodium intake is {total_sodium - sodium_daily_target:.0f}mg above target. Reduce salt and processed foods.")
                    else:
                        st.success("✓ Sodium intake is within acceptable range.")
                    
                    if total_fiber < 25:
                        st.info(f"ℹ Fiber intake is {25 - total_fiber:.1f}g below recommended. Add more fruits, vegetables, and whole grains.")
                    else:
                        st.success(f"✓ Fiber intake is adequate.")
                    
                    # Meal log
                    st.subheader("Detailed Meal Log:")
                    st.dataframe(user_consumption[['date', 'meal_type', 'food_name', 'calories', 'sugars_g', 'sodium_mg', 'fiber_g']])
                else:
                    st.info(f"No consumption data available for {selected_user}")
    else:
        st.info("No user profiles available.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Project: Food Label Analysis System")
st.sidebar.write("Use `python startup.py` to verify dependencies.")
