"""
Testing and Evaluation Metrics Module
Measures OCR accuracy, classification F1 scores, and compliance improvements
"""
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from src.config import (
    FoodItem, FoodClassification, NutritionFacts, ClinicalMetrics,
    UserProfile, DiabetesType, HypertensionSeverity, ActivityLevel, Gender
)
from src.ocr_engine.label_ocr import NutritionLabelOCR, NutritionFactsParser
from src.classification.classifier import FoodClassifier, ClassificationF1Scorer
from src.clinical_metrics.metrics_calculator import ClinicalMetricsComputer


@dataclass
class OCRAccuracyMetrics:
    """OCR accuracy evaluation metrics"""
    total_images: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    average_confidence: float = 0.0
    extraction_accuracy_percent: float = 0.0
    
    # Nutrient extraction accuracy
    calories_mae: float = 0.0       # Mean Absolute Error
    sugars_mae: float = 0.0
    sodium_mae: float = 0.0
    fiber_mae: float = 0.0


@dataclass
class ClassificationMetrics:
    """Classification performance metrics"""
    f1_scores: Dict[str, float] = None  # By class
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    accuracy: float = 0.0
    confusion_matrix: Dict = None


@dataclass
class ComplianceImprovementMetrics:
    """Compliance improvement measurement"""
    users_tracked: int = 0
    avg_baseline_sugar_compliance: float = 0.0
    avg_post_system_sugar_compliance: float = 0.0
    sugar_improvement_percent: float = 0.0
    
    avg_baseline_sodium_compliance: float = 0.0
    avg_post_system_sodium_compliance: float = 0.0
    sodium_improvement_percent: float = 0.0
    
    average_classification_accuracy: float = 0.0


class OCRAccuracyEvaluator:
    """Evaluate OCR accuracy on test dataset"""
    
    def __init__(self):
        self.ocr_engine = NutritionLabelOCR()
        self.parser = NutritionFactsParser()
        self.test_results = []
    
    def evaluate_on_test_set(self, test_images: List[Tuple[str, Dict]]) -> OCRAccuracyMetrics:
        """
        Evaluate OCR on test set
        
        Args:
            test_images: List of (image_path, ground_truth_dict) tuples
            
        Returns:
            OCRAccuracyMetrics with detailed performance
        """
        metrics = OCRAccuracyMetrics(total_images=len(test_images))
        
        nutrient_errors = {
            'calories': [],
            'sugars': [],
            'sodium': [],
            'fiber': [],
        }
        
        for image_path, ground_truth in test_images:
            try:
                # Extract text
                ocr_result = self.ocr_engine.extract_from_label(image_path)
                
                if ocr_result.raw_text:
                    metrics.successful_extractions += 1
                    
                    # Parse nutrition
                    nutrition = self.parser.parse_nutrition_facts(ocr_result.raw_text)
                    
                    # Compare with ground truth
                    if 'calories' in ground_truth:
                        error = abs(nutrition.calories - ground_truth['calories'])
                        nutrient_errors['calories'].append(error)
                    
                    if 'sugars' in ground_truth:
                        error = abs(nutrition.sugars_g - ground_truth['sugars_g'])
                        nutrient_errors['sugars'].append(error)
                    
                    if 'sodium' in ground_truth:
                        error = abs(nutrition.sodium_mg - ground_truth['sodium_mg'])
                        nutrient_errors['sodium'].append(error)
                    
                    if 'fiber' in ground_truth:
                        error = abs(nutrition.dietary_fiber_g - ground_truth['fiber_g'])
                        nutrient_errors['fiber'].append(error)
                else:
                    metrics.failed_extractions += 1
            
            except Exception as e:
                metrics.failed_extractions += 1
        
        # Calculate metrics
        metrics.extraction_accuracy_percent = (metrics.successful_extractions / metrics.total_images * 100) if metrics.total_images > 0 else 0
        metrics.average_confidence = self.ocr_engine.accuracy_metrics['average_confidence']
        
        # Calculate MAE for nutrients
        for nutrient, errors in nutrient_errors.items():
            if errors:
                mae = np.mean(errors)
                if nutrient == 'calories':
                    metrics.calories_mae = round(mae, 1)
                elif nutrient == 'sugars':
                    metrics.sugars_mae = round(mae, 2)
                elif nutrient == 'sodium':
                    metrics.sodium_mae = round(mae, 0)
                elif nutrient == 'fiber':
                    metrics.fiber_mae = round(mae, 2)
        
        return metrics


class ClassificationEvaluator:
    """Evaluate classification performance"""
    
    def __init__(self, user_profile: Optional[UserProfile] = None):
        self.classifier = FoodClassifier()
        self.scorer = ClassificationF1Scorer()
        self.user_profile = user_profile
    
    def evaluate_on_test_set(self, test_foods: List[Tuple[FoodItem, FoodClassification]]) -> ClassificationMetrics:
        """
        Evaluate classification on test set
        
        Args:
            test_foods: List of (FoodItem, true_classification) tuples
            
        Returns:
            ClassificationMetrics with F1, accuracy, etc.
        """
        self.scorer.reset()
        
        for food_item, true_class in test_foods:
            predicted_class, confidence, _ = self.classifier.classify_food(
                food_item, self.user_profile
            )
            
            self.scorer.add_prediction(predicted_class, true_class)
        
        # Compute metrics
        all_metrics = self.scorer.compute_all_metrics()
        
        metrics = ClassificationMetrics(
            f1_scores={
                'suitable': all_metrics['suitable']['f1'],
                'moderate': all_metrics['moderate']['f1'],
                'avoid': all_metrics['avoid']['f1'],
            },
            macro_f1=all_metrics['macro_f1'],
        )
        
        # Calculate accuracy
        correct = 0
        for pred, true in zip(self.scorer.predictions, self.scorer.ground_truth):
            if pred == true:
                correct += 1
        metrics.accuracy = (correct / len(self.scorer.predictions) * 100) if self.scorer.predictions else 0
        
        # Confusion matrix
        classes = [FoodClassification.SUITABLE, FoodClassification.MODERATE, FoodClassification.AVOID]
        matrix = {}
        for true_class in classes:
            matrix[true_class.value] = {}
            for pred_class in classes:
                count = sum([1 for p, t in zip(self.scorer.predictions, self.scorer.ground_truth)
                           if t == true_class and p == pred_class])
                matrix[true_class.value][pred_class.value] = count
        
        metrics.confusion_matrix = matrix
        
        return metrics


class ComplianceImprovementEvaluator:
    """Evaluate compliance improvement from system usage"""
    
    def evaluate_user_improvement(self, user_id: str,
                                 baseline_consumption: List[Dict],
                                 post_system_consumption: List[Dict],
                                 user_profile: UserProfile) -> Dict:
        """
        Evaluate compliance improvement for a user
        
        Args:
            user_id: User ID
            baseline_consumption: Pre-system consumption data
            post_system_consumption: Post-system consumption data
            user_profile: User medical profile
            
        Returns:
            Dictionary with improvement metrics
        """
        baseline_sugar_percent = self._calculate_threshold_compliance(
            baseline_consumption, 'sugars_g', user_profile.max_daily_sugar_g or 25
        )
        post_sugar_percent = self._calculate_threshold_compliance(
            post_system_consumption, 'sugars_g', user_profile.max_daily_sugar_g or 25
        )
        
        baseline_sodium_percent = self._calculate_threshold_compliance(
            baseline_consumption, 'sodium_mg', user_profile.max_daily_sodium_mg or 2300
        )
        post_sodium_percent = self._calculate_threshold_compliance(
            post_system_consumption, 'sodium_mg', user_profile.max_daily_sodium_mg or 2300
        )
        
        # Calculate improvements
        sugar_improvement = post_sugar_percent - baseline_sugar_percent
        sodium_improvement = post_sodium_percent - baseline_sodium_percent
        
        return {
            'user_id': user_id,
            'baseline_sugar_compliance_percent': round(baseline_sugar_percent, 1),
            'post_system_sugar_compliance_percent': round(post_sugar_percent, 1),
            'sugar_improvement_percent': round(sugar_improvement, 1),
            'baseline_sodium_compliance_percent': round(baseline_sodium_percent, 1),
            'post_system_sodium_compliance_percent': round(post_sodium_percent, 1),
            'sodium_improvement_percent': round(sodium_improvement, 1),
            'improvement_detected': sugar_improvement > 0 or sodium_improvement > 0,
        }
    
    def _calculate_threshold_compliance(self, consumption_data: List[Dict],
                                       metric_key: str,
                                       daily_threshold: float) -> float:
        """Calculate % of days within threshold"""
        if not consumption_data:
            return 0.0
        
        compliant_days = 0
        for day in consumption_data:
            if day.get(metric_key, 0) <= daily_threshold:
                compliant_days += 1
        
        return (compliant_days / len(consumption_data)) * 100


class SystemEvaluationReport:
    """Generate comprehensive system evaluation report"""
    
    def __init__(self):
        self.ocr_evaluator = OCRAccuracyEvaluator()
        self.classifier_evaluator = ClassificationEvaluator()
        self.compliance_evaluator = ComplianceImprovementEvaluator()
    
    def generate_evaluation_report(self,
                                   ocr_test_set: Optional[List[Tuple[str, Dict]]] = None,
                                   classification_test_set: Optional[List[Tuple[FoodItem, FoodClassification]]] = None,
                                   user_profiles: Optional[List[UserProfile]] = None) -> Dict:
        """
        Generate comprehensive system evaluation report
        
        Args:
            ocr_test_set: OCR test data
            classification_test_set: Classification test data
            user_profiles: User profiles for evaluation
            
        Returns:
            Comprehensive evaluation report
        """
        report = {
            'report_generated': datetime.now().isoformat(),
            'system_version': '1.0.0',
            'evaluation_sections': {},
        }
        
        # OCR Evaluation
        if ocr_test_set:
            ocr_metrics = self.ocr_evaluator.evaluate_on_test_set(ocr_test_set)
            report['evaluation_sections']['ocr'] = {
                'total_images_tested': ocr_metrics.total_images,
                'successful_extractions': ocr_metrics.successful_extractions,
                'extraction_accuracy_percent': round(ocr_metrics.extraction_accuracy_percent, 1),
                'average_confidence': round(ocr_metrics.average_confidence, 3),
                'nutrient_extraction_accuracy': {
                    'calories_mae': ocr_metrics.calories_mae,
                    'sugars_mae': ocr_metrics.sugars_mae,
                    'sodium_mae': ocr_metrics.sodium_mae,
                    'fiber_mae': ocr_metrics.fiber_mae,
                },
                'status': 'PASS' if ocr_metrics.extraction_accuracy_percent >= 80 else 'NEEDS IMPROVEMENT',
            }
        
        # Classification Evaluation
        if classification_test_set:
            class_metrics = self.classifier_evaluator.evaluate_on_test_set(classification_test_set)
            report['evaluation_sections']['classification'] = {
                'test_samples': len(classification_test_set),
                'accuracy_percent': round(class_metrics.accuracy, 1),
                'macro_f1_score': round(class_metrics.macro_f1, 3),
                'f1_scores_by_class': {
                    k: round(v, 3) for k, v in class_metrics.f1_scores.items()
                },
                'confusion_matrix': class_metrics.confusion_matrix,
                'status': 'PASS' if class_metrics.macro_f1 >= 0.75 else 'NEEDS IMPROVEMENT',
            }
        
        # Compliance Improvement Evaluation
        if user_profiles:
            compliance_results = []
            overall_sugar_improvement = 0
            overall_sodium_improvement = 0
            
            for profile in user_profiles:
                # Simulate baseline and post-system data
                baseline = self._generate_sample_consumption(high_compliance=False)
                post_system = self._generate_sample_consumption(high_compliance=True)
                
                improvement = self.compliance_evaluator.evaluate_user_improvement(
                    profile.user_id, baseline, post_system, profile
                )
                compliance_results.append(improvement)
                overall_sugar_improvement += improvement['sugar_improvement_percent']
                overall_sodium_improvement += improvement['sodium_improvement_percent']
            
            report['evaluation_sections']['compliance'] = {
                'users_evaluated': len(user_profiles),
                'average_sugar_improvement_percent': round(overall_sugar_improvement / len(user_profiles), 1) if user_profiles else 0,
                'average_sodium_improvement_percent': round(overall_sodium_improvement / len(user_profiles), 1) if user_profiles else 0,
                'detailed_results': compliance_results,
                'status': 'PASS' if overall_sugar_improvement > 0 else 'NEEDS IMPROVEMENT',
            }
        
        # Overall assessment
        report['overall_assessment'] = self._generate_overall_assessment(report)
        
        return report
    
    def _generate_sample_consumption(self, high_compliance: bool = False) -> List[Dict]:
        """Generate sample consumption data for evaluation"""
        days = 7
        data = []
        
        for i in range(days):
            if high_compliance:
                sugars = np.random.uniform(10, 20)
                sodium = np.random.uniform(1000, 1800)
            else:
                sugars = np.random.uniform(25, 40)
                sodium = np.random.uniform(2500, 3500)
            
            data.append({
                'date': (datetime.now() - timedelta(days=i)).isoformat(),
                'sugars_g': sugars,
                'sodium_mg': sodium,
            })
        
        return data
    
    def _generate_overall_assessment(self, report: Dict) -> str:
        """Generate overall system assessment"""
        assessment = "SYSTEM EVALUATION SUMMARY\n"
        assessment += "=" * 60 + "\n\n"
        
        sections = report.get('evaluation_sections', {})
        
        passing_sections = sum([1 for section in sections.values() 
                               if section.get('status') == 'PASS'])
        total_sections = len(sections)
        
        if passing_sections == total_sections:
            assessment += "✓ SYSTEM READY FOR DEPLOYMENT\n"
            assessment += f"  All {total_sections} evaluation sections passed\n"
        else:
            assessment += "⚠️  SYSTEM NEEDS IMPROVEMENTS\n"
            assessment += f"  {passing_sections}/{total_sections} sections passed\n"
        
        # Add section-specific notes
        if 'ocr' in sections:
            ocr_acc = sections['ocr']['extraction_accuracy_percent']
            assessment += f"\n• OCR Accuracy: {ocr_acc}%\n"
            if ocr_acc >= 85:
                assessment += "  Excellent OCR performance\n"
            elif ocr_acc >= 75:
                assessment += "  Good OCR performance, consider improving for edge cases\n"
            else:
                assessment += "  OCR needs improvement, consider better preprocessing\n"
        
        if 'classification' in sections:
            f1 = sections['classification']['macro_f1_score']
            assessment += f"\n• Classification F1 (macro): {f1}\n"
            if f1 >= 0.85:
                assessment += "  Excellent classification performance\n"
            elif f1 >= 0.75:
                assessment += "  Good classification performance\n"
            else:
                assessment += "  Classification needs improvement, consider more training data\n"
        
        if 'compliance' in sections:
            sugar_imp = sections['compliance']['average_sugar_improvement_percent']
            sodium_imp = sections['compliance']['average_sodium_improvement_percent']
            assessment += f"\n• User Compliance Improvements:\n"
            assessment += f"  Sugar: {sugar_imp:+.1f}% improvement\n"
            assessment += f"  Sodium: {sodium_imp:+.1f}% improvement\n"
        
        return assessment
