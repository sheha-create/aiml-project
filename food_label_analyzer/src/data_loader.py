"""
Data Loader Module
Loads datasets from CSV files for the food label analysis system
"""
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class FoodData:
    """Container for food nutrition data"""
    food_id: str
    food_name: str
    category: str
    serving_size_g: float
    calories: float
    carbs_g: float
    sugars_g: float
    fiber_g: float
    protein_g: float
    fat_g: float
    sodium_mg: float
    
    def to_dict(self):
        return {
            'food_id': self.food_id,
            'food_name': self.food_name,
            'category': self.category,
            'serving_size_g': self.serving_size_g,
            'calories': self.calories,
            'carbs_g': self.carbs_g,
            'sugars_g': self.sugars_g,
            'fiber_g': self.fiber_g,
            'protein_g': self.protein_g,
            'fat_g': self.fat_g,
            'sodium_mg': self.sodium_mg
        }


class DataLoader:
    """Load datasets from CSV files"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.nutrition_db: Optional[pd.DataFrame] = None
        self.gi_db: Optional[pd.DataFrame] = None
        self.fraud_ranges_db: Optional[pd.DataFrame] = None
        self.user_profiles_db: Optional[pd.DataFrame] = None
        self.consumption_log_db: Optional[pd.DataFrame] = None
        self._load_all()
    
    def _load_all(self):
        """Load all datasets"""
        try:
            self.load_nutrition_database()
            self.load_gi_database()
            self.load_fraud_detection_ranges()
            self.load_user_profiles()
            self.load_consumption_log()
            print("[DataLoader] All datasets loaded successfully")
        except Exception as e:
            print(f"[DataLoader] Error loading datasets: {e}")
    
    def load_nutrition_database(self) -> pd.DataFrame:
        """Load food nutrition database"""
        if self.nutrition_db is not None:
            return self.nutrition_db
        
        path = DATA_DIR / "food_database" / "nutrition_database.csv"
        if path.exists():
            self.nutrition_db = pd.read_csv(path)
            print(f"[DataLoader] [OK] Loaded {len(self.nutrition_db)} foods from nutrition database")
            return self.nutrition_db
        else:
            print(f"[DataLoader] [WARN] Nutrition database not found at {path}")
            return pd.DataFrame()
    
    def load_gi_database(self) -> pd.DataFrame:
        """Load glycemic index database"""
        if self.gi_db is not None:
            return self.gi_db
        
        path = DATA_DIR / "reference_data" / "glycemic_index_database.csv"
        if path.exists():
            self.gi_db = pd.read_csv(path)
            print(f"[DataLoader] [OK] Loaded {len(self.gi_db)} GI records")
            return self.gi_db
        else:
            print(f"[DataLoader] [WARN] GI database not found at {path}")
            return pd.DataFrame()
    
    def load_fraud_detection_ranges(self) -> pd.DataFrame:
        """Load fraud detection reference ranges"""
        if self.fraud_ranges_db is not None:
            return self.fraud_ranges_db
        
        path = DATA_DIR / "reference_data" / "fraud_detection_ranges.csv"
        if path.exists():
            self.fraud_ranges_db = pd.read_csv(path)
            print(f"[DataLoader] [OK] Loaded {len(self.fraud_ranges_db)} fraud detection categories")
            return self.fraud_ranges_db
        else:
            print(f"[DataLoader] [WARN] Fraud detection ranges not found at {path}")
            return pd.DataFrame()
    
    def load_user_profiles(self) -> pd.DataFrame:
        """Load user profiles"""
        if self.user_profiles_db is not None:
            return self.user_profiles_db
        
        path = DATA_DIR / "food_database" / "user_profiles.csv"
        if path.exists():
            self.user_profiles_db = pd.read_csv(path)
            print(f"[DataLoader] [OK] Loaded {len(self.user_profiles_db)} user profiles")
            return self.user_profiles_db
        else:
            print(f"[DataLoader] [WARN] User profiles not found at {path}")
            return pd.DataFrame()
    
    def load_consumption_log(self) -> pd.DataFrame:
        """Load sample consumption log"""
        if self.consumption_log_db is not None:
            return self.consumption_log_db
        
        path = DATA_DIR / "food_database" / "sample_consumption_log.csv"
        if path.exists():
            self.consumption_log_db = pd.read_csv(path)
            print(f"[DataLoader] [OK] Loaded {len(self.consumption_log_db)} consumption records")
            return self.consumption_log_db
        else:
            print(f"[DataLoader] [WARN] Consumption log not found at {path}")
            return pd.DataFrame()
    
    def get_food_by_id(self, food_id: str) -> Optional[FoodData]:
        """Get a specific food by ID"""
        if self.nutrition_db is None or len(self.nutrition_db) == 0:
            return None
        
        match = self.nutrition_db[self.nutrition_db['food_id'] == food_id]
        if len(match) > 0:
            row = match.iloc[0]
            return FoodData(
                food_id=row['food_id'],
                food_name=row['food_name'],
                category=row['category'],
                serving_size_g=float(row['serving_size_g']),
                calories=float(row['calories']),
                carbs_g=float(row['carbs_g']),
                sugars_g=float(row['sugars_g']),
                fiber_g=float(row['fiber_g']),
                protein_g=float(row['protein_g']),
                fat_g=float(row['fat_g']),
                sodium_mg=float(row['sodium_mg'])
            )
        return None
    
    def search_foods(self, query: str) -> List[FoodData]:
        """Search for foods by name"""
        if self.nutrition_db is None or len(self.nutrition_db) == 0:
            return []
        
        query_lower = query.lower()
        matches = self.nutrition_db[
            self.nutrition_db['food_name'].str.lower().str.contains(query_lower)
        ]
        
        results = []
        for _, row in matches.iterrows():
            results.append(FoodData(
                food_id=row['food_id'],
                food_name=row['food_name'],
                category=row['category'],
                serving_size_g=float(row['serving_size_g']),
                calories=float(row['calories']),
                carbs_g=float(row['carbs_g']),
                sugars_g=float(row['sugars_g']),
                fiber_g=float(row['fiber_g']),
                protein_g=float(row['protein_g']),
                fat_g=float(row['fat_g']),
                sodium_mg=float(row['sodium_mg'])
            ))
        return results
    
    def get_gi_for_food(self, food_name: str) -> Optional[float]:
        """Get glycemic index for a food"""
        if self.gi_db is None or len(self.gi_db) == 0:
            return None
        
        food_lower = food_name.lower()
        match = self.gi_db[self.gi_db['food_name'].str.lower() == food_lower]
        if len(match) > 0:
            return float(match.iloc[0]['glycemic_index'])
        
        # Try partial match
        match = self.gi_db[self.gi_db['food_name'].str.lower().str.contains(food_lower)]
        if len(match) > 0:
            return float(match.iloc[0]['glycemic_index'])
        
        return None
    
    def get_all_foods(self) -> List[FoodData]:
        """Get all foods from database"""
        if self.nutrition_db is None or len(self.nutrition_db) == 0:
            return []
        
        results = []
        for _, row in self.nutrition_db.iterrows():
            results.append(FoodData(
                food_id=row['food_id'],
                food_name=row['food_name'],
                category=row['category'],
                serving_size_g=float(row['serving_size_g']),
                calories=float(row['calories']),
                carbs_g=float(row['carbs_g']),
                sugars_g=float(row['sugars_g']),
                fiber_g=float(row['fiber_g']),
                protein_g=float(row['protein_g']),
                fat_g=float(row['fat_g']),
                sodium_mg=float(row['sodium_mg'])
            ))
        return results
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile by ID"""
        if self.user_profiles_db is None or len(self.user_profiles_db) == 0:
            return None
        
        match = self.user_profiles_db[self.user_profiles_db['user_id'] == user_id]
        if len(match) > 0:
            return match.iloc[0].to_dict()
        return None
    
    def get_sample_users(self) -> List[str]:
        """Get list of sample user IDs"""
        if self.user_profiles_db is None or len(self.user_profiles_db) == 0:
            return []
        return self.user_profiles_db['user_id'].tolist()


# Singleton instance
_loader = None

def get_data_loader() -> DataLoader:
    """Get or create DataLoader singleton"""
    global _loader
    if _loader is None:
        _loader = DataLoader()
    return _loader
