import pandas as pd
from typing import Dict, Any, Tuple

# Mapping dictionaries to convert frontend strings into ML numbers
SOIL_TYPE_MAP = {'sand': 1, 'loam': 2, 'clay': 3}
SOIL_TEXTURE_MAP = {'coarse': 1, 'medium': 2, 'fine': 3}
LITHOLOGY_MAP = {'igneous': 1, 'metamorphic': 2, 'sedimentary': 3}

def validate_weather_input(raw_data: Dict[str, Any]) -> Tuple[bool, str]:
    required_numeric = [
        'rainfall_mm', 'river_water_level', 'soil_moisture_percent', 
        'slope_angle_degrees', 'wind_speed_kmh', 'vegetation_density_ndvi'
    ]
    required_categorical = ['soil_type', 'soil_texture', 'lithology']
    
    for key in required_numeric:
        if key not in raw_data:
            return False, f"Missing required data: '{key}'"
        try:
            float(raw_data[key])
        except (ValueError, TypeError):
            return False, f"Invalid value: '{key}' must be a number."
            
    for key in required_categorical:
        if key not in raw_data:
            return False, f"Missing required data: '{key}'"
            
    if str(raw_data['soil_type']).lower() not in SOIL_TYPE_MAP:
         return False, f"Invalid soil_type. Must be one of: {list(SOIL_TYPE_MAP.keys())}"
    if str(raw_data['soil_texture']).lower() not in SOIL_TEXTURE_MAP:
         return False, f"Invalid soil_texture. Must be one of: {list(SOIL_TEXTURE_MAP.keys())}"
    if str(raw_data['lithology']).lower() not in LITHOLOGY_MAP:
         return False, f"Invalid lithology. Must be one of: {list(LITHOLOGY_MAP.keys())}"
            
    return True, "Valid"

def transform_features(raw_data: Dict[str, Any]) -> pd.DataFrame:
    # Convert absolute river level to relative baseline in the pipeline
    NORMAL_BASELINE_M = 45.0
    relative_river = float(raw_data['river_water_level']) - NORMAL_BASELINE_M
    
    features = {
        'rainfall_mm': [float(raw_data['rainfall_mm'])],
        'relative_river_level_m': [relative_river], # Send the new metric to the brain
        'soil_moisture_percent': [float(raw_data['soil_moisture_percent'])],
        'slope_angle_degrees': [float(raw_data['slope_angle_degrees'])],
        'wind_speed_kmh': [float(raw_data['wind_speed_kmh'])],
        'vegetation_density_ndvi': [float(raw_data['vegetation_density_ndvi'])],
        'soil_type_idx': [SOIL_TYPE_MAP[str(raw_data['soil_type']).lower()]],
        'soil_texture_idx': [SOIL_TEXTURE_MAP[str(raw_data['soil_texture']).lower()]],
        'lithology_idx': [LITHOLOGY_MAP[str(raw_data['lithology']).lower()]]
    }
    return pd.DataFrame(features)

def clean_and_prepare(raw_data: Dict[str, Any]) -> pd.DataFrame:
    is_valid, message = validate_weather_input(raw_data)
    if not is_valid:
        raise ValueError(message)
    return transform_features(raw_data)
