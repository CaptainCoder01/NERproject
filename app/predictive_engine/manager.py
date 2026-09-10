import pickle
from pathlib import Path
from .data_cleaner import clean_and_prepare

def predict_disaster_risk(raw_weather_data: dict):
    # 1. Dynamically find the model.pkl file
    current_folder = Path(__file__).parent
    model_path = current_folder / "model.pkl"
    
    try:
        # 2. Clean and prepare the data (Translates strings to ints)
        clean_dataframe = clean_and_prepare(raw_weather_data)
        
        # 3. Load the trained brain
        with model_path.open('rb') as file:
            model = pickle.load(file)
            
        # 4. Ask the model for a prediction
        prediction = model.predict(clean_dataframe)
        
        # 5. Translate the result
        if prediction[0] == 1:
            return "High Risk: Landslide/Flood conditions detected."
        else:
            return "Low Risk: Normal conditions."
            
    except ValueError as validation_error:
        return f"Data Error: {str(validation_error)}"
    except FileNotFoundError:
        return "System Error: model.pkl missing. Run training notebook first."
    except Exception as e:
        return f"Prediction Error: {str(e)}"

# --- Quick Test ---
if __name__ == "__main__":
    test_payload = {
        'rainfall_mm': 180.0,
        'river_water_level': 24.0,
        'soil_moisture_percent': 85.0,
        'slope_angle_degrees': 42.0,
        'wind_speed_kmh': 35.5,
        'vegetation_density_ndvi': 0.2,
        'soil_type': 'clay',
        'soil_texture': 'fine',
        'lithology': 'sedimentary'
    }
    print("Testing with high-risk landslide data:")
    print(f"Result: {predict_disaster_risk(test_payload)}")
