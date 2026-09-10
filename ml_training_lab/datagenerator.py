import random
import numpy as np
import pandas as pd

def generate_ner_dataset(num_samples: int = 1000):
    np.random.seed(42)
    random.seed(42)
    
    NORMAL_BASELINE_M = 45.0
    data = []
    
    for _ in range(num_samples):
        terrain_type = random.choice(['plains', 'hills'])
        
        if terrain_type == 'plains':
            slope = round(random.uniform(0.0, 10.0), 1)
            soil_type = random.choice([2, 3])  # Loam or Clay
            lithology = 3  # Sedimentary
            ndvi = round(random.uniform(0.3, 0.7), 2)
            
            # Flood risk conditions
            if random.random() > 0.15:
                rain = round(random.uniform(0.0, 40.0), 1)
                river = round(random.uniform(43.0, 47.0), 1)
                disaster = 0
            else:
                rain = round(random.uniform(100.0, 260.0), 1)
                river = round(random.uniform(49.0, 52.0), 1)
                disaster = 1
                
        else:  # Hills
            slope = round(random.uniform(25.0, 60.0), 1)
            soil_type = random.choice([1, 2])  # Sand or Loam
            lithology = random.choice([1, 2])  # Igneous or Metamorphic
            river = round(random.uniform(40.0, 45.0), 1)
            
            # Landslide risk conditions
            if random.random() > 0.15:
                rain = round(random.uniform(0.0, 50.0), 1)
                ndvi = round(random.uniform(0.4, 0.8), 2)
                disaster = 0
            else:
                rain = round(random.uniform(120.0, 200.0), 1)
                ndvi = round(random.uniform(0.1, 0.3), 2)
                disaster = 1

        # Environmental correlations
        moisture = min(100.0, round(rain * 0.4 + random.uniform(10.0, 30.0), 1))
        wind = round(random.uniform(5.0, 45.0), 1)
        texture = random.choice([1, 2, 3])
        relative_river = round(river - NORMAL_BASELINE_M, 2)
        
        data.append([
            rain, river, relative_river, moisture, slope, 
            wind, ndvi, soil_type, texture, lithology, disaster
        ])
        
    df = pd.DataFrame(data, columns=[
        'rainfall_mm',
        'river_water_level',
        'relative_river_level_m',
        'soil_moisture_percent',
        'slope_angle_degrees',
        'wind_speed_kmh',
        'vegetation_density_ndvi',
        'soil_type_idx',
        'soil_texture_idx',
        'lithology_idx',
        'disaster_occurred'
    ])
    
    df.to_csv('dataset.csv', index=False)
    print(f"Success! dataset.csv created with {num_samples} rows based on NER parameters.")

if __name__ == "__main__":
    generate_ner_dataset(1000)
