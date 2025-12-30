import numpy as np
import pandas as pd

def generate_traffic_data(days, base_flow, noise_level):
    """
    Generates synthetic traffic data simulating PeMS characteristics.
    Source: Daily and weekly periodic patterns commonly reported in traffic studies.
    """
    hours = days * 24
    time = np.arange(hours)
    # Simulation of daily patterns (Sine waves)
    daily_pattern = np.sin(time * 2 * np.pi / 24) * 0.5 + 0.5 
    rush_hour = np.sin(time * 2 * np.pi / 12) * 0.3
    
    traffic_flow = base_flow + (daily_pattern * base_flow) + (rush_hour * base_flow * 0.5)
    noise = np.random.normal(0, noise_level, hours)
    final_flow = traffic_flow + noise
    
    return np.maximum(final_flow, 0)

def get_research_datasets():
    # Source Domain: Mature sensing environment (High Volume)
    source = generate_traffic_data(days=300, base_flow=500, noise_level=50)
    
    # Target Domain: Newly deployed sensor (Lower Volume, Domain Shift)
    target = generate_traffic_data(days=30, base_flow=150, noise_level=20)
    return source, target
