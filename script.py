import random
import time
import numpy as np
from scipy.stats import norm

# Step 1: Suitable Algorithm
# We'll use the Isolation Forest algorithm for anomaly detection.
# It's an unsupervised machine learning algorithm that identifies anomalies by isolating outliers in the data.

from sklearn.ensemble import IsolationForest

# Step 2: Data Stream Simulation
def generate_data_stream(n_samples, pattern_freq, seasonal_freq, noise_level):
    # Generate regular patterns
    pattern = np.sin(np.linspace(0, 2 * np.pi * pattern_freq, n_samples))
    
    # Generate seasonal elements
    seasonal = np.sin(np.linspace(0, 2 * np.pi * seasonal_freq, n_samples))
    
    # Generate random noise
    noise = np.random.normal(0, noise_level, n_samples)
    
    # Combine all components
    data_stream = pattern + seasonal + noise
    
    return data_stream

# Step 3: Anomaly Detection
def detect_anomalies(data_stream, contamination_fraction):
    # Initialize the Isolation Forest model
    model = IsolationForest(contamination=contamination_fraction)
    
    # Reshape the data_stream to fit the model
    data_stream_2d = data_stream.reshape(-1, 1)
    
    # Fit the model to the data_stream
    model.fit(data_stream_2d)
    
    # Predict anomalies
    anomaly_scores = model.decision_function(data_stream_2d)
    anomaly_predictions = model.predict(data_stream_2d)
    
    # Flag anomalies as True (1) and normal points as False (0)
    anomalies = anomaly_predictions == -1
    
    return anomalies, anomaly_scores

# Step 4: Optimization
# The Isolation Forest algorithm is already optimized for speed and efficiency.
# However, we can further optimize the data generation process by using numpy's vectorized operations.

# Step 5: Visualization
def visualize_data_stream(data_stream, anomalies):
    import matplotlib.pyplot as plt
    
    # Create a figure and axes
    fig, ax = plt.subplots()
    
    # Plot the data stream
    ax.plot(data_stream, label='Data Stream')
    
    # Plot the anomalies
    anomaly_indices = np.where(anomalies)[0]
    ax.scatter(anomaly_indices, data_stream[anomaly_indices], color='red', label='Anomalies')
    
    # Add labels and legend
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    
    # Show the plot
    plt.show()

# Example usage
n_samples = 1000
pattern_freq = 0.01
seasonal_freq = 0.05
noise_level = 0.1
contamination_fraction = 0.01

# Generate the data stream
data_stream = generate_data_stream(n_samples, pattern_freq, seasonal_freq, noise_level)

# Detect anomalies
anomalies, anomaly_scores = detect_anomalies(data_stream, contamination_fraction)

# Visualize the data stream and anomalies
visualize_data_stream(data_stream, anomalies)