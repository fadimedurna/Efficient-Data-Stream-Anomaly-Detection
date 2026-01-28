# Efficient Data Stream Anomaly Detection

## Overview
This project demonstrates anomaly detection on a simulated data stream using an unsupervised machine learning approach. Synthetic sequential data is generated to represent regular patterns, seasonal behavior, and random noise, and anomalies are identified using the Isolation Forest algorithm.

The goal of the project is to show how anomalous data points can be detected in sequential data for monitoring and analysis purposes.

---

## Problem Statement
In many systems, data follows recurring patterns with occasional irregular behavior caused by errors, unexpected events, or unusual conditions. Detecting these anomalies is important for understanding data behavior and identifying potential issues.

This project focuses on:
- Simulating a continuous data stream
- Detecting anomalies without labeled data
- Visualizing abnormal data points for interpretation

---

## Approach

### Data Stream Simulation
- Generated synthetic sequential data combining:
  - Periodic patterns
  - Seasonal components
  - Gaussian noise
- Used NumPy to efficiently generate and combine these components

### Anomaly Detection
- Applied the **Isolation Forest** algorithm for unsupervised anomaly detection
- Reshaped the data stream to fit the model requirements
- Used a contamination parameter to control the expected proportion of anomalies
- Identified anomalies based on model predictions

### Visualization
- Plotted the full data stream
- Highlighted detected anomalies using scatter points
- Enabled visual interpretation of abnormal behavior

---

## Tools & Technologies
- Python
- NumPy
- Scikit-learn (Isolation Forest)
- Matplotlib

---

## Use Cases
- Monitoring sequential or time-based data
- Identifying unusual patterns in system metrics
- Detecting abnormal behavior in generated or collected signals
- Exploratory anomaly detection analysis

---

## Outcome
The project demonstrates how unsupervised machine learning techniques such as Isolation Forest can be applied to sequential data to detect anomalies. It provides a clear example of data simulation, anomaly detection, and visualization in a single workflow.

---

## Author
Fadime Durna
