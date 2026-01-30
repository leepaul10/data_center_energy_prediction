# Data Center Energy Prediction

Due to the classified nature of real-world data center energy consumption data, this project uses the ASHRAE Great Energy Predictor Dataset as a realistic proxy. The dataset includes office, education, healthcare, and public-service buildings, which share comparable energy behavior patterns (temporal usage, weather dependency, and load variability).
The focus of this project is on modeling methodology, feature engineering, and predictive performance rather than access to proprietary datasets.The Original dataset used in this project('energy_usage.csv') is too large to include in this repository.
All codes and models provided are full functional.

# This project predicts energy consumption in  centers for next 1 Hour, 24 Hour, and 48 Hour at the present time.
# The Energy Consumption is  Predicted using historical Energy data, weather condtions, and  building information.

# Languages and Machine Learning Packages Used:
python
Pandas
Numpy
Sklearn
Streamlit
Joblib

# Installation:
# 1: Clone the repository
     git clone: (https://github.com/leepaul10/data_center_energy_prediction.git)

# 2: move into folder: 
     cd data_center_energy_prediction
     use vs code and python version 3.10 or 3.11

# 3: Install the Packages:
     refer requirements.txt 
     pip install -r requirements.txt

# 4: Run
     streamlit run app.py

# Features: 
# ::: Predicts Energy Consumption for 1 hour, 24 hour and 48 hour for the exact time
# ::: Handles historical weather and building data 
# ::: Enabled Streamlit for user Interface and inputs for prediction


# About Me :
I'm Lee Paul, the person behind this project,
I've built this project to understand energy patterns in data centers and make energy management easy optimization.

