#!/usr/bin/env python
# coding: utf-8

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("fivethirtyeight")
import datetime
# Add these imports at the top
import time
import uuid
from functools import wraps
from typing import Any, Dict

# API
import requests
import tensorflow as tf
from bs4 import BeautifulSoup
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
# ML
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# Configure logging
logging.basicConfig(filename='output.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress TensorFlow warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import tensorflow as tf

tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

# Optional: Suppress CUDA warnings
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# ## Data Gathering and Evaluation
# I found 5 sources for daily weather data in New York City as follows:
# 1. National Centers for Environmental Information (NCEI)
# 2. National Weather Service
# 3. Visual Crossing
# 4. Meteomatics
# 5. Yahoo Weather

# ### National Weather Service

# I was initially thrilled about the this data source, as it closely aligns with what's behind Kalshi's weather event trading. However, upon delving into their API documentation, my excitement was met with a degree of disappointment. It became evident that while they only offer forecast weather data through their API, and the data available for direct download is confined to a-month record. This limitation prompted me to explore alternative data sources that better suit the needs.

# In[ ]:


# Define the URL for NYC data
url = "https://forecast.weather.gov/MapClick.php?lat=40.714530000000025&lon=-74.00711999999999"

try:
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")
        observed_data_divs = soup.find_all("div", class_="tombstone-container")

        timestamps = []
        temperatures = []

        # Extract data from API
        for observed_div in observed_data_divs:
            timestamp = observed_div.find("p", class_="period-name").get_text(strip=True)
            temperature_element = observed_div.find("p", class_="temp")
            if temperature_element:
                temperature = temperature_element.get_text(strip=True)
                timestamps.append(timestamp)
                temperatures.append(temperature)
            else:
                logging.warning(f"Temperature element not found for timestamp: {timestamp}")

        # Store data
        nws_data = pd.DataFrame({
            "Timestamp": timestamps,
            "Temperature (Fahrenheit)": temperatures
        })
        logging.info("API request successful")
    else:
        logging.error(f"API request failed with status code {response.status_code}")
except requests.exceptions.RequestException as e:
    logging.error(f"Error occurred during API request: {str(e)}")


# ### National Centers for Environmental Information (NCEI)

# In[405]:


# NCEI API Constants
NCEI_API_KEY = "hQjOAltlsPnryPJlIEkjkzQqJFPtGOpe"
NCEI_BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
NCEI_STATION_ID = "GHCND:USW00094728"  # Central Park Station

def get_ncei_data(start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    api_data = pd.DataFrame()
    
    while start_date <= end_date:
        # NCEI limits to 1000 records per request
        period_end = min(start_date + datetime.timedelta(days=100), end_date)
        
        params = {
            "datasetid": "GHCND",
            "stationid": NCEI_STATION_ID,
            "startdate": start_date.strftime("%Y-%m-%d"),
            "enddate": period_end.strftime("%Y-%m-%d"),
            "units": "standard",
            "datatypeid": "AWND,PRCP,SNOW,SNWD,TMAX,TMIN,WT01,WT03,WT08",
            "limit": 1000
        }
        
        headers = {
            "token": NCEI_API_KEY
        }
        
        try:
            response = requests.get(NCEI_BASE_URL, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if "results" in data:
                    json_data = pd.DataFrame(data["results"])
                    api_data = pd.concat([api_data, json_data], ignore_index=True)
                else:
                    logging.warning(f"No results found for period {start_date} to {period_end}")
            else:
                logging.error(f"API request failed with status code {response.status_code}")
                if response.status_code == 429:  # Too many requests
                    logging.warning("Rate limit exceeded, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                else:
                    logging.error(f"API request failed with status code {response.status_code}: {response.text}")
                    break  # Break the loop if a non-429 error occurs
            
            # Rate limiting - NCEI recommends no more than 5 requests per second
            time.sleep(0.2)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed for period {start_date} to {period_end}: {str(e)}")
            continue
        
        start_date = period_end + datetime.timedelta(days=1)
    
    if api_data.empty:
        logging.error("No data was retrieved from NCEI API")
    
    return api_data

# Main NCEI data retrieval and processing
try:
    start_date = datetime.datetime.strptime("2019-08-01", "%Y-%m-%d").date()
    last_date = datetime.datetime.strptime("2023-08-01", "%Y-%m-%d").date()
    
    # Get data from NCEI API
    ncei_data = get_ncei_data(start_date, last_date)
    
    if not ncei_data.empty:
        # Process the data using ncei_processing function
        ncei_df = ncei_processing(ncei_data)
        
        if not ncei_df.empty:
            logging.info("Successfully processed NCEI data")
            
            # Set index and create visualization
            ncei_df.set_index('date', inplace=True)
            
            # Visualize the data
            # ... (visualization code remains the same)
        else:
            logging.error("Failed to process NCEI data")
            ncei_df = pd.DataFrame()  # Set ncei_df to an empty DataFrame if processing fails
    else:
        logging.error("Failed to retrieve NCEI data")
        ncei_df = pd.DataFrame()  # Set ncei_df to an empty DataFrame if retrieval fails
        
except Exception as e:
    logging.error(f"Error processing NCEI data: {str(e)}")
    ncei_df = pd.DataFrame()


# ### Visual Crossing

# I attempted to pull the data from Visual Crossing's API, but unfortunately there're limits on the amount of records per request and on the number of calls every day. I ended up having to download csv data from their website.

# In[406]:


# Define Visual Crossing API constants
VC_API_KEY = "WS8SPUNTV45687SM8PE4SP8EC"  # Your API key
VC_BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

def get_visual_crossing_data():
    """Get weather data from Visual Crossing API"""
    try:
        # Format dates and location
        location = "New York City,USA"
        start_date = "2020-11-01"
        end_date = "2021-11-01"
        
        # Construct URL and parameters
        url = f"{VC_BASE_URL}/{urllib.parse.quote(location)}/{start_date}/{end_date}"
        params = {
            "unitGroup": "us",
            "include": "days",  # Changed from "day" to "days" per API docs
            "key": VC_API_KEY,
            "contentType": "json"  # Changed to JSON for easier processing
        }

        # Make API request
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Process JSON response
        data = response.json()
        if 'days' not in data:
            raise ValueError("No daily data found in API response")
            
        # Convert to DataFrame
        df = pd.DataFrame(data['days'])
        
        # Select relevant columns
        df = df[['datetime', 'tempmax', 'dew', 'humidity', 'precip', 'windgust',
                'windspeed', 'sealevelpressure', 'solarradiation', 'solarenergy']]
        
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        logging.info("Successfully retrieved and processed Visual Crossing data")
        return df
        
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {str(e)}")
        return None
    except (ValueError, KeyError) as e:
        logging.error(f"Error processing API response: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return None

# Get data from API or fallback to CSV
try:
    vs_df = get_visual_crossing_data()
    
    # If API fails, try reading from CSV
    if vs_df is None:
        logging.info("Falling back to CSV file")
        try:
            vs_df = pd.read_csv("visual_crossing.csv")
            if vs_df.empty:
                logging.error("Visual Crossing CSV file is empty")
                vs_df = None
            else:
                vs_df['datetime'] = pd.to_datetime(vs_df['datetime'])
                vs_df = vs_df[['datetime', 'tempmax', 'dew', 'humidity', 'precip', 'windgust',
                              'windspeed', 'sealevelpressure', 'solarradiation', 'solarenergy']]
        except FileNotFoundError:
            logging.error("Visual Crossing CSV file not found")
            vs_df = None
        except Exception as e:
            logging.error(f"Error reading Visual Crossing CSV: {str(e)}")
            vs_df = None
    
    # Process the DataFrame
    if vs_df is not None:
        # Drop NaN values
        vs_df = vs_df.dropna()
        logging.debug(f"Visual Crossing DataFrame after dropping NaNs: {vs_df.head()}")
        
        # Set 'datetime' as the index
        vs_df.set_index('datetime', inplace=True)
        
        # Create the time series plot
        plt.figure(figsize=(14, 6))
        plt.plot(vs_df.index, vs_df['tempmax'], marker='o', linestyle='-', color='b')
        plt.title('New York Max Temperature (2022 - 2023)')
        plt.xlabel('Date')
        plt.ylabel('Temperature (°F)')
        plt.show()
    else:
        logging.error("Failed to obtain weather data from both API and CSV")
        
except Exception as e:
    logging.error(f"Error processing weather data: {str(e)}")
    vs_df = None


# ### Yahoo Weather

# Yahoo Weather does not allow me to scrape historical data. They mostly have just forecast data.

# In[153]:


try:
    yahoo = pd.read_html("https://www.yahoo.com/news/weather/united-states/new-york/new-york-2459115")
    if yahoo:  # Check if any tables were found
        print("Yahoo Weather Data:")
        print(yahoo)
    else:
        logging.warning("No weather data tables found on Yahoo Weather page")
except Exception as e:
    logging.error(f"Error fetching Yahoo Weather data: {str(e)}")
    yahoo = None


# Gathering relevant weather data to predict the daily maximum temperature in New York was a challenging endeavor. Identifying credible and dependable data sources was not easy. Once I had pinpointed these sources, I encountered further obstacles in scraping or downloading the data. The insufficiency of some API documentation added a layer of complexity, as understanding how to extract the right data often demanded a significant investment of time and effort.
# 
# I faced numerous hurdles, such as locating station ID numbers, circumventing the limitations on API calls, and, in some cases, having to consider the costs associated with certain APIs. While it may seem that there is a plethora of weather data available, it became clear that only a handful of sources provided data that was truly usable for my prediction task. Some sources offered solely monthly or historical data with a limited time frame, while others provided only forecast data. 
# 
# After a thorough assessment of these challenges and limitations, I decided to rely on data from source NCEI as my foundation for processing and predicting daily maximum temperatures because it is the most consistent dataset. The only downside I have noticed is that the data does not update daily as expected, especially when we approach weekends or holidays.

# ## Data Processing

# ### NCEI Data

# In[408]:


def ncei_processing(df):
    """Process NCEI weather data into a clean DataFrame"""
    try:
        # Unpivot relevant columns
        processed_df = df.pivot(index='date', columns=['datatype'], values='value')
        processed_df = processed_df.reset_index()

        # Rename columns to be descriptive
        processed_df = processed_df.rename(columns={
            'AWND': 'avg wind speed',
            'PRCP': 'precipitation',
            'SNOW': 'snowfall',
            'SNWD': 'snow depth',
            'TMAX': 'max temp',
            'TMIN': 'min temp',
            'WT01': 'fog',
            'WT03': 'thunder',
            'WT08': 'smoke/haze'
        })

        # Process NaN values
        processed_df = processed_df.dropna(subset=[
            'avg wind speed',
            'precipitation',
            'snowfall',
            'snow depth',
            'max temp',
            'min temp'
        ])
        processed_df = processed_df.fillna(0)

        # Extract date elements
        processed_df['date'] = pd.to_datetime(processed_df['date'])
        processed_df['year'] = processed_df['date'].dt.year
        processed_df['month'] = processed_df['date'].dt.month
        processed_df['day'] = processed_df['date'].dt.day
        processed_df['quarter'] = processed_df['date'].dt.quarter
        
        logging.debug(f"Processed NCEI DataFrame: {processed_df.head()}")
        return processed_df
        
    except Exception as e:
        logging.error(f"Error in ncei_processing: {str(e)}")
        return pd.DataFrame()

ncei_df = ncei_processing(api)


# In[409]:


# Set 'Date' as the index
ncei_df.set_index('date', inplace=True)
logging.debug(f"NCEI DataFrame with 'date' as index: {ncei_df.head()}")
# # Visualize the data
plt.figure(figsize=(14, 5))
plt.plot(ncei_df.index, ncei_df['max temp'], marker='o', linestyle='-', color='b')
plt.title('New York Max Temperature (2019 - 2023)')
plt.xlabel('Date')
plt.ylabel('Temperature (°F)')

plt.figure(figsize=(14, 5))
plt.plot(ncei_df.index, ncei_df['precipitation'], marker='o', linestyle='-', color='b')
plt.title('New York Daily Precipitation (2019 - 2023)')
plt.xlabel('Date')
plt.ylabel('Temperature (°F)')


# ### Visual Crossing Data

# In[410]:


# Drop NaN values
vs_df = vs_df.dropna()
logging.debug(f"Visual Crossing DataFrame after dropping NaNs: {vs_df.head()}")
# Set 'Date' as the index (required for time series plotting)
vs_df.set_index('datetime', inplace=True)
# Create the time series plot
plt.figure(figsize=(14, 6))
plt.plot(vs_df.index, vs_df['tempmax'], marker='o', linestyle='-', color='b')
plt.title('New York Max Temperature (2022 - 2023)')
plt.xlabel('Date')
plt.ylabel('Temperature (°F)')


# ## Model Training and Selection
# ### Linear Regression

# #### Visual Crossing Data

# In[411]:


# Shift the max temp by one row to predict next day's max temperature
vs_df_shift = vs_df.copy()
vs_df_shift['tempmax_next'] = vs_df_shift['tempmax'].shift(-1)
logging.debug(f"Visual Crossing DataFrame after shifting 'tempmax': {vs_df_shift.head()}")

# Remove the first and last row because of the shift
vs_df_shift = vs_df_shift.iloc[1:-1]

# Define features (x) and the target variable (y)
x = vs_df_shift[['tempmax', 'dew', 'humidity', 'precip', 'windgust', 'windspeed', 'sealevelpressure', 'solarradiation', 'solarenergy']]
y = vs_df_shift['tempmax_next']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Grid search for optimal parameters for a ridge linear regression model
param_grid = {'alpha': [0.1, 1, 10]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
grid_search.fit(x_train, y_train)
best_alpha = grid_search.best_params_['alpha']

# Fit the model
best_model = Ridge(alpha=best_alpha)
best_model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = best_model.predict(x_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# Visualize the actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Max Temperature (°F)')
plt.ylabel('Predicted Max Temperature (°F)')
plt.title('Actual vs. Predicted Max Temperature')
plt.show()


# #### NCEI Data

# In[412]:


# Shift the max temp by one row to predict next day's max temperature
ncei_df_shift = ncei_df.copy()
ncei_df_shift['max temp next'] = ncei_df_shift['max temp'].shift(-1)
logging.debug(f"NCEI DataFrame after shifting 'max temp': {ncei_df_shift.head()}")

# Remove the first and last row because of the shift
ncei_df_shift = ncei_df_shift.iloc[1:-1]

# Define features (x) and the target variable (y)
x = ncei_df_shift[['avg wind speed','precipitation','snowfall','snow depth','min temp','max temp','fog','thunder','smoke/haze']]
y = ncei_df_shift['max temp next']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Grid search for optimal parameters for a ridge linear regression model
param_grid = {'alpha': [0.1, 1, 10]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
grid_search.fit(x_train, y_train)
best_alpha = grid_search.best_params_['alpha']
print(f'The best alpha is {best_alpha}')

# Fit the model
best_model = Ridge(alpha=best_alpha)
best_model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = best_model.predict(x_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# Visualize the actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Max Temperature (°F)')
plt.ylabel('Predicted Max Temperature (°F)')
plt.title('Actual vs. Predicted Max Temperature')
plt.show()


# Building a robust linear regression model was a meticulous process that involved several critical steps. To ensure the model's optimal performance, I employed a grid search to fine-tune its parameters. This systematic search exhaustively explored all possible parameter combinations, allowing me to pinpoint the regularization term that delivered the most accurate results. Through the grid search, I found out that the optimal alpha was 10. 

# ### Long Short Term Memory (LSTM)

# From the above visualizations of the daily temperature, I realized that there is seasonality factor, and thought it would be appropriate to try implementing a Long Short Term Memory (LSTM) modelfor the following reasons:
# 1. Weather data is inherently sequential, where past conditions can significantly impact future conditions. LSTM can handle sequential data and capture long-range dependencies. They can effectively model the temporal relationships in weather data..
# 
# 2. LSTM can automatically learn relevant features from the data, reducing the need for manual feature engineering. They can extract complex patterns and relationships within the data, such as the impact of multiple weather variables on future conditions.
# 
# 3. Weather forecasting often involves multiple variables (e.g., temperature, precipitation, wind speed) that interact with each other. LSTM can handle multivariate time series data and model complex, nonlinear relationships in weather data.

# ### Multivariate LSTM

# In[418]:


# Extract the feature and target data
features = ncei_df[['avg wind speed','precipitation','snowfall','snow depth','min temp','max temp']].values.astype(float)
target = ncei_df['max temp'].values.astype(float)
logging.debug(f"Features shape: {features.shape}, Target shape: {target.shape}")

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)
scaled_target = scaler.fit_transform(target.reshape(-1, 1))

# Define the number of time steps and features
n_steps = 30
n_features = features.shape[1]

# Create sequences for training
x, y = [], []
for i in range(len(ncei_df) - n_steps):
    x.append(scaled_features[i:i + n_steps, :])
    y.append(scaled_target[i + n_steps])

x, y = np.array(x), np.array(y)

# Split the data into training, validation and testing sets
train_size = int(0.8 * len(ncei_df))
val_size = int(0.9 * len(ncei_df))
x_train, x_val, x_test = x[:train_size], x[train_size:val_size], x[val_size:]
y_train, y_val, y_test =  y[:train_size], y[train_size:val_size], y[val_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(n_steps, n_features)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics='mean_squared_error')
model.summary()


# In training these LSTM models, I monitored validation loss and stopped training when the metric starts to degrade or level off. Here are the reasons why I implemented early stopping:
# 1. Preventing Overfitting: Early stopping helps by halting training when the model's performance on a validation dataset starts to degrade, indicating overfitting.
# 
# 2. Optimizing Training Time: Training LSTMs can be computationally expensive and time-consuming. Early stopping allows me to save time and resources by avoiding unnecessary training epochs. When the model reaches an optimal level of performance, the model will stop training early rather than running for a fixed number of epochs.
# 
# 3. Automating Model Selection: Early stopping automates the process of selecting the optimal number of training epochs. Instead of manually specifying the number of epochs, early stopping dynamically determines the stopping point based on the model's performance.

# In[419]:


# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5,         
                               restore_best_weights=True) 

# Train the model
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val), callbacks=[early_stopping])


# In[420]:


# Create a range of epochs
best_epoch = 51
epochs = range(1, best_epoch+1)

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[421]:


# Make predictions
y_pred = model.predict(x_test)

# Inverse transform the predictions to get real values
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Temperature (°F)')
plt.title('Daily Max Temperature Forecasting with Multivariate LSTM')
plt.show()


# ### Univariate LSTM

# In[422]:


# Extract the temperature values and convert them to an array
temperatures = ncei_df['max temp'].values.astype(float)
logging.debug(f"Temperatures shape: {temperatures.shape}")

# Normalize the data to be in the range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
temperatures = scaler.fit_transform(temperatures.reshape(-1, 1))

# Define a function to create sequences for training the LSTM model
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

# Set the sequence length (number of past days to consider for prediction)
seq_length = 30

# Create sequences for training
x, y = create_sequences(temperatures, seq_length)

# Split the data into training (80%), validation (10%) and testing sets (10%)
train_size = int(0.8 * len(ncei_df))
val_size = int(0.9 * len(ncei_df))
x_train, x_val, x_test = x[:train_size], x[train_size:val_size], x[val_size:]
y_train, y_val, y_test =  y[:train_size], y[train_size:val_size], y[val_size:]

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(seq_length, 1)))
lstm_model.add(Dense(16, activation='relu'))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse', metrics='mean_squared_error')
lstm_model.summary()


# In[423]:


# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5,         
                               restore_best_weights=True) 
# Train the model
history = lstm_model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val), callbacks=[early_stopping])


# In[ ]:


best_epoch = 40
epochs = range(1, best_epoch+1)

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


# Make predictions on the test set
y_pred = lstm_model.predict(x_test)

# Inverse transform the scaled predictions to get actual temperature values
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred)

# Calculate the Root Mean Squared Error (RMSE) as a measure of prediction accuracy
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot the actual vs. predicted temperatures
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Temperature')
plt.plot(y_pred, label='Predicted Temperature')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Temperature (°F)')
plt.title('Daily Max Temperature Prediction with Univariate LSTM')
plt.show()


# After rigorous experimentation and training linear regression, univariate LSTM, and multivariate LSTM, I decided to choose the univariate LSTM model that was trained on NCEI's data because it exhibited superior performance. Beyond the metrics, the decision was also informed by the qualities of the data itself. NCEI's data presented as the more dependable and consistent source of information, enabling the machine learning to make more accurate and reliable predictions. Furthermore, the accessibility of NCEI's data, available for extraction without additional cost makes it a better option than VS's data.

# ## Kalshi's API Trading

# ### Predict today's maximum temperature

# In[ ]:


# Get recent NCEI's data to predict
start_date = datetime.datetime.strptime("2023-08-01", "%Y-%m-%d").date()
last_date = datetime.date.today() - datetime.timedelta(days=1)
api_recent = pd.DataFrame()

while start_date <= last_date:
    end_date = min(start_date + datetime.timedelta(days=100), last_date)
    params = {
        "datasetid": "GHCND",
        "stationid": "GHCND:USW00094728",
        "startdate": start_date,
        "enddate": end_date,
        "units": "standard",
        "datatypeid": "AWND,PRCP,SNOW,SNWD,TMAX,TMIN,WT01,WT03,WT08",
        "limit": 1000
    }
    # API request
    response = requests.get(url, params=params, headers={"token": "hQjOAltlsPnryPJlIEkjkzQqJFPtGOpe"})
    start_date = end_date + datetime.timedelta(days=1)
    if response.status_code == 200:
        print("API request successful")
        # Convert JSON to DataFrame
        json_data = pd.DataFrame(response.json()["results"])
        result = pd.DataFrame(json_data)
        api_recent = api_recent.append(json_data)
    else:
        print(f"API request failed with status code {response.status_code}")

ncei_df_recent = ncei_processing(api_recent)


# In[ ]:


# Extract the temperature values and convert them to an array
temperatures_recent = ncei_df_recent['max temp'].values.astype(float)

# Normalize the data to be in the range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
temperatures_recent = scaler.fit_transform(temperatures_recent.reshape(-1, 1))

# Create sequences for training
x, y = create_sequences(temperatures_recent, seq_length)

# Predict tomorrow's temperature
y_pred_recent = lstm_model.predict(x)

# Inverse transform the predicted value to get the actual temperature
y_pred_recent = scaler.inverse_transform(y_pred_recent)
y_pred_today = y_pred_recent[-1]
print(f"The maximum temperature today is: {y_pred_today}")


# ### Implement Kalshi's API to trade

# In[ ]:


# Add these imports at the top
import time
import uuid
from functools import wraps
from typing import Any, Dict


# Custom exceptions
class KalshiAPIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Kalshi API Error {status_code}: {message}")
        logging.error(f"Kalshi API Error {status_code}: {message}")

# Rate limiter class
class RateLimiter:
    def __init__(self, requests_per_second=10):
        self.requests_per_second = requests_per_second
        self.last_request_time = 0
        
    def wait_if_needed(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < (1.0 / self.requests_per_second):
            time.sleep((1.0 / self.requests_per_second) - time_since_last)
        self.last_request_time = time.time()

# Helper functions
def get_headers(token: str) -> Dict[str, str]:
    """Get standard headers required for authenticated requests"""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

def handle_response(response):
    """Handle API response and raise appropriate errors"""
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 401:
        raise KalshiAPIError(401, "Unauthorized - check credentials")
    elif response.status_code == 429:
        raise KalshiAPIError(429, "Rate limit exceeded")
    else:
        raise KalshiAPIError(
            response.status_code,
            f"API request failed: {response.text}"
        )

# Main Kalshi client class
class KalshiClient:
    def __init__(self, email: str, password: str, demo: bool = True):
        self.email = email
        self.password = password
        self.demo = demo
        self.base_url = "https://demo-api.kalshi.co/trade-api/v2" if demo else "https://trading-api.kalshi.com/trade-api/v2"
        self.token = None
        self.rate_limiter = RateLimiter()
        self.login()

    def login(self):
        """Login and get auth token"""
        login_url = f"{self.base_url}/login"
        login_data = {
            "email": self.email,
            "password": self.password
        }
        response = requests.post(login_url, json=login_data)
        data = handle_response(response)
        self.token = data["token"]

    def get_exchange_status(self):
        """Get exchange status"""
        self.rate_limiter.wait_if_needed()
        response = requests.get(
            f"{self.base_url}/exchange/status",
            headers=get_headers(self.token)
        )
        return handle_response(response)

    def get_event(self, event_ticker: str):
        """Get event details"""
        self.rate_limiter.wait_if_needed()
        response = requests.get(
            f"{self.base_url}/events/{event_ticker}",
            headers=get_headers(self.token)
        )
        return handle_response(response)

    def create_order(self, order_params: Dict[str, Any]):
        """Create a new order"""
        # Validate required fields
        required_fields = ['ticker', 'action', 'side', 'count', 'type']
        for field in required_fields:
            if field not in order_params:
                raise ValueError(f"Missing required field: {field}")
        
        # Add client_order_id if not provided
        if 'client_order_id' not in order_params:
            order_params['client_order_id'] = str(uuid.uuid4())

        self.rate_limiter.wait_if_needed()
        response = requests.post(
            f"{self.base_url}/portfolio/orders",
            headers=get_headers(self.token),
            json=order_params
        )
        return handle_response(response)

    def get_market_prices(self, ticker: str):
        """Get current market prices for a ticker"""
        self.rate_limiter.wait_if_needed()
        response = requests.get(
            f"{self.base_url}/markets/{ticker}",
            headers=get_headers(self.token)
        )
        return handle_response(response)

# Usage example
def trade_temperature():
    try:
        # Initialize client
        client = KalshiClient(
            email="richardadonnell@gmail.com",
            password="akg_UYA-eqj4zqn7udv",
            demo=True
        )

        # Check exchange status
        status = client.get_exchange_status()
        logging.info(f"Exchange status: {status}")

        # Get today's event ticker
        today_date = datetime.date.today()
        event_ticker = f'HIGHNY-{today_date.strftime("%y%b").upper()}{today_date.strftime("%d")}'

        # Get event details
        event = client.get_event(event_ticker)
        markets = event.get('markets', [])
        
        if not markets:
            raise ValueError(f"No markets found for event {event_ticker}")

        # Process markets and create order
        temp_list = []
        market_tickers = []
        for m in markets:
            subtitle = m['subtitle'].split()
            market_tickers.append(m['ticker'])
            if "or" in subtitle:
                temp_list.append(int(subtitle[0][:-1]))
            else:
                temp_list.append((int(subtitle[0][:-1]) + int(subtitle[-1][:-1])) / 2)

        # Find market with furthest distance from prediction
        i = np.argmax(abs(np.array(temp_list) - y_pred_today))
        selected_ticker = market_tickers[i]

        # Create order
        order_params = {
            'ticker': selected_ticker,
            'type': 'market',
            'action': 'buy',
            'side': 'no',
            'count': 10
        }

        order_response = client.create_order(order_params)
        logging.info(f"Order placed: {order_response}")

    except KalshiAPIError as e:
        logging.error(f"API Error: {e.message}")
    except Exception as e:
        logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    trade_temperature()

