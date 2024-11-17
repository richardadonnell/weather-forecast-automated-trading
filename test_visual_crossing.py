#!/usr/bin/env python
# coding: utf-8

import logging
import time
import urllib.parse
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_visual_crossing_api(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Test function for Visual Crossing API with improved rate limiting and chunking.
    """
    VC_API_KEY = "WS8SPUNTV45687SM8PE4SP8EC"
    VC_BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    
    # Test single day first
    test_date = datetime.strptime(start_date, "%Y-%m-%d")
    location = "New York City,USA"
    
    # Test parameters
    params = {
        "unitGroup": "us",
        "include": "days",
        "key": VC_API_KEY,
        "contentType": "json",
        "elements": "datetime,tempmax",
        "options": "nonulls"
    }
    
    # Construct test URL
    url = f"{VC_BASE_URL}/{urllib.parse.quote(location)}/{test_date.strftime('%Y-%m-%d')}"
    
    logging.info(f"Testing single day request for {test_date.strftime('%Y-%m-%d')}")
    
    try:
        # Make test request
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            logging.info("Single day test successful!")
            logging.info(f"Response: {response.json()}")
        else:
            logging.error(f"Test failed with status code: {response.status_code}")
            logging.error(f"Response: {response.text}")
            return pd.DataFrame()
            
        # Wait before next test
        time.sleep(60)  # Wait 1 minute before multi-day test
        
        # Test multi-day request with minimal data
        params["elements"] = "datetime,tempmax"  # Minimize data requested
        
        # Test URL for 3 days
        end_test_date = test_date + timedelta(days=2)
        url = f"{VC_BASE_URL}/{urllib.parse.quote(location)}/{test_date.strftime('%Y-%m-%d')}/{end_test_date.strftime('%Y-%m-%d')}"
        
        logging.info(f"Testing multi-day request from {test_date.strftime('%Y-%m-%d')} to {end_test_date.strftime('%Y-%m-%d')}")
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            logging.info("Multi-day test successful!")
            data = response.json()
            if 'days' in data:
                df = pd.DataFrame(data['days'])
                logging.info(f"Retrieved {len(df)} days of data")
                return df
            else:
                logging.error("No data found in response")
                return pd.DataFrame()
        else:
            logging.error(f"Multi-day test failed with status code: {response.status_code}")
            logging.error(f"Response: {response.text}")
            return pd.DataFrame()
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {str(e)}")
        return pd.DataFrame()

def main():
    """
    Main test function
    """
    # Test with recent dates
    start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    logging.info("Starting Visual Crossing API test")
    
    df = test_visual_crossing_api(start_date, end_date)
    
    if not df.empty:
        logging.info("Test successful! Displaying results:")
        logging.info("\nRetrieved Data:")
        logging.info(df)
        
        # Create simple visualization of test data
        df['datetime'] = pd.to_datetime(df['datetime'])
        plt.figure(figsize=(10, 6))
        plt.plot(df['datetime'], df['tempmax'], marker='o')
        plt.title('Test Data: Maximum Temperature')
        plt.xlabel('Date')
        plt.ylabel('Temperature (°F)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        logging.error("Test failed - no data retrieved")

if __name__ == "__main__":
    main() 