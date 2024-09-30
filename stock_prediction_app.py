import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
import streamlit as st
from datetime import datetime

# Streamlit App Configuration
st.set_page_config(page_title="📈 Stock Price Prediction", page_icon="📊", layout="wide")

import streamlit as st

# Define custom CSS
st.markdown(
    """
    <style>
    /* Main app background and text color */
    .stApp {
        background-color: white; /* background for the main app */
        color: black; /* Dark gray text for contrast */
    }

    /* Headings color for the main app */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #452c63 !important; /* Dark purple for headings */
    }

    /* Sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #5A0DA6; /* Dark purple background for the sidebar */
    }

    /* Sidebar text color */
    [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .css-1lcbmhc {
        color: white; /* White color for all text in the sidebar */
    }

    /* Sidebar headings color */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: white !important; /* White color for headings in the sidebar */
    }

    /* Input box and other interactive elements in the sidebar */
    [data-testid="stSidebar"] .css-1d8p8hv, [data-testid="stSidebar"] .css-2trqyj {
        background-color: #604879; /* Darker purple shade for input boxes */
        color: black; /* Black text for input boxes */
    }

    /* Customize selectbox and slider styles in the sidebar */
    [data-testid="stSidebar"] .css-10trblm, [data-testid="stSidebar"] .css-1ekf4uk {
        background-color: #5a4577; /* Dark shade for selectboxes and sliders */
        color: black; /* Black text for selectboxes and sliders */
    }

    /* Make sidebar label text white */
    [data-testid="stSidebar"] .css-1d1ql6, [data-testid="stSidebar"] label {
        color: white; /* White color for labels */
    }

    /* Customize the slider track and handle */
    .css-1wa3eu0 { /* Slider track */
        background-color: #32CD32 !important; /* Green for the slider track */
        border-radius: 4px; /* Optional: Adds rounded corners to the track */
    }

    .css-1wy8k1p { /* Slider handle */
        background-color: #32CD32 !important; /* Green for the slider handle */
    }

    /* Customize the text below the slider */
    [data-testid="stSidebar"] .stMarkdown p {
        color: white !important; /* White for the description below the slider */
    }

    </style>
    """,
    unsafe_allow_html=True
)





# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('multi_company_stock_data.csv')
    df['date_column'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    return df

df = load_data()

# Function to clean NaN values from the data dictionary
def clean_data_dict(data_dict):
    cleaned_dict = {}
    for company, data in data_dict.items():
        print(f"Processing {company}...")

        # Extract X_test and y_test
        X_test = data['X_test']
        y_test = data['y_test']

        # Initialize masks for NaN detection
        nan_mask_X = np.ones(X_test.shape[0], dtype=bool)
        nan_mask_y = np.ones(y_test.shape[0], dtype=bool)

        # Check and remove NaN values from X_test
        if np.any(np.isnan(X_test)):
            print("NaN found in X_test")
            nan_mask_X = ~np.isnan(X_test).any(axis=1)
        
        # Check and remove NaN values from y_test
        if np.any(np.isnan(y_test)):
            print("NaN found in y_test")
            nan_mask_y = ~np.isnan(y_test)

        # Combine masks to remove rows with NaN in either X_test or y_test
        final_mask = nan_mask_X & nan_mask_y
        X_test = X_test[final_mask]
        y_test = y_test[final_mask]

        # Save cleaned data back to the dictionary
        cleaned_dict[company] = {
            'X_train': data['X_train'],  # Include training data as is
            'y_train': data['y_train'],  # Include training data as is
            'X_test': X_test,
            'y_test': y_test,
            'scaler': data['scaler'],  # Keep the scaler as is
            'scaled_close': data['scaled_close']  # Keep the scaled_close as is
        }

    return cleaned_dict

# List of companies
companies = ['BAJFINANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'ITC', 'KOTAKBANK', 'RELIANCE', 'SBIN', 'TCS']

# Dictionary to store data for each company
data_dict = {}

for company in companies:
    # Filter data for the current company
    company_data = df[df[f'Company_{company}'] == 1].reset_index()
    
    # Extract the 'close' column
    company_close = company_data['close']
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(np.array(company_close).reshape(-1, 1))
    
    # Split the dataset
    training_size = int(len(scaled_close) * 0.65)
    train_data, test_data = scaled_close[0:training_size, :], scaled_close[training_size:, :1]

    # Create datasets
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    # Reshape for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Store the data in the dictionary
    data_dict[company] = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'scaled_close': scaled_close  # Added for future prediction
    }

# Clean the data_dict from NaN values
data_dict = clean_data_dict(data_dict)

# Train or load the model with caching
@st.cache_resource
def get_model(company, X_train, y_train, X_test, y_test):
    try:
        model = load_model(f'{company}_lstm_model.keras')
    except:
        st.write(f"Training new model for {company}...")
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)
        model.save(f'{company}_lstm_model.keras')
    return model

# Load or train models for each company
model_dict = {company: get_model(company, data['X_train'], data['y_train'], data['X_test'], data['y_test'])
for company, data in data_dict.items()}

# Plot predictions function with error handling
# Function to plot the last 50 dates' predictions
def plot_predictions(company, model, X_test, y_test, scaler):
    # Predict using the model
    test_predict = model.predict(X_test)

    # Inverse transform to get the original scale
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Ensure y_test and test_predict have the same length
    min_len = min(len(y_test), len(test_predict))
    y_test = y_test[:min_len]
    test_predict = test_predict[:min_len]

    # Extract the last 50 days
    last_50_actual = y_test[-50:]
    last_50_predicted = test_predict[-50:]

    # Calculate RMSE for the last 50 days
    rmse_last_50 = math.sqrt(mean_squared_error(last_50_actual, last_50_predicted))
    
    # Plot predictions for the last 50 dates
    plt.figure(figsize=(12, 6))
    plt.plot(last_50_actual, label='Actual (Last 50 Days)', color='blue')
    plt.plot(last_50_predicted, label='Predicted (Last 50 Days)', color='green')
    plt.title(f'{company} Stock Price Prediction (Last 50 Days)')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    
# Function to predict future stock prices
def predict_future(company, model, data, days=30):
    test_data = data['X_test']
    scaler = data['scaler']
    time_step = 100  # Define time_step based on the shape of X_test
    
    # Ensure test_data is reshaped correctly
    x_input = test_data[-1].reshape(1, -1)
    temp_input = x_input.flatten().tolist()

    lst_output = []

    # Predict future values
    for i in range(days):
        x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)  # Reshape to match model input shape
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())

    # Convert predictions back to the original scale by reshaping to 2D
    historical_data = scaler.inverse_transform(test_data[-time_step:].reshape(-1, test_data.shape[2]))
    future_predictions = scaler.inverse_transform(np.array(lst_output).reshape(-1, test_data.shape[2]))

    # Save the last 100 values of historical data
    if len(historical_data) > 100:
        last_100_values = historical_data[-100:]
    else:
        last_100_values = historical_data  # If less than 100 values, take all

    # Generate future dates
    last_date = df['date_column'].max()
    historical_dates = [last_date - timedelta(days=x) for x in range(time_step)][::-1]  # Last `time_step` days
    future_dates = [last_date + timedelta(days=x) for x in range(1, days + 1)]  # Next `days` days

    # Plot future predictions
    plt.figure(figsize=(12, 6))
    # Plot historical data
    plt.plot(historical_dates, last_100_values[:, 0], label='Historical Data')
    # Plot future predictions
    plt.plot(future_dates[:days], future_predictions[:days, 0], label='Future Predictions')
    # Format x-axis to show dates
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=7))
    plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
    # Plot titles and labels
    plt.title(f'{company} Stock Close Price Prediction for Next {days} Days')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot(plt)
    
    
    # Prepare data for DataFrame
    future_data = {
        'Date': future_dates[:days],
        'Predicted Price (Rs)': future_predictions[:days, 0]
    }
    
    future_df = pd.DataFrame(future_data)
    
    # Format the 'Predicted Price (Rs)' column with Rs symbol
    future_df['Predicted Price (Rs)'] = future_df['Predicted Price (Rs)'].apply(lambda x: f"₹{x:.2f}")

    return future_df


# Sidebar for user input with styling
st.sidebar.title("📊 Stock Price Prediction Dashboard")
st.sidebar.markdown("---")  # Adds a horizontal line for separation

# Select a company with an icon
selected_company = st.sidebar.selectbox("🏢 Select a Company:", companies)

# Date range input with emojis and a heading
st.sidebar.header("📅 Date Range Selection")
start_date = st.sidebar.date_input('Start Date', df['date_column'].min())
end_date = st.sidebar.date_input('End Date', df['date_column'].max())

# Separator for styling
st.sidebar.markdown("---")

# Input for the number of days to predict with a slider
st.sidebar.header("🔮 Prediction Settings")
days_to_predict = st.sidebar.slider("Select Number of Days to Predict:", min_value=1, max_value=50, value=10)
st.sidebar.markdown("Adjust the slider to choose how many days ahead you want to predict.")

# Add a button for future price prediction
st.sidebar.header("✨ Predict Future Prices")

# Main page layout with columns for better organization
st.title(f"📉 {selected_company} Stock Price Prediction")

#st.markdown("### Overview of Stock Data and Predictions")
st.write(f"Displaying historical and predicted data for **{selected_company}**. Select a different company or adjust settings from the sidebar to explore more.")

# Two-column layout for displaying plots and information side-by-side
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Stock Price and Indicators")
    # Plot predictions and stock indicators
    try:
        plot_predictions(
            selected_company,
            model_dict[selected_company],
            data_dict[selected_company]['X_test'],
            data_dict[selected_company]['y_test'],
            data_dict[selected_company]['scaler']
        )
    except Exception as e:
        st.error(f"Error plotting predictions: {e}")

with col2:
    st.subheader("📈 Key Insights and Analysis")
    st.write("Visualize the trends and make informed decisions based on the data presented.")
    st.markdown("- **Close Price** shows the daily closing price.")
    st.markdown("- **Open Price** shows the daily opening price.")
    
# Two-column layout for displaying plots and information side-by-side
col3, col4 = st.columns(2)

with col3:
    if st.sidebar.button(f'Predict {days_to_predict} Days of Stock Prices for {selected_company}'):
        try:
            # Predict future stock prices
            future_df = predict_future(selected_company, model_dict[selected_company], data_dict[selected_company], days=days_to_predict)
            
            # Display future predictions in a table
            st.success(f"Future {days_to_predict} Day Prediction for {selected_company}:")
            st.table(future_df)
        except Exception as e:
            st.error(f"Error predicting future stock prices: {e}")

# Footer
st.markdown("---")
st.markdown("Developed by **Investra** | © 2024 Stock Prediction App. All rights reserved.")


