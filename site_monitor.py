import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# Set page configuration
st.set_page_config(
    page_title="USGS Water Data Monitor",
    page_icon="💧",
    layout="wide"
)

# App title and description
st.title("USGS Water Data Monitor")
st.markdown("""
This application monitors turbidity and flow measurements from USGS water data sites.
It allows you to check these measurements against optimal thresholds and predict future values.
""")

# Define states for site search
STATES = {
    "Texas": "TX",
    "Utah": "UT"
}

# Function to search for sites in a given state
@st.cache_data(ttl=3600)
def search_sites(state_code):
    """Search for USGS water sites in the given state."""
    base_url = "https://waterservices.usgs.gov/nwis/site/"
    params = {
        "format": "rdb",
        "stateCd": state_code,
        "parameterCd": "00010,00060,63680",  # Temperature, Flow, Turbidity
        "siteStatus": "active"
    }
    
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            # Parse the RDB format (tab-delimited with header lines)
            lines = response.text.splitlines()
            data_lines = [line for line in lines if not line.startswith("#") and len(line.strip()) > 0]
            
            if len(data_lines) < 2:
                return pd.DataFrame()
            
            # Extract headers and data
            headers = data_lines[0].split("\t")
            data_start = 2  # Skip the header line and the dash line
            
            # Create a list of dictionaries for each site
            sites_data = []
            for i in range(data_start, len(data_lines)):
                values = data_lines[i].split("\t")
                if len(values) == len(headers):
                    site_dict = {headers[j]: values[j] for j in range(len(headers))}
                    sites_data.append(site_dict)
            
            # Convert to DataFrame
            sites_df = pd.DataFrame(sites_data)
            
            # Select relevant columns if they exist
            relevant_columns = ["site_no", "station_nm", "dec_lat_va", "dec_long_va", "state_cd"]
            existing_columns = [col for col in relevant_columns if col in sites_df.columns]
            
            if not existing_columns:
                return pd.DataFrame()
                
            return sites_df[existing_columns]
        else:
            st.error(f"Error: Unable to fetch data. Status code: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return pd.DataFrame()

# Function to get time series data for a site
@st.cache_data(ttl=1800)
def get_site_data(site_id, parameter_codes, start_date, end_date):
    """Get time series data for a specific site and parameters."""
    base_url = "https://waterservices.usgs.gov/nwis/iv/"
    params = {
        "format": "json",
        "sites": site_id,
        "startDT": start_date,
        "endDT": end_date,
        "parameterCd": parameter_codes,
        "siteStatus": "all"
    }
    
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            
            # Extract time series data
            all_series = []
            if 'value' in data and 'timeSeries' in data['value']:
                for series in data['value']['timeSeries']:
                    param_code = series['variable']['variableCode'][0]['value']
                    param_name = series['variable']['variableName']
                    param_unit = series['variable']['unit']['unitCode']
                    
                    # Extract values
                    if 'values' in series and len(series['values']) > 0:
                        values = series['values'][0]['value']
                        
                        # Create DataFrame for this series
                        series_df = pd.DataFrame(values)
                        if 'dateTime' in series_df.columns and 'value' in series_df.columns:
                            series_df['dateTime'] = pd.to_datetime(series_df['dateTime'])
                            series_df['value'] = pd.to_numeric(series_df['value'], errors='coerce')
                            series_df['parameter'] = param_name
                            series_df['parameterCode'] = param_code
                            series_df['unit'] = param_unit
                            all_series.append(series_df)
            
            if all_series:
                # Combine all series
                combined_df = pd.concat(all_series, ignore_index=True)
                return combined_df
            else:
                return pd.DataFrame()
        else:
            st.error(f"Error: Unable to fetch data. Status code: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return pd.DataFrame()

# Function to prepare data for ML prediction
def prepare_data_for_prediction(df, parameter_name, model_type="rf"):
    """Prepare data for machine learning prediction."""
    # Filter data for the specific parameter
    param_df = df[df['parameter'] == parameter_name].copy()
    
    if param_df.empty:
        return None, None, None, None
    
    # Sort by datetime
    param_df = param_df.sort_values('dateTime')
    
    # For time series models, we need a clean datetime index and value series
    ts_df = param_df.copy()
    ts_df = ts_df[['dateTime', 'value']].set_index('dateTime')
    ts_df = ts_df.resample('H').mean().fillna(method='ffill')  # Hourly resampling for uniform time series
    
    # For ML models, create features
    if model_type == "rf":
        # Create features for Random Forest
        param_df['hour'] = param_df['dateTime'].dt.hour
        param_df['day'] = param_df['dateTime'].dt.day
        param_df['month'] = param_df['dateTime'].dt.month
        param_df['dayofweek'] = param_df['dateTime'].dt.dayofweek
        param_df['year'] = param_df['dateTime'].dt.year
        param_df['quarter'] = param_df['dateTime'].dt.quarter
        
        # Create lag features (previous values)
        for i in range(1, 25):  # 24 hour lags
            param_df[f'lag_{i}h'] = param_df['value'].shift(i)
        
        # Create rolling mean features
        for window in [6, 12, 24, 48]:
            param_df[f'rolling_mean_{window}h'] = param_df['value'].rolling(window=window).mean()
        
        # Drop NaN values after creating lag features
        param_df = param_df.dropna()
        
        # Target variable
        y = param_df['value'].values
        
        # Basic time features
        time_features = ['hour', 'day', 'month', 'dayofweek', 'year', 'quarter']
        
        # Lag features
        lag_features = [f'lag_{i}h' for i in range(1, 25)]
        
        # Rolling mean features
        rolling_features = [f'rolling_mean_{window}h' for window in [6, 12, 24, 48]]
        
        # Combine all features
        all_features = time_features + lag_features + rolling_features
        
        # Features
        X = param_df[all_features].values
        
        # Feature names for importance analysis
        feature_names = all_features
        
        # Original datetimes for plotting
        dates = param_df['dateTime'].values
        
        return X, y, dates, feature_names, ts_df
    
    else:
        # For time series models, return the resampled DataFrame
        return None, None, None, None, ts_df

# Function to train prediction model
def train_prediction_model(X, y, feature_names, ts_df, parameter_name, model_type="rf"):
    """Train a prediction model based on the specified type."""
    
    if model_type == "rf" and X is not None and len(X) >= 24:  # Need enough data points for RF
        # Split data for Random Forest
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {"type": "rf", "model": model, "mse": mse, "r2": r2, "feature_names": feature_names}
    
    elif model_type == "arima" and ts_df is not None and len(ts_df) >= 24:
        try:
            # Fit ARIMA model (p,d,q) parameters
            # p: AR order, d: differencing, q: MA order
            model = ARIMA(ts_df, order=(5,1,0))
            model_fit = model.fit()
            
            # Get in-sample predictions and calculate metrics
            in_sample_pred = model_fit.fittedvalues
            mse = mean_squared_error(ts_df['value'][5:], in_sample_pred)
            r2 = r2_score(ts_df['value'][5:], in_sample_pred)
            
            return {"type": "arima", "model": model_fit, "mse": mse, "r2": r2}
        except Exception as e:
            st.warning(f"Error fitting ARIMA model: {str(e)}")
            return None
    
    elif model_type == "prophet" and ts_df is not None and len(ts_df) >= 24:
        try:
            # Prepare data for Prophet - ensure datetime has no timezone
            prophet_df = pd.DataFrame({
                'ds': ts_df.index.tz_localize(None),
                'y': ts_df['value'].values
            })
            
            # Train Prophet model
            model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
            model.fit(prophet_df)
            
            # Make in-sample predictions for metrics
            future = model.make_future_dataframe(periods=0, freq='H')
            forecast = model.predict(future)
            
            # Calculate metrics on the training data
            prophet_pred = forecast['yhat'].values
            prophet_actual = prophet_df['y'].values[:len(prophet_pred)]
            
            mse = mean_squared_error(prophet_actual, prophet_pred)
            r2 = r2_score(prophet_actual, prophet_pred)
            
            return {"type": "prophet", "model": model, "mse": mse, "r2": r2}
        except Exception as e:
            st.warning(f"Error fitting Prophet model: {str(e)}")
            return None
    
    elif model_type == "holt_winters" and ts_df is not None and len(ts_df) >= 48:
        try:
            # Fit Holt-Winters Exponential Smoothing
            model = ExponentialSmoothing(
                ts_df['value'],
                trend='add',
                seasonal='add',
                seasonal_periods=24  # Assuming daily seasonality with hourly data
            )
            model_fit = model.fit()
            
            # Get in-sample predictions and calculate metrics
            in_sample_pred = model_fit.fittedvalues
            mse = mean_squared_error(ts_df['value'][24:], in_sample_pred[24:])
            r2 = r2_score(ts_df['value'][24:], in_sample_pred[24:])
            
            return {"type": "holt_winters", "model": model_fit, "mse": mse, "r2": r2}
        except Exception as e:
            st.warning(f"Error fitting Holt-Winters model: {str(e)}")
            return None
    
    else:
        return None

# Function to predict future values
def predict_future(model_results, last_date, ts_df, days_ahead=7):
    """Predict future values based on the trained model."""
    if model_results is None:
        return None, None
    
    model_type = model_results["type"]
    model = model_results["model"]
    
    # Generate future dates (hourly frequency for all models)
    future_dates = pd.date_range(
        start=last_date,
        periods=days_ahead*24+1,  # hourly predictions for the specified days
        freq='H'
    )[1:]  # Skip the first entry which is the last observed date
    
    if model_type == "rf":
        # Create features for future dates
        future_features = []
        
        # Get the feature names
        feature_names = model_results["feature_names"]
        
        # For lag and rolling features, we need to make sequential predictions
        # Start with the last known values
        last_values = list(ts_df['value'].tail(48))  # Get last 48 hours of data
        
        # Make hourly predictions for the future period
        hourly_predictions = []
        
        for future_date in future_dates:
            # Prepare time features
            time_feats = [
                future_date.hour,
                future_date.day,
                future_date.month,
                future_date.weekday(),
                future_date.year,
                future_date.quarter
            ]
            
            # Prepare lag features (using actual data and then predictions)
            lag_feats = last_values[-24:]  # Last 24 hours
            
            # Prepare rolling mean features
            rolling_6h = np.mean(last_values[-6:])
            rolling_12h = np.mean(last_values[-12:])
            rolling_24h = np.mean(last_values[-24:])
            rolling_48h = np.mean(last_values[-48:])
            
            rolling_feats = [rolling_6h, rolling_12h, rolling_24h, rolling_48h]
            
            # Combine all features
            all_feats = time_feats + lag_feats + rolling_feats
            
            # Make prediction for this hour
            pred = model.predict([all_feats])[0]
            hourly_predictions.append(pred)
            
            # Update last_values with the new prediction for next iteration
            last_values.append(pred)
        
        # Return future dates and predictions
        return future_dates, hourly_predictions
    
    elif model_type == "arima":
        # Use ARIMA's built-in forecast method
        forecast_result = model.forecast(steps=len(future_dates))
        return future_dates, forecast_result
    
    elif model_type == "prophet":
        # Create future dataframe for Prophet
        future = model.make_future_dataframe(periods=len(future_dates), freq='H')
        forecast = model.predict(future)
        
        # Get the predictions for the future dates only
        future_predictions = forecast.tail(len(future_dates))['yhat'].values
        
        return future_dates, future_predictions
    
    elif model_type == "holt_winters":
        # Use Holt-Winters forecast method
        forecast_result = model.forecast(steps=len(future_dates))
        return future_dates, forecast_result
    
    else:
        return None, None

# Sidebar for configuration
st.sidebar.header("Configuration")

# State selection
selected_state = st.sidebar.selectbox("Select State", list(STATES.keys()))
state_code = STATES[selected_state]

# Site selection
site_input_method = st.sidebar.radio("Site Selection Method", ["Search Sites", "Enter Site ID"])

if site_input_method == "Search Sites":
    # Search for sites in the selected state
    sites_df = search_sites(state_code)
    
    if not sites_df.empty:
        site_options = [f"{row['site_no']} - {row['station_nm']}" for _, row in sites_df.iterrows()]
        selected_site_option = st.sidebar.selectbox("Select Site", site_options)
        selected_site_id = selected_site_option.split(" - ")[0].strip()
    else:
        st.sidebar.warning(f"No sites found in {selected_state}. Try entering a site ID directly.")
        selected_site_id = st.sidebar.text_input("Enter Site ID", "08068500")
else:
    selected_site_id = st.sidebar.text_input("Enter Site ID", "08068500")

# Date range selection
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
date_range = st.sidebar.date_input(
    "Select Date Range",
    [start_date.date(), end_date.date()],
    min_value=end_date - timedelta(days=365),
    max_value=end_date
)

if len(date_range) == 2:
    start_date = date_range[0]
    end_date = date_range[1]

# Parameter codes (00060=flow, 63680=turbidity)
parameter_codes = "00060,63680"

# Threshold settings
st.sidebar.header("Threshold Settings")
with st.sidebar.expander("Configure Thresholds"):
    threshold_flow = st.number_input("Optimal Flow (CFS)", value=40.0, step=1.0)
    threshold_gauge_min = st.number_input("Optimal Gauge Height Min (ft)", value=73.0, step=0.1)
    threshold_gauge_max = st.number_input("Optimal Gauge Height Max (ft)", value=74.0, step=0.1)
    threshold_turbidity = st.number_input("Optimal Turbidity (Max)", value=25.0, step=1.0)

# Prediction settings
st.sidebar.header("Prediction Settings")
days_to_predict = st.sidebar.slider("Days to Predict", min_value=1, max_value=14, value=7)

# Load and display data
if selected_site_id:
    st.header(f"Data for Site: {selected_site_id}")
    
    with st.spinner("Loading data..."):
        # Get data
        df = get_site_data(
            selected_site_id, 
            parameter_codes, 
            start_date.strftime("%Y-%m-%d"), 
            end_date.strftime("%Y-%m-%d")
        )
    
    if not df.empty:
        # Display tabs for different views
        tab1, tab2, tab3 = st.tabs(["Current Data", "Historical Analysis", "Predictions"])
        
        with tab1:
            st.subheader("Current Measurements")
            
            # Get the latest measurements for each parameter
            latest_values = {}
            for param in df['parameter'].unique():
                param_df = df[df['parameter'] == param]
                if not param_df.empty:
                    latest_param_df = param_df.sort_values('dateTime', ascending=False).iloc[0]
                    latest_values[param] = {
                        'value': latest_param_df['value'],
                        'dateTime': latest_param_df['dateTime'],
                        'unit': latest_param_df['unit']
                    }
            
            # Display latest values in metrics
            cols = st.columns(len(latest_values))
            
            for i, (param, data) in enumerate(latest_values.items()):
                with cols[i]:
                    # Determine if value is within optimal range
                    is_optimal = False
                    if "Streamflow" in param and abs(data['value'] - threshold_flow) <= threshold_flow * 0.1:
                        is_optimal = True
                    elif "Gage height" in param and threshold_gauge_min <= data['value'] <= threshold_gauge_max:
                        is_optimal = True
                    elif "Turbidity" in param and data['value'] <= threshold_turbidity:
                        is_optimal = True
                    
                    # Display with color coding
                    value_display = f"{data['value']:.2f} {data['unit']}"
                    datetime_display = data['dateTime'].strftime("%Y-%m-%d %H:%M")
                    
                    if is_optimal:
                        st.metric(
                            label=f"{param} (OPTIMAL)", 
                            value=value_display,
                            delta=datetime_display,
                            delta_color="off"
                        )
                    else:
                        st.metric(
                            label=param, 
                            value=value_display,
                            delta=datetime_display,
                            delta_color="off"
                        )
            
            # Display thresholds
            st.subheader("Configured Thresholds")
            threshold_cols = st.columns(3)
            with threshold_cols[0]:
                st.info(f"Optimal Flow: {threshold_flow} CFS")
            with threshold_cols[1]:
                st.info(f"Optimal Gauge Height: {threshold_gauge_min} to {threshold_gauge_max} ft")
            with threshold_cols[2]:
                st.info(f"Optimal Turbidity: ≤ {threshold_turbidity}")
            
            # Plot recent data
            st.subheader("Recent Measurements")
            
            # Separate each parameter into its own plot
            for param in df['parameter'].unique():
                param_df = df[df['parameter'] == param].sort_values('dateTime')
                
                if not param_df.empty:
                    fig = px.line(
                        param_df, 
                        x='dateTime', 
                        y='value',
                        title=f"{param} ({param_df['unit'].iloc[0]})",
                        labels={'dateTime': 'Date & Time', 'value': f"Value ({param_df['unit'].iloc[0]})"}
                    )
                    
                    # Add threshold lines if applicable
                    if "Streamflow" in param:
                        fig.add_hline(
                            y=threshold_flow,
                            line_dash="dash",
                            line_color="green",
                            annotation_text="Optimal Flow"
                        )
                    elif "Gage height" in param:
                        fig.add_hline(
                            y=threshold_gauge_min,
                            line_dash="dash",
                            line_color="green",
                            annotation_text="Min Optimal Height"
                        )
                        fig.add_hline(
                            y=threshold_gauge_max,
                            line_dash="dash",
                            line_color="green",
                            annotation_text="Max Optimal Height"
                        )
                    elif "Turbidity" in param:
                        fig.add_hline(
                            y=threshold_turbidity,
                            line_dash="dash",
                            line_color="green",
                            annotation_text="Optimal Turbidity Max"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Historical Analysis")
            
            # Calculate daily statistics
            df['date'] = df['dateTime'].dt.date
            daily_stats = df.groupby(['parameter', 'date']).agg(
                mean_value=('value', 'mean'),
                min_value=('value', 'min'),
                max_value=('value', 'max')
            ).reset_index()
            
            # Plot daily statistics
            for param in daily_stats['parameter'].unique():
                param_stats = daily_stats[daily_stats['parameter'] == param]
                
                if not param_stats.empty:
                    fig = go.Figure()
                    
                    # Add mean line
                    fig.add_trace(go.Scatter(
                        x=param_stats['date'],
                        y=param_stats['mean_value'],
                        mode='lines',
                        name='Mean',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Add min and max range
                    fig.add_trace(go.Scatter(
                        x=param_stats['date'],
                        y=param_stats['min_value'],
                        mode='lines',
                        name='Min',
                        line=dict(color='lightblue', width=1)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=param_stats['date'],
                        y=param_stats['max_value'],
                        mode='lines',
                        name='Max',
                        line=dict(color='lightblue', width=1),
                        fill='tonexty'  # Fill area between min and max
                    ))
                    
                    fig.update_layout(
                        title=f"Daily {param} Statistics",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        legend_title="Statistic"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Display optimal conditions percentage
            st.subheader("Optimal Conditions Analysis")
            
            # Calculate percentage of time conditions were optimal
            optimal_stats = []
            
            for param in df['parameter'].unique():
                param_df = df[df['parameter'] == param]
                total_records = len(param_df)
                
                if total_records > 0:
                    optimal_count = 0
                    
                    if "Streamflow" in param:
                        optimal_count = sum(abs(param_df['value'] - threshold_flow) <= threshold_flow * 0.1)
                    elif "Gage height" in param:
                        optimal_count = sum((param_df['value'] >= threshold_gauge_min) & 
                                           (param_df['value'] <= threshold_gauge_max))
                    elif "Turbidity" in param:
                        optimal_count = sum(param_df['value'] <= threshold_turbidity)
                    
                    optimal_pct = (optimal_count / total_records) * 100
                    
                    optimal_stats.append({
                        'Parameter': param,
                        'Optimal Percentage': optimal_pct,
                        'Optimal Count': optimal_count,
                        'Total Records': total_records
                    })
            
            if optimal_stats:
                # Display as a bar chart
                optimal_df = pd.DataFrame(optimal_stats)
                
                fig = px.bar(
                    optimal_df,
                    x='Parameter',
                    y='Optimal Percentage',
                    title="Percentage of Time in Optimal Range",
                    labels={'Parameter': 'Parameter', 'Optimal Percentage': 'Optimal %'},
                    color='Optimal Percentage',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                
                fig.update_layout(yaxis_range=[0, 100])
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display as a table
                st.dataframe(optimal_df)
        
        with tab3:
            st.subheader("Predictions")
            
            # Model selection
            model_type = st.selectbox(
                "Select Prediction Model",
                ["Random Forest (ML)", "ARIMA (Time Series)", "Prophet (Time Series)", "Holt-Winters (Time Series)"],
                help="Choose the type of model for predictions"
            )
            
            # Map selected model to internal code
            model_map = {
                "Random Forest (ML)": "rf",
                "ARIMA (Time Series)": "arima",
                "Prophet (Time Series)": "prophet",
                "Holt-Winters (Time Series)": "holt_winters"
            }
            selected_model = model_map[model_type]
            
            # Train and make predictions for each parameter
            for param in df['parameter'].unique():
                st.write(f"### {param} Predictions")
                
                # Prepare data based on model type
                X, y, dates, feature_names, ts_df = prepare_data_for_prediction(df, param, selected_model)
                
                # Determine if we have enough data
                min_data_points = 48 if selected_model == "holt_winters" else 24
                has_enough_data = (
                    (selected_model == "rf" and X is not None and len(X) >= min_data_points) or
                    (selected_model != "rf" and ts_df is not None and len(ts_df) >= min_data_points)
                )
                
                if has_enough_data:
                    # Split the container
                    pred_cols = st.columns([3, 1])
                    
                    with pred_cols[0]:
                        # Train model based on selected type
                        model_results = train_prediction_model(X, y, feature_names, ts_df, param, selected_model)
                        
                        if model_results is not None:
                            # Get last date from data
                            if selected_model == "rf":
                                last_date = pd.to_datetime(dates[-1])
                            else:
                                last_date = ts_df.index[-1]
                            
                            # Predict future values
                            future_dates, predictions = predict_future(model_results, last_date, ts_df, days_to_predict)
                            
                            if future_dates is not None and len(future_dates) > 0:
                                # Create DataFrame for plotting
                                if selected_model == "rf":
                                    # For Random Forest, use the original dates and values
                                    hist_df = pd.DataFrame({
                                        'Date': dates,
                                        'Value': y,
                                        'Type': 'Historical'
                                    })
                                else:
                                    # For time series models, use the time series DataFrame
                                    hist_df = pd.DataFrame({
                                        'Date': ts_df.index,
                                        'Value': ts_df['value'].values,
                                        'Type': 'Historical'
                                    })
                                
                                # Create prediction DataFrame
                                pred_df = pd.DataFrame({
                                    'Date': future_dates,
                                    'Value': predictions,
                                    'Type': 'Predicted'
                                })
                                
                                # Combine historical and predicted data
                                plot_df = pd.concat([hist_df, pred_df])
                                
                                # Create daily average prediction for a cleaner visualization
                                if len(future_dates) > 24:  # If we have more than a day of predictions
                                    # Convert to DataFrame for easier manipulation
                                    pred_df_daily = pred_df.copy()
                                    pred_df_daily['Day'] = pred_df_daily['Date'].dt.date
                                    
                                    # Get daily averages
                                    daily_avg = pred_df_daily.groupby('Day')['Value'].mean().reset_index()
                                    daily_avg['Date'] = pd.to_datetime(daily_avg['Day'])
                                    daily_avg['Type'] = 'Daily Average (Predicted)'
                                    
                                    # Add to plot DataFrame
                                    plot_df = pd.concat([plot_df, daily_avg[['Date', 'Value', 'Type']]])
                                
                                # Plot
                                fig = px.line(
                                    plot_df,
                                    x='Date',
                                    y='Value',
                                    color='Type',
                                    title=f"{param} - Historical and Predicted Values using {model_type}",
                                    labels={'Date': 'Date & Time', 'Value': 'Value'}
                                )
                                
                                # Add threshold lines if applicable
                                if "Streamflow" in param:
                                    fig.add_hline(
                                        y=threshold_flow,
                                        line_dash="dash",
                                        line_color="green",
                                        annotation_text="Optimal Flow"
                                    )
                                elif "Gage height" in param:
                                    fig.add_hline(
                                        y=threshold_gauge_min,
                                        line_dash="dash",
                                        line_color="green",
                                        annotation_text="Min Optimal Height"
                                    )
                                    fig.add_hline(
                                        y=threshold_gauge_max,
                                        line_dash="dash",
                                        line_color="green",
                                        annotation_text="Max Optimal Height"
                                    )
                                elif "Turbidity" in param:
                                    fig.add_hline(
                                        y=threshold_turbidity,
                                        line_dash="dash",
                                        line_color="green",
                                        annotation_text="Optimal Turbidity Max"
                                    )
                                
                                # Format x-axis for better readability
                                fig.update_xaxes(
                                    tickformat="%m/%d\n%H:%M",
                                    tickangle=0,
                                    nticks=20
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Calculate when conditions will be optimal in the future
                                optimal_times = []
                                
                                for i, (date, value) in enumerate(zip(future_dates, predictions)):
                                    is_optimal = False
                                    
                                    if "Streamflow" in param and abs(value - threshold_flow) <= threshold_flow * 0.1:
                                        is_optimal = True
                                    elif "Gage height" in param and threshold_gauge_min <= value <= threshold_gauge_max:
                                        is_optimal = True
                                    elif "Turbidity" in param and value <= threshold_turbidity:
                                        is_optimal = True
                                    
                                    if is_optimal:
                                        optimal_times.append({
                                            'Date': date,
                                            'Value': value
                                        })
                                
                                if optimal_times:
                                    st.success(f"**Predicted Optimal Conditions for {param}:**")
                                    optimal_df = pd.DataFrame(optimal_times)
                                    optimal_df['Date'] = optimal_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
                                    st.dataframe(optimal_df)
                                else:
                                    st.warning(f"No optimal conditions predicted for {param} in the next {days_to_predict} days.")
                    
                    with pred_cols[1]:
                        # Display model metrics and details
                        if model_results is not None:
                            st.metric("Model MSE", f"{model_results['mse']:.4f}")
                            st.metric("Model R²", f"{model_results['r2']:.4f}")
                            
                            # Display model-specific information
                            if selected_model == "rf":
                                # Display feature importances for Random Forest
                                importances = model_results["model"].feature_importances_
                                
                                # Create DataFrame for importance visualization
                                importance_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': importances
                                }).sort_values('Importance', ascending=False)
                                
                                # Get top 10 features
                                top_features = importance_df.head(10)
                                
                                st.write("Top 10 Feature Importances:")
                                fig = px.bar(
                                    top_features,
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title="Feature Importance"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            elif selected_model == "prophet":
                                st.write("#### Prophet Model Components")
                                st.write("Prophet decomposes the time series into trend, weekly and daily seasonality components.")
                                
                                # Display top seasonality components if available
                                try:
                                    model = model_results["model"]
                                    components = ["trend", "weekly", "yearly", "daily"]
                                    available_components = []
                                    
                                    for component in components:
                                        if component in model.component_modes:
                                            available_components.append(component)
                                    
                                    if available_components:
                                        st.write(f"Components: {', '.join(available_components)}")
                                except:
                                    pass
                            
                            elif selected_model == "arima":
                                st.write("#### ARIMA Model")
                                model = model_results["model"]
                                
                                # Display the model order (p,d,q)
                                try:
                                    order = model.model.order
                                    st.write(f"Model Order (p,d,q): {order}")
                                    st.write("- p: Autoregressive order")
                                    st.write("- d: Differencing order")
                                    st.write("- q: Moving average order")
                                except:
                                    pass
                            
                            elif selected_model == "holt_winters":
                                st.write("#### Holt-Winters Model")
                                try:
                                    model = model_results["model"]
                                    params = model.params
                                    
                                    # Display alpha, beta, gamma values if available
                                    if hasattr(params, 'smoothing_level'):
                                        st.write(f"Alpha (level): {params.smoothing_level:.4f}")
                                    if hasattr(params, 'smoothing_trend'):  
                                        st.write(f"Beta (trend): {params.smoothing_trend:.4f}")
                                    if hasattr(params, 'smoothing_seasonal'):
                                        st.write(f"Gamma (seasonal): {params.smoothing_seasonal:.4f}")
                                except:
                                    pass
                else:
                    min_data_msg = "48 hours" if selected_model == "holt_winters" else "24 hours"
                    st.warning(f"Not enough data available for {param} predictions using {model_type}. Need at least {min_data_msg} of data.")
    else:
        st.error(f"No data available for site {selected_site_id} with the selected parameters and date range.")
else:
    st.warning("Please select a site or enter a site ID.")

# Add footer with information
st.markdown("---")
st.markdown("""
**About this app:**  
This application uses data from the USGS Water Data API to monitor and analyze water conditions.  
It compares current measurements against optimal thresholds and provides predictions based on historical data.
""")
