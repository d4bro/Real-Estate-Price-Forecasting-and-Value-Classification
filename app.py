import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Description
st.title('Data Analysis Project: Real Estate Price Forecasting and Value Classification')
st.markdown("""
This project aims to forecast property prices in California using regression models and classify properties into value segments.
""")

# Load data
df = pd.read_csv('housing.csv') 
numeric_columns = df.drop(columns=['ocean_proximity']).columns
numeric_data = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
# Drop rows with missing values
df_clean = df.dropna()

# Basic EDA
st.header('Exploratory Data Analysis')
st.write(df.describe())

# User Inputs
with st.sidebar:
    st.header("Filters")
    ocean_proximity = st.multiselect(
        'ocean_proximity',
        sorted(df["ocean_proximity"].dropna().unique())
    )
    x_axis = st.selectbox("X axis", numeric_columns, index=0)
    y_axis = st.selectbox("Y axis", numeric_columns, index=3)
    color_by = st.selectbox("color by", numeric_columns)
    age_range = st.slider(
        'housing_median_age',
        min_value=1, max_value=52, value=(10, 35)
    )
    latitude_range = st.slider(
        'latitude',
        min_value=32.5, max_value=42.0, value=(34.0, 38.0)
    )
    longitude_range = st.slider(
        'longitude',
        min_value=-124.35, max_value=-114.31, value=(-120.0, -118.0)
    )
    median_house_value = st.slider(
        'median_house_value',
        min_value=0, max_value=5_000_000, value=(100_000, 1_000_000)
    )
    model_type = st.selectbox(
        "Choose Model",
        ["Regression: Random Forest", "Gradient Boosting", "Classification: Random Forest"]
    )
    n_estimators = st.number_input("Number of Trees (50-500)", min_value=50, max_value=500, value=100, step=25)
    max_depth = st.slider("Maximum Depth", 1, 20, 5)
    
# Data Filtering
if age_range:
    df = df[(df['housing_median_age'] >= age_range[0]) & (df['housing_median_age'] <= age_range[1])]
if ocean_proximity:
    df = df[df["ocean_proximity"].isin(ocean_proximity)]
if latitude_range:
    df = df[(df['latitude'] >= latitude_range[0]) & (df['latitude'] <= latitude_range[1])]
if longitude_range:
    df = df[(df['longitude'] >= longitude_range[0]) & (df['longitude'] <= longitude_range[1])]
if x_axis and y_axis:
    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, trendline="ols")
    st.plotly_chart(fig)

# Visualize Data
fig = px.scatter_mapbox(
    df,
    lat="latitude",
    lon="longitude",
    color="median_house_value",
    zoom=4,
    mapbox_style="open-street-map",
    hover_name="ocean_proximity",
    title="Property Prices in California"
)
st.plotly_chart(fig, use_container_width=True)

# Plot the heatmap
st.subheader('Correlation Matrix Analysis for Landscape Properties')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Matrix')

st.pyplot(fig)

# Data Preparation
X = df_clean[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]
y = df_clean['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Selection and Training
if model_type in ["Regression: Random Forest", "Gradient Boosting"]:
    if model_type == "Regression: Random Forest":
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    st.markdown("""
### Understanding the Metrics
- **Mean Absolute Error (MAE):** Reflects the average magnitude of errors in a set of predictions, without considering their direction. It's the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight. Lower values indicate a better model.
- **R-squared (R²):** Represents the proportion of variance for a dependent variable that's explained by an independent variable or variables in a regression model. The value of R² is between 0 and 1. In general, the higher the R², the better the model fits your data.
""")
    # Results Display
    st.subheader(f"Results: {model_type}")
    col1, col2 = st.columns(2)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("R²", f"{r2:.2f}")

    # Feature Importance
    st.subheader('Feature Importance')
    feature_importances = model.feature_importances_
    features = X.columns
    fig, ax = plt.subplots()
    ax.barh(features, feature_importances, color='skyblue')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

    # Actual vs. Predicted
    st.subheader('Actual vs. Predicted Values')
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=predictions, alpha=0.4, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs. Predicted')
    st.pyplot(fig)

    # Residual Plot
    st.subheader('Residual Plot')
    residuals = y_test - predictions
    fig, ax = plt.subplots()
    sns.scatterplot(x=predictions, y=residuals, alpha=0.4, ax=ax)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    st.pyplot(fig)

# Classification Example
if model_type == "Classification: Random Forest":
    y_class = pd.cut(y, bins=[0, 150000, 300000, 500000, np.inf], labels=[0, 1, 2, 3])  # Example binning
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train_scaled, y_train_c)
    y_pred_c = clf.predict(X_test_scaled)

    st.subheader("Classification Results")
    accuracy = accuracy_score(y_test_c, y_pred_c)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.markdown("""
### Understanding the Confusion Matrix
A confusion matrix is used to evaluate the accuracy of a classification. It shows the number of true positive, true negative, false positive, and false negative predictions, allowing you to gain insights into where the model is making errors.

- **True Positive (TP):** Correctly predicted positive cases
- **True Negative (TN):** Correctly predicted negative cases
- **False Positive (FP):** Incorrectly predicted as positive
- **False Negative (FN):** Incorrectly predicted as negative

From these, we derive metrics such as accuracy, precision, recall, and F1 score to better understand performance.
""")
    # Confusion Matrix
    cm = confusion_matrix(y_test_c, y_pred_c)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

# Additional Visualizations
st.subheader('Distribution of House Prices')
fig, ax = plt.subplots()
sns.histplot(df['median_house_value'], bins=50, kde=True, ax=ax)
ax.set_title('Median House Value Distribution')
st.pyplot(fig)
