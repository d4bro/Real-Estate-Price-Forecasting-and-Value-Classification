# Real Estate Price Forecasting and Value Classification

This project is a web application designed to analyze and predict real estate prices in California. It utilizes machine learning models to forecast property prices and classify properties into value segments.
The application is built using Streamlit,providing an interactive interface for users to explore the data and model results.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Technologies](#technologies)
- [Data](#data)
- [Results](#results)
## Installation

To run this application, you need to have Python installed along with some necessary libraries. You can use the following commands to set up your environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/real-estate-forecasting.git

# Navigate into the project directory
cd real-estate-forecasting

# Install required packages
pip install -r requirements.txt
```

## Usage

To start the Streamlit application, run the following command in your terminal within the project directory:

```bash
streamlit run app.py
```

This will launch the web application in your default web browser.

## Features

- **Interactive Data Filtering**: Users can filter data based on geographic location and property attributes such as age, latitude, and proximity to the ocean.
- **Regression Models**: Includes options for Random Forest and Gradient Boosting regression to predict property prices.
- **Classification Model**: Uses Random Forest to classify properties into different value segments.
- **Visualizations**: Provides a variety of plots including scatter maps, heatmaps, feature importance bar charts, and more to visualize data and model outputs.
- **Performance Metrics**: Displays Mean Absolute Error (MAE) and R-squared (R²) scores for regression models, and accuracy and confusion matrix for the classification model.

## Technologies

- **Python**: Programming language used.
- **Streamlit**: Framework for creating the web application front end.
- **Scikit-learn**: Library for machine learning models and evaluation metrics.
- **Pandas and NumPy**: For data manipulation and numerical computations.
- **Matplotlib, Seaborn, and Plotly**: For data visualization.

## Data

The dataset used in this project is the California housing dataset, which includes information on various housing properties in the state.

## Results

- **Regression Model Performance**: The model achieved an R² score of X.XX, indicating a good fit to the actual data.
- **Classification Model Performance**: The classification model achieved an accuracy of XX%, demonstrating its effectiveness in categorizing properties based on value.
