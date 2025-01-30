import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Wczytaj dane
df = pd.read_csv('housing.csv')  # Upewnij się, że masz poprawną ścieżkę
numeric_columns = df.drop(columns=['ocean_proximity']).columns

# Drop rows with missing values
df_clean = df.dropna()

# Podstawowa EDA
st.title('Analiza Eksploracyjna Danych')
st.write(df.describe())

with st.sidebar:
    ocean_proximity = st.multiselect(
        'ocean_proximity',
        sorted(df["ocean_proximity"].dropna().unique())
    )
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
        "Wybierz model",
        ["Regresja: Random Forest", "Gradient Boosting"]
    )
    n_estimators = st.sidebar.number_input("Liczba drzew", 50, 500, 100)
    max_depth = st.sidebar.slider("Maksymalna głębokość", 1, 20, 5)

# Filtracja danych
if age_range:
    df = df[(df['housing_median_age'] >= age_range[0]) & (df['housing_median_age'] <= age_range[1])]
if ocean_proximity:
    df = df[df["ocean_proximity"].isin(ocean_proximity)]
if latitude_range:
    df = df[(df['latitude'] >= latitude_range[0]) & (df['latitude'] <= latitude_range[1])]
if longitude_range:
    df = df[(df['longitude'] >= longitude_range[0]) & (df['longitude'] <= longitude_range[1])]

fig = px.scatter_mapbox(
    df,
    lat="latitude",
    lon="longitude",
    color="median_house_value",
    zoom=4,
    mapbox_style="open-street-map",
    hover_name="ocean_proximity",
    title="Ceny domów w Kalifornii"
)
st.plotly_chart(fig, use_container_width=True)

# Przygotowanie danych
X = df_clean[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]
y = df_clean['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Skalowanie cech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Wybór modelu
if model_type == "Regresja: Random Forest":
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
elif model_type == "Gradient Boosting":
    model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

# Wyświetlanie wyników
st.subheader(f"Wyniki: {model_type}")
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
ax.set_xlabel('Feature Importance')
ax.set_title('Feature Importance')
st.pyplot(fig)

# Actual vs. Predicted Values
st.subheader('Actual vs. Predicted Values')
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=predictions, alpha=0.4, ax=ax)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
ax.set_title('Actual vs. Predicted')
st.pyplot(fig)

# Residual Plot
st.subheader('Residual Plot')
residuals = y_test - predictions
fig, ax = plt.subplots()
sns.scatterplot(x=predictions, y=residuals, alpha=0.4, ax=ax)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot')
st.pyplot(fig)

# Histogram ceny domu
st.subheader('Median house value')
fig, ax = plt.subplots()
sns.histplot(df['median_house_value'], bins=50, kde=True, ax=ax)
ax.set_title('Median house value')
st.pyplot(fig)

# Mapa cen domów
st.subheader('Median house value on the map')
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['longitude'], df['latitude'], c=df['median_house_value'], cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='MedHouseVal')
ax.set_title('Median house value')
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
st.pyplot(fig)

# Wykres mapowy z Cartopy
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND.with_scale('50m'))
ax.add_feature(cfeature.OCEAN.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))
ax.add_feature(cfeature.BORDERS.with_scale('50m'))
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.set_extent([-125, -114, 32, 42], crs=ccrs.PlateCarree())
scatter = ax.scatter(
    df['longitude'],
    df['latitude'],
    c=df['median_house_value'],
    cmap='viridis',
    s=10,
    alpha=0.6,
    transform=ccrs.PlateCarree()
)
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=1,
    color='gray',
    alpha=0.5,
    linestyle='--'
)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}
cities = {
    'Los Angeles': (-118.2437, 34.0522),
    'San Francisco': (-122.4194, 37.7749),
    'San Diego': (-117.1611, 32.7157),
    'Sacramento': (-121.4944, 38.5816)
}
for city, (lon, lat) in cities.items():
    ax.plot(lon, lat, 'ro', markersize=5, transform=ccrs.PlateCarree())
    ax.text(
        lon + 0.1,
        lat + 0.1,
        city,
        fontsize=12,
        color='black',
        weight='bold',
        transform=ccrs.PlateCarree()
    )
plt.colorbar(scatter, label='Mediana ceny domu (w $100 000)', ax=ax)
ax.set_title('Median house price in California on the map', fontsize=16)
st.pyplot(fig)