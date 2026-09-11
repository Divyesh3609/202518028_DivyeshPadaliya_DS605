import streamlit as st
import pandas as pd
import joblib

model = joblib.load("airbnb_price_model.pkl")

st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏠"
)

st.title("🏠 Airbnb Price Predictor")
st.write("Enter the Airbnb listing details to estimate the nightly price.")

neighbourhood_group = st.selectbox(
    "Neighbourhood Group",
    ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
)

neighbourhood = st.text_input(
    "Neighbourhood",
    "Midtown"
)

latitude = st.number_input(
    "Latitude",
    value=40.7549,
    format="%.4f"
)

longitude = st.number_input(
    "Longitude",
    value=-73.9840,
    format="%.4f"
)

room_type = st.selectbox(
    "Room Type",
    ["Entire home/apt", "Private room", "Shared room"]
)

minimum_nights = st.number_input(
    "Minimum Nights",
    min_value=1,
    value=3
)

number_of_reviews = st.number_input(
    "Number of Reviews",
    min_value=0,
    value=50
)

reviews_per_month = st.number_input(
    "Reviews Per Month",
    min_value=0.0,
    value=2.5
)

calculated_host_listings_count = st.number_input(
    "Host Listings Count",
    min_value=1,
    value=2
)

availability_365 = st.number_input(
    "Availability (365 days)",
    min_value=0,
    max_value=365,
    value=200
)

if st.button("💰 Predict Price"):

    reviews_per_availability = (
        number_of_reviews / (availability_365 + 1)
    )

    host_listing_density = (
        calculated_host_listings_count /
        (availability_365 + 1)
    )

    new_input = pd.DataFrame({
        "neighbourhood_group": [neighbourhood_group],
        "neighbourhood": [neighbourhood],
        "latitude": [latitude],
        "longitude": [longitude],
        "room_type": [room_type],
        "minimum_nights": [minimum_nights],
        "number_of_reviews": [number_of_reviews],
        "reviews_per_month": [reviews_per_month],
        "calculated_host_listings_count": [
            calculated_host_listings_count
        ],
        "availability_365": [availability_365],
        "reviews_per_availability": [
            reviews_per_availability
        ],
        "host_listing_density": [
            host_listing_density
        ]
    })

    prediction = model.predict(new_input)[0]

    prediction = max(0, prediction)

    st.success(
        f"Estimated Nightly Price: ${prediction:.2f}"
    )