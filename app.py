import streamlit as st
import pandas as pd
import joblib

# Load saved objects
saved_objects = joblib.load("rf_with_scaler_encoder.pkl")
model = saved_objects["model"]
scaler = saved_objects["scaler"]
encoder = saved_objects["encoder"]

st.set_page_config(page_title="Facebook Engagement Classifier", layout="centered")

st.title("📊 Facebook Post Engagement Classifier")
st.write("Enter post details and the model will predict its engagement level.")

# Use the exact classes learned by the encoder
status_type = st.selectbox("📌 Post Type", encoder.classes_)

num_likes = st.number_input("👍 Likes", min_value=0, value=0)
num_loves = st.number_input("😍 Loves", min_value=0, value=0)
num_wows = st.number_input("😮 Wows", min_value=0, value=0)
num_hahas = st.number_input("😂 Hahas", min_value=0, value=0)
num_sads = st.number_input("😢 Sads", min_value=0, value=0)
num_angrys = st.number_input("😡 Angrys", min_value=0, value=0)

num_comments = st.number_input("💬 Comments", min_value=0, value=0)
num_shares = st.number_input("🔁 Shares", min_value=0, value=0)

# Auto calculate reactions
num_reactions = num_likes + num_loves + num_wows + num_hahas + num_sads + num_angrys
st.info(f"📌 Total Reactions (calculated): {num_reactions}")

if st.button("🔮 Predict Engagement Level"):
    # Create dataframe
    input_data = pd.DataFrame([[
        status_type, num_reactions, num_comments, num_shares,
        num_likes, num_loves, num_wows, num_hahas,
        num_sads, num_angrys
    ]], columns=[
        "status_type", "num_reactions", "num_comments", "num_shares",
        "num_likes", "num_loves", "num_wows", "num_hahas",
        "num_sads", "num_angrys"
    ])

    # Encode status_type using the loaded encoder
    input_data["status_type"] = encoder.transform(input_data["status_type"])

    # Scale all inputs
    input_scaled = scaler.transform(input_data)

    # Predict cluster
    prediction = model.predict(input_scaled)[0]

    # Engagement levels (0,1,2 only)
    category_names = {
        0: "💤 Low Engagement",
        1: "👍 Medium Engagement",
        2: "🔥 High Engagement"
    }
    predicted_name = category_names.get(prediction, f"Cluster {prediction}")

    st.success(f"📌 Predicted Engagement Level: {predicted_name}")