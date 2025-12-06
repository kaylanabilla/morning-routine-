import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ============================
# LOAD MODEL
# ============================
model = pickle.load(open("morningdataset_model.sav", "rb"))

st.title("Morning Routine Productivity Predictor")
st.write("Masukkan rutinitas pagi kamu untuk memprediksi Productivity Score (1–10).")


# ============================
# INPUT FORM
# ============================
st.subheader("Input Rutinitas Pagi")

Sleep = st.number_input("Sleep Duration (hrs)", min_value=0.0, max_value=24.0, step=0.1)
Meditation = st.number_input("Meditation (mins)", min_value=0, max_value=120)
Exercise = st.number_input("Exercise (mins)", min_value=0, max_value=300)

Breakfast = st.selectbox("Breakfast Type", ["Toast", "Cereal", "Oatmeal", "Fruit", "None"])
Journal = st.selectbox("Journaling (Y/N)", ["Y", "N"])
Mood = st.selectbox("Mood", ["Happy", "Neutral", "Sad"])
WorkStart = st.selectbox("Work Start Time", ["6:00 AM", "7:00 AM", "8:00 AM", "9:00 AM"])
Notes = st.selectbox("Notes", ["Normal", "Stressed", "Well-rested"])


# ============================
# KONVERSI KE DATAFRAME
# ============================

# Input mentah
raw_data = {
    "Sleep Duration (hrs)": Sleep,
    "Meditation (mins)": Meditation,
    "Exercise (mins)": Exercise,
    "Breakfast Type": Breakfast,
    "Journaling (Y/N)": Journal,
    "Mood": Mood,
    "Work Start Time": WorkStart,
    "Notes": Notes
}

raw_df = pd.DataFrame([raw_data])

# ============================
# PREPROCESSING SAMA DGN TRAINING
# ============================

# One-hot encode
df_encoded = pd.get_dummies(raw_df, drop_first=True)

# Buat kolom yang hilang agar match dengan model
model_features = model.feature_names_in_

for col in model_features:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

# Pastikan urutan kolom sama
df_encoded = df_encoded[model_features]


# ============================
# PREDIKSI
# ============================

if st.button("Prediksi Productivity Score"):
    pred = model.predict(df_encoded)[0]
    st.success(f"Productivity Score kamu: **{pred:.2f} / 10**")
