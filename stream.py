import pickle
import streamlit as st
import numpy as np
import pandas as pd

# ===================== LOAD MODEL ==========================
model = pickle.load(open('morningdataset_model.sav', 'rb'))

# ====================== TITLE WEB ==========================
st.title('Prediksi Productivity Score – Morning Routine')

st.write("Masukkan data rutinitas pagi kamu untuk memprediksi skor produktivitas (1–10).")

# ===================== INPUT FORM ==========================
col1, col2 = st.columns(2)

with col1:
    SleepHours = st.number_input('Jam tidur (Sleep Hours)', min_value=0.0, max_value=24.0, step=0.1)
    WaterIntake = st.number_input('Jumlah minum air (Liter)', min_value=0.0, max_value=10.0, step=0.1)
    BreakfastCalories = st.number_input('Kalori sarapan', min_value=0, max_value=2000)

with col2:
    ExerciseMinutes = st.number_input('Menit berolahraga', min_value=0, max_value=300)
    ScreenTime = st.number_input('Screen Time pagi (menit)', min_value=0, max_value=600)
    Mood = st.selectbox("Mood Pagi", ["Bad", "Neutral", "Good"])

# ===================== KONVERSI KE DATAFRAME ==========================
input_dict = {
    'SleepHours': SleepHours,
    'WaterIntake': WaterIntake,
    'BreakfastCalories': BreakfastCalories,
    'ExerciseMinutes': ExerciseMinutes,
    'ScreenTime': ScreenTime,
    'Mood': Mood
}

input_df = pd.DataFrame([input_dict])

# ===================== PREDIKSI ==========================
if st.button("Prediksi Productivity Score"):
    try:
        pred = model.predict(input_df)[0]
        st.success(f"Prediksi Productivity Score kamu adalah: **{pred:.2f} / 10**")
    except Exception as e:
        st.error("Terjadi error saat memproses prediksi.")
        st.error(str(e))
