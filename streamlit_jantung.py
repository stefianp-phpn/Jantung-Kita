import streamlit as st
import pandas as pd
import pickle

# Muat model dan scaler
try:
    model = pickle.load(open('penyakit_jantung.sav', 'rb'))
    #scaler = pickle.load(open('skaler.pkl', 'rb'))  # Konsisten gunakan 'scaler'
    #st.write("Model dan scaler berhasil dimuat.")
except Exception as e:
    st.write(f"Kesalahan saat memuat model atau scaler: {e}")

def main():
    st.title('Heart Disease Prediction')
    age = st.number_input('Age', 28 , 66)
    sex_options = ['Male', 'Female']
    sex = st.selectbox('Sex', sex_options)
    sex_num = 1 if sex == 'Male' else 0 
    cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
    cp = st.selectbox('Chest Pain Type', cp_options)
    cp_num = cp_options.index(cp)
    trestbps = st.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.slider('Cholesterol', 100, 600, 250)
    fbs_options = ['False', 'True']
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', fbs_options)
    fbs_num = fbs_options.index(fbs)
    restecg_options = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy']
    restecg = st.selectbox('Resting Electrocardiographic Results', restecg_options)
    restecg_num = restecg_options.index(restecg)
    thalach = st.slider('Maximum Heart Rate Achieved', 70, 220, 150)
    exang_options = ['No', 'Yes']
    exang = st.selectbox('Exercise Induced Angina', exang_options)
    exang_num = exang_options.index(exang)
    oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)
    slope_options = ['Upsloping', 'Flat', 'Downsloping']
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', slope_options)
    slope_num = slope_options.index(slope)
    ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 1)
    thal_options = ['Tidak diketahui','Normal', 'Fixed Defect', 'Reversible Defect']
    thal = st.selectbox('Thalassemia', thal_options)
    thal_num = thal_options.index(thal)
    
    mean_std_values = pickle.load(open('mean_std_values.pkl', 'rb'))
    
    if st.button('Predict'):
         user_input = pd.DataFrame(data={
            'age': [age],
            'sex': [sex_num],  
            'cp': [cp_num],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs_num],
            'restecg': [restecg_num],
            'thalach': [thalach],
            'exang': [exang_num],
            'oldpeak': [oldpeak],
            'slope': [slope_num],
            'ca': [ca],
            'thal': [thal_num]
        })
        
        # Apply saved transformation to new data
        user_input = (user_input - mean_std_values['mean']) / mean_std_values['std']
     
        # Terapkan transformasi scaler pada input data
        #try:
            #user_input_scaled = scaler.transform(user_input)
            #st.write("Data input berhasil distandarisasi.")
        #except AttributeError as e:
            #st.write(f"Kesalahan pada scaler: {e}")
            #return
        #except Exception as e:
            #st.write(f"Kesalahan tidak terduga: {e}")
            #return
     
        # Prediksi menggunakan model
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)

        if prediction[0] == 1:
            bg_color = 'red'
            prediction_result = 'Positive'
        else:
            bg_color = 'green'
            prediction_result = 'Negative'
     
        confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]

        st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Hasil Prediksi : {prediction_result}<br>Confidence: {((confidence*10000)//1)/100}%</p>", unsafe_allow_html=True)
        if prediction_result == 'Positive':
            st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Jantung Anda dalam bahaya<br>Segera lakukan pemeriksaan lebih lanjut</p>")

if __name__ == '__main__':
    main()
