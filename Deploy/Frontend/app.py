import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import pickle
import tensorflow as tf

# load pipe
pipe = pickle.load(open("preprocess.pkl", "rb"))

#load model
h5_model = tf.keras.models.load_model("model_f.h5", compile=False)

# Membuat title
st.title('Churn Prediction')


# Membuat sub header
st.subheader('Customer Information')
# Membuat Form
with st.form(key='form_parameters'):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'], index=1)
        senior = st.selectbox('Senior Citizen', ['1', '0'], index=1)
        partner = st.selectbox('Partner', ['Yes', 'No'], index=1)
        depends = st.selectbox('Dependents', ['Yes', 'No'], index=1)
        tenure = st.number_input('Tenure', min_value=1, max_value=72)
        phone = st.selectbox('Phone Service', ['Yes', 'No'], index=1)
        multiple_lines = st.selectbox(
            'Multiple Lines', ['Yes', 'No', 'No phone service'], index=1)

    with col2:
        internet_service = st.selectbox(
            'Internet Service', ['Fiber optic', 'DSL', 'No'], index=2)
        security = st.selectbox('Internet Security', [
                                'Yes', 'No', 'No internet service'], index=1)
        backup = st.selectbox(
            'Online Backup', ['Yes', 'No', 'No internet service'], index=1)
        protection = st.selectbox('Device Protection', [
                                  'Yes', 'No', 'No internet service'], index=1)
        support = st.selectbox(
            'Tech Support', ['Yes', 'No', 'No internet service'], index=1)
        tv = st.selectbox(
            'Streaming TV', ['Yes', 'No', 'No internet service'], index=1)
        movies = st.selectbox('Streaming Movies', [
                              'Yes', 'No', 'No internet service'], index=1)

    st.subheader('Payment Information')
    contract = st.selectbox(
        'Contract', ['Month-to-month', 'Two year', 'One year'], index=0)
    billing = st.selectbox('Paperless Billing', ['Yes', 'No'], index=1)
    payment = st.selectbox('Payment Method', [
                           'Mailed check', 'Electronic check', 'Bank transfer (automatic)', 'Credit card (automatic)'], index=0)
    monthly = st.number_input(
        'Monthly Charges', min_value=18, max_value=150)
    total = st.number_input('Total Charges', min_value=18, max_value=9000)
    st.markdown('---')

    submitted = st.form_submit_button('Predict')

# Membuat data inference
data_inf = {
    'gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': depends,
    'tenure': tenure,
    'PhoneService': phone,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': security,
    'OnlineBackup': backup,
    'DeviceProtection': protection,
    'TechSupport': support,
    'StreamingTV': tv,
    'StreamingMovies': movies,
    'Contract': contract,
    'PaperlessBilling': billing,
    'PaymentMethod': payment,
    'MonthlyCharges': monthly,
    'TotalCharges': total
}

data_inf = pd.DataFrame([data_inf])
st.dataframe(data_inf)

# preprocessing
new_data = pipe.transform(data_inf)
new_data = new_data.astype(float)
#new_data = np.asarray(new_data)
#new_data = new_data.tolist()

if submitted:
    # Predict using Linear Regression
    y_pred_inf = h5_model.predict(new_data)

    if y_pred_inf >= 0.5:
        st.subheader('Churn')
    else:
        st.subheader('Not Churn')



# input ke model
#input_data_json = json.dumps({
#    "signature_name": "serving_default",
#    "instances": new_data
#})

# inference
#URL = "http://churn-yosi.herokuapp.com/v1/models/model_functional:predict"
#r = requests.post(URL, data=input_data_json)
#  
#    if r.status_code == 200:
#        res = r.json()
#        if res['predictions'][0][0] >= 0.5:
#            st.write('Churn')
#        else:
#            st.write('Not Churn')
#    else:
#        st.write('Error')

