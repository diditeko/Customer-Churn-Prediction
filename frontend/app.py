import pandas as pd
import pickle
import streamlit as st
import requests
import json


page_bg_img = """
<style>
[data-testid="stAppViewContainer"] > .main {
background-image: url("https://wallpapercave.com/wp/wp7747847.jpg");
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
}

[data-testid="stHeader"] {
background: rgba(0,0,0,0);
}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# load pipe
pipe = pickle.load(open("preprocess_pipe.pkl", "rb"))

# widget input
st.title("Aplikasi Predict Churn")
with st.form(key='form_parameter'):
    col1, col2 = st.columns(2)

    with col1:
        Partner = st.selectbox("Partner", ('Yes','No'), index=1)
        Dependents = st.selectbox("Dependents", ('Yes', 'No'), index=1)
        SeniorCitizen = st.selectbox("SeniorCitizen", ('0', '1'), index=1)
        tenure = st.number_input('tenure', min_value=0, max_value=250, value=2, help = 'bulan berlangganan')
        InternetService = st.selectbox("InternetService", ('Dsl','Fiber optic', 'No'), index=1)
        OnlineSecurity = st.selectbox("OnlineSecurity", ('Yes','No internet service', 'No'), index=1)
        OnlineBackup = st.selectbox("OnlineBackup", ('Yes','No internet service', 'No'), index=1)
        DeviceProtection = st.selectbox("DeviceProtection", ('Yes','No internet service', 'No'), index=1)
        st.markdown('---')

    with col2:
        TechSupport = st.selectbox("TechSupport", ('Yes','No internet service', 'No'), index=1)
        StreamingTV = st.selectbox("StreamingTV", ('Yes','No internet service', 'No'), index=1)
        StreamingMovies = st.selectbox("StreamingMovies", ('Yes','No internet service', 'No'), index=1)
        Contract = st.selectbox("Contract", ('Month-to-month','One year', 'Two year'), index=1)
        PaperlessBilling = st.selectbox("PaperlessBilling", ('Yes', 'No'), index=1)
        PaymentMethod = st.selectbox("PaymentMethod", ('Mailed check', 'Electronic check', 'Bank transfer (automatic)', 'Credit card (automatic)'), index=1)
        MonthlyCharges = st.number_input('MonthlyCharges', min_value=0, max_value=1000000, value=10000, help = 'bayar perbulan')
        TotalCharges = st.number_input('TotalCharges', min_value=0, max_value=1000000, value=10000, help = 'TotalCharges')
        st.markdown('---')

        submitted= st.form_submit_button('Predict')



# input to dataframe
new_data = {'Partner': Partner,
         'Dependents': Dependents,
         'SeniorCitizen' : SeniorCitizen,
         'tenure' : tenure,
         'InternetService' : InternetService,
         'OnlineSecurity' : OnlineSecurity,
         'OnlineBackup' : OnlineBackup,
         'DeviceProtection' : DeviceProtection,
         'TechSupport' : TechSupport,
         'StreamingTV' : StreamingTV,
         'StreamingMovies' : StreamingMovies,
         'Contract' : Contract,
         'PaperlessBilling' : PaperlessBilling,
         'PaymentMethod' : PaymentMethod,
         'MonthlyCharges' : MonthlyCharges,
         'TotalCharges' : TotalCharges}
new_data = pd.DataFrame([new_data])

# preprocessing
new_data = pipe.transform(new_data)
new_data = new_data.astype(float)
new_data = new_data.tolist()

# input ke model
input_data_json = json.dumps({
    "signature_name": "serving_default",
    "instances": new_data
})

# inference
URL ="http://backend-churn-app.herokuapp.com/v1/models/churn_model:predict"
r = requests.post(URL, data=input_data_json)

if submitted:
    if r.status_code == 200:
         res = r.json()
         if res['predictions'][0][0] >= 0.5:
            st.write('## CHURN')
         else:
            st.write('## TIDAK CHURN')
    else:
        st.write('## ERROR')