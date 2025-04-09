import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Customer Churn Predictor App",
    page_icon="🎯",
    layout="centered",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/rathindra-narayan-hait-a0ba9015a/',
        'Report a bug': "https://rathindra.onrender.com/",
        'About': "### Customer churn predictor application by ADFC Bank"
    }
)

def log_transform(X):
    return np.log1p(X)

def square_transform(X):
    return X ** 2

log_col = ["age"]
square_col = ["creditscore"]


pipeline=joblib.load('Customer churn prediction app/model_pipeline.joblib')
@st.dialog("Prediction")
def your_dialog_function():
    with st.container(border=True):

        colu1, colu2, colu3=st.columns([0.06, 2, 0.06])
        with colu2:

            st.write("######")
            st.write(dataf)
            prediction = pipeline.predict(dataf)
            proba = pipeline.predict_proba(dataf)[0][1]

            risk_level = "High Risk" if proba >= 0.7 else "Medium Risk" if proba >= 0.5 and proba < 0.7 else "Low Risk"

            if proba>=0.7:
                st.error(f"🚨This customer is in **{risk_level}** category")
                st.markdown(f"This customer has a **{proba:.1%}** probability of leaving the bank.")
                st.write("💔This customer might be saying goodbye soon! Immediate action is recommended.")
            elif proba>=0.5 and proba<0.7:
                st.warning(f"📉This customer is in **{risk_level}** category")
                st.markdown(f"Customer has a **{proba:.1%}** probability of leaving the bank.")
                st.write("🛠️Some action is recommended.")
            else:
                st.success(f"📈This customer is in **{risk_level}** category")
                st.markdown(f"Customer has a **{proba:.1%}** probability of leaving the bank.")
                st.write("😊This customer seems happy with us.")
            st.write("######")

            if st.button("Back", type="secondary", use_container_width=True):
                st.rerun()
        st.write("######")

st.title("🎯 Customer Churn Predictor")

with st.container():
    st.write("Please fillup the customer details below")

with (st.container(border=True)):
    st.write("######")

    col1, col2, col3, col4, col5 = st.columns([0.4, 2, 0.1, 2, 0.4])
    with col2:
        age = st.number_input("Age", min_value=0, max_value=130, value=None, step=1, placeholder="Type customers age...")
        point_earned= st.number_input("Reward points", min_value=0.0, max_value=5000.0, value=None, placeholder="Type customers reward points...")
        tenure = st.number_input("Tenure", min_value=0.0, max_value=50.0, value=None, step=0.01, placeholder="Type customers tenure...")
        numofproducts = st.number_input("Number of products subscribed", min_value=0, max_value=30, value=None, step=1, placeholder="Type number of product subcription...")
        gender = st.pills("Sex", options=["Male", "Female"])
        geography= st.pills("Country", options=["France", "Spain", "Germany"])
        member = st.pills("Is she/he is an active member?", options=["Yes", "No"])
    with col4:
        estimatedsalary = st.number_input("Salary", min_value=0.0, max_value=100000000.0, value=None, placeholder="Type customers salary...")
        balance = st.number_input("Balance", min_value=0.0, max_value=100000000.0, value=None, placeholder="Type customers account balance...")
        satisfaction_score = st.number_input("Satisfaction score", min_value=0.0, max_value=5000.0, value=None, step=1.0, placeholder="Type customers satisfaction score...")
        creditscore = st.number_input("Credit score", min_value=0.0, max_value=10000.0, value=None, placeholder="Type customers credit score...")
        crcard = st.pills("Has credit card?", options=["Yes", "No"], key="credit_card_pills")
        card_type = st.pills("Card type", options=["DIAMOND", "GOLD", "SILVER", "PLATINUM"])

        st.write("######")

        coll1, coll2 =st.columns([1,3])
        with coll2:

            button_disabled = not (age and point_earned and tenure and numofproducts and gender and geography and member and estimatedsalary and
                                   balance and satisfaction_score and creditscore and crcard and card_type)

            if st.button("Predict", icon=":material/online_prediction:", type="secondary", use_container_width=True, disabled=button_disabled):
                isactivemember = 1 if member == 'Yes' else 0
                hascrcard= 1 if crcard == 'Yes' else 0

                dataf=pd.DataFrame({"creditscore": [creditscore], "geography": [geography], "gender": [gender], "age": [age], "tenure": [tenure],
                                "balance": [balance], "numofproducts": [numofproducts], "hascrcard": [hascrcard], "isactivemember": [isactivemember],
                                "estimatedsalary": [estimatedsalary], "satisfaction_score": [satisfaction_score], "card_type": [card_type],
                                "point_earned": [point_earned]})

                prediction = pipeline.predict(dataf)
                proba = pipeline.predict_proba(dataf)[0][1]

                your_dialog_function()

st.caption("*This app predicts customer churn using a SMOTE-enhanced XGBoost model")

with st.container():

    st.write("----")
    colum1, colum2, colum3=st.columns([0.1,4.5,0.1])
    with colum2:

        image_path = "Customer churn prediction app/pipeline.png"
        st.image(image_path, caption="Machine Learning Pipeline")
