
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import sklearn
import imblearn


Model = joblib.load("Model.pkl")
Inputs = joblib.load("Input.pkl")

def prediction(Gender,Married,Dependents,Education,Self_Employed,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,Log_Total_Income,log_Loan_Monthly_Paid,log_Income_After_Loan):
    df = pd.DataFrame(columns=Inputs)
    df.at[0,"Gender"] = Gender
    df.at[0,"Married"] = Married
    df.at[0,"Dependents"] = Dependents
    df.at[0,"Education"] = Education
    df.at[0,"Self_Employed"] = Self_Employed
    df.at[0,"LoanAmount"] = LoanAmount
    df.at[0,"Loan_Amount_Term"] = Loan_Amount_Term
    df.at[0,"Credit_History"] = Credit_History
    df.at[0,"Property_Area"] = Property_Area
    df.at[0,"Log_Total_Income"] = Log_Total_Income
    df.at[0,"log_Loan_Monthly_Paid"] = log_Loan_Monthly_Paid
    df.at[0,"log_Income_After_Loan"] = log_Income_After_Loan
    result = Model.predict(df)[0]
    return result

def Main():
    st.title("Loan Prediction")
    Gender = st.selectbox("Gender",['Male' , 'Female'])
    Married = st.selectbox("Married",['No' , 'Yes'])
    Dependents = st.selectbox("Dependents",[ '0' ,  '1' ,  '2' ,  '3+'])
    Education = st.selectbox("Education",['Graduate', 'Not Graduate'])
    Self_Employed = st.selectbox("Self employed",['No' , 'Yes'])
    LoanAmount = st.slider("Loan amount in Thousands" , min_value = 9.0 , max_value = 700.0 , step = 1.0 , value = 10.0)
    Loan_Amount_Term = st.slider("Loan amount term" , min_value = 12.0 , max_value = 480.0 , step = 1.0 , value = 10.0)
    Credit_History = st.selectbox("Credit history" , [1 , 0])
    Property_Area = st.selectbox("Property area", ['Urban' , 'Rural' , 'Semi urban'])
    ApplicantIncome = st.slider("Applicant income", min_value = 0.0 , max_value =  81000.0 , step = 5.0,value = 500.0)
    CoapplicantIncome = st.slider("Coapplicant income", min_value = 0.0 , max_value = 41667.0 , step = 5.0,value = 500.0)
    Total_Income = ApplicantIncome + CoapplicantIncome
    Loan_Monthly_Paid = round((LoanAmount * 1000) / Loan_Amount_Term)
    Income_After_Loan = ApplicantIncome - Loan_Monthly_Paid
    Log_Total_Income = np.log(Total_Income)
    log_Loan_Monthly_Paid = np.log(Loan_Monthly_Paid)
    log_Income_After_Loan = np.log(Income_After_Loan)
    
    if st.button("Predict"):
        result = prediction(Gender , Married , Dependents , Education , Self_Employed , LoanAmount , Loan_Amount_Term , Credit_History , Property_Area , Log_Total_Income , log_Loan_Monthly_Paid , log_Income_After_Loan)
        list_result = ["Rejected" , "Accepted"]
        st.text(f"Your Loan Is {list_result[result]}.")
Main()
 
