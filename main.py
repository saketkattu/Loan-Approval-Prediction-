import streamlit as st
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,f1_score

st.title("Loan Approval Prediction ")
st.write("""
Find out if you loan application will get approved 
""")

# Variable for getting test data 
gender=st.selectbox("Gender",("Male","Female"))
married=st.selectbox("Are you married",("Yes","No"))
dependents=st.selectbox("Dependents",("0","1","2","3+"))
education =st.selectbox("Education",("Graduate","Not Graduate"))
self_employed=st.selectbox("Self Employed",("Yes","No"))
property_area=st.selectbox("Property Area",("Urban","Rural","SemiUrban"))
credit_history=st.selectbox("Credit History ",(1,0))
loan_time=st.slider("Loan Time",1,480)
model=st.sidebar.selectbox("Select training model",("Decision Tree Classifier","Random Forest Classifier","Logistic Regression"))

output_data=pd.DataFrame({"gender":[gender],"married":[married],"dependents":[dependents],"education":[education],"self_employed":[self_employed]
,"property_area":[property_area],"credit_history":[credit_history],"loan_time":[loan_time]})


# Loading training Data 
df=pd.read_csv("test.csv")

# Preprocessing the training Data 
df_encoded=pd.get_dummies(df,drop_first=True)
X = df_encoded.drop(columns='Loan_Status_Y')
y = df_encoded['Loan_Status_Y']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify =y,random_state =42)


imp = SimpleImputer(strategy='mean')
imp_train = imp.fit(X_train)
X_train = imp_train.transform(X_train)
X_test_imp = imp_train.transform(X_test)


#preprocessing test data 
output_data_encoded=pd.get_dummies(output_data,drop_first=True)

