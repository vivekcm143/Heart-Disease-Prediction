# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved models and scaler
lr_model = joblib.load('models/heart_disease_lr_model_5.pkl')
dt_model = joblib.load('models/heart_disease_dt_model_5.pkl')
rf_model = joblib.load('models/heart_disease_rf_model_5.pkl')
svm_model = joblib.load('models/heart_disease_svm_model_5.pkl')
scaler = joblib.load('models/scaler_5.pkl')

# Load the columns used during training
with open('models/columns.pkl', 'rb') as f:
    columns_used = joblib.load(f)

# Streamlit app title
st.title("Heart Disease Prediction App")

# Description of the app
st.write("""
    This application predicts the likelihood of heart disease based on input health parameters. 
    Select different models to see their predictions.
""")

# Input fields for user data
st.sidebar.header("Enter Health Information")

# Input fields for each feature
age = st.sidebar.number_input('Age', min_value=1, max_value=120, value=50)
sex = st.sidebar.selectbox('Sex', options=['Male', 'Female'])
cp = st.sidebar.selectbox('Chest Pain Type (cp)', options=[0, 1, 2, 3])
trestbps = st.sidebar.number_input('Resting Blood Pressure (trestbps)', min_value=90, max_value=200, value=120)
chol = st.sidebar.number_input('Cholesterol (chol)', min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[0, 1])  # 0 = No, 1 = Yes
restecg = st.sidebar.selectbox('Resting Electrocardiographic Results (restecg)', options=[0, 1, 2])
thalach = st.sidebar.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=150)
exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', options=[0, 1])
oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=6.0, value=1.0)
slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment (slope)', options=[0, 1, 2])
ca = st.sidebar.selectbox('Number of Major Vessels Colored by Fluoroscopy (ca)', options=[0, 1, 2, 3, 4])
thal = st.sidebar.selectbox('Thalassemia (thal)', options=[0, 1, 2, 3])

# Prepare the input data into the same format used for training
user_input = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                          columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

# Convert sex to numeric value
user_input['sex'] = user_input['sex'].map({'Female': 0, 'Male': 1})

# One-hot encode the categorical features using the same encoding as during training
user_input = pd.get_dummies(user_input, columns=['sex', 'cp', 'restecg', 'slope', 'ca', 'thal'], drop_first=True)

# Ensure the user input matches the columns used during training
missing_cols = set(columns_used) - set(user_input.columns)
for col in missing_cols:
    user_input[col] = 0
user_input = user_input[columns_used]

# Scale the features using the same scaler
user_input_scaled = scaler.transform(user_input)

# Prediction function for each model
def predict(model):
    prediction = model.predict(user_input_scaled)
    return prediction[0]

# Display the prediction results
if st.button("Predict"):
    # Predictions from all models
    st.write("### Model Predictions:")

    # Logistic Regression Prediction
    lr_prediction = predict(lr_model)
    st.write(f"**Logistic Regression Prediction**: {'Heart Disease' if lr_prediction == 1 else 'No Heart Disease'}")

    # Decision Tree Prediction
    dt_prediction = predict(dt_model)
    st.write(f"**Decision Tree Prediction**: {'Heart Disease' if dt_prediction == 1 else 'No Heart Disease'}")

    # Random Forest Prediction
    rf_prediction = predict(rf_model)
    st.write(f"**Random Forest Prediction**: {'Heart Disease' if rf_prediction == 1 else 'No Heart Disease'}")

    # SVM Prediction
    svm_prediction = predict(svm_model)
    st.write(f"**SVM Prediction**: {'Heart Disease' if svm_prediction == 1 else 'No Heart Disease'}")

# Optional section to view additional insights
st.sidebar.subheader("Show Additional Insights")
if st.sidebar.checkbox("Show feature importance (Random Forest only)"):
    # Display feature importance for Random Forest
    rf_feature_importance = pd.DataFrame({
        'Feature': columns_used,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.write("### Random Forest Feature Importance")
    st.dataframe(rf_feature_importance)

st.sidebar.write("Thank you for using the app!")
