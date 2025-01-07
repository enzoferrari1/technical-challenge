import streamlit as st
import numpy as np
from src.models import CustomRandomForestModel

@st.cache_resource  # Caches the model for faster subsequent runs
def load_model():
    model = CustomRandomForestModel()
    # model.build()
    model.load_model()
    return model

# Define the function
def preprocess_input(age, yoe, gender, education, roles):
    """
    Preprocess the input for the model.
    
    Args:
    - age (float): Age of the individual.
    - yoe (float): Years of experience.
    - gender (str): Either "Female" or "Male".
    - education (str): One of "Bachelor's", "Master's", or "PhD".
    - roles (list of str): List of active roles (e.g., ["Finance", "HR"]).
    
    Returns:
    - numpy array: Preprocessed input for the model.
    """
    # Initialize a zero vector for all features
    feature_vector = np.zeros(24)  # Total number of columns
    
    # Set numerical features
    feature_vector[0] = age
    feature_vector[1] = yoe
    
    # Set gender features
    if gender == "Female":
        feature_vector[19] = 1
    elif gender == "Male":
        feature_vector[20] = 1
    
    # Set education features
    education_map = {
        "Bachelor's": 21,
        "Master's": 22,
        "PhD": 23
    }
    if education in education_map:
        feature_vector[education_map[education]] = 1
    
    # Set role features
    role_map = {
        "Finance": 2,
        "HR": 3,
        "Marketing": 4,
        "Sales": 5,
        "Customer Support": 6,
        "Operations": 7,
        "Data": 8,
        "Analyst": 9,
        "Scientist": 10,
        "Project": 11,
        "Product": 12,
        "Engineer": 13,
        "Development": 14,
        "Junior": 15,
        "Senior": 16,
        "Director": 17,
        "Manager": 18
    }
    for role in roles:
        if role in role_map:
            feature_vector[role_map[role]] = 1
    
    return feature_vector

def main():
    st.title('Salary prediction')
    st.text("Input your description")
    model = load_model()
    # User inputs
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    yoe = st.number_input("Years of Experience", min_value=0, max_value=40, step=1)
    gender = st.selectbox("Gender", options=["Female", "Male"])
    education = st.selectbox("Education Level", options=["Bachelor's", "Master's", "PhD"])
    roles = st.multiselect("Roles", options=[
        "Finance", "HR", "Marketing", "Sales", "Customer Support", 
        "Operations", "Data", "Analyst", "Scientist", "Project", 
        "Product", "Engineer", "Development", "Junior", 
        "Senior", "Director", "Manager"
    ])
    # Preprocess and predict
    if st.button("Predict"):
        input_vector = preprocess_input(age, yoe, gender, education, roles)
        prediction = model.inference(input_vector.reshape(1, -1))
        st.write(f"Prediction: {prediction[0]:.2f}")
    

if __name__ == '__main__':
    main()