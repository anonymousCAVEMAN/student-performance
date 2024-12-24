import streamlit as st
import requests

# Create select boxes for each variable
st.title("Student Information")
gender = st.selectbox("Gender", ['female', 'male'])
race_ethnicity = st.selectbox("Race/Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'])
parental_level_of_education = st.selectbox("Parental Level of Education", ["bachelor's degree", "some college", "master's degree", "associate's degree", 'high school', 'some high school'])
lunch = st.selectbox("Lunch", ['standard', 'free/reduced'])
test_preparation_course = st.selectbox("Test Preparation Course", ['none', 'completed'])
reading_score = st.number_input("Reading Score (0-100)", min_value=0, max_value=100)
writing_score = st.number_input("Writing Score (0-100)", min_value=0, max_value=100)


# Create a dictionary of the input values
input_data = {
    "gender": gender,
    "race_ethnicity": race_ethnicity,
    "parental_level_of_education": parental_level_of_education,
    "lunch": lunch,
    "test_preparation_course": test_preparation_course,
    "reading_score":reading_score,
    "writing_score":writing_score
    
}

# Send the dictionary to the FastAPI server
if st.button("Submit"):
    # Send the data to FastAPI
    url = "http://127.0.0.1:8000/predict"  
    try:
        # Send the request as JSON payload
        response = requests.post(url, json=input_data)  # Use the 'json' argument instead of 'data'
        
        if response.status_code == 200:
            st.write("ML Model Output: ", response.json())
        else:
            st.error("Server returned an error: " + response.text)
    except Exception as e:
        st.error("Error occurred while communicating with server: " + str(e))
    st.write("Sending the following data to server: ", input_data)