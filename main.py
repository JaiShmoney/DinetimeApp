import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import openai
from datetime import datetime

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]


# Load the wait time dataset
file_path = './popular_fast_food_wait_times.csv'
wait_time_data = pd.read_csv(file_path)

# Function to get restaurant recommendations from OpenAI
def get_completion(prompt, model="gpt-4"):
    chat_completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing restaurant recommendations with opening and closing hours."},
            {"role": "user", "content": prompt},
        ]
    )
    return chat_completion.choices[0].message.content

# Function to get adjusted wait time based on dataset
def get_dynamic_wait_time(restaurant, location, day_of_week, time_of_day, line_length):
    # Filter the dataset to match the selected conditions
    filtered_data = wait_time_data[
        (wait_time_data['Restaurant Name'] == restaurant) &
        (wait_time_data['Location'] == location) &
        (wait_time_data['Day of the Week'] == day_of_week) &
        (wait_time_data['Time of Day'] == time_of_day) &
        (wait_time_data['Line Length (People)'] >= line_length)
    ]
    
    # Return the average wait time if available; otherwise, use a default
    if not filtered_data.empty:
        return filtered_data['Estimated Wait Time (Minutes)'].mean()
    else:
        return 5  # Default wait time if no matching data is found

# Function to detect and count people in an image, blur faces, and draw bounding boxes
def detect_people(image):
    # Convert the PIL image to a format OpenCV can use
    open_cv_image = np.array(image.convert('RGB'))
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    # Initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Initialize the face detector (Haar cascade for face detection)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect people in the image
    (rects, _) = hog.detectMultiScale(open_cv_image, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # Blur faces for privacy
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        # Apply Gaussian blur to the face area
        face = open_cv_image[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
        open_cv_image[y:y+h, x:x+w] = blurred_face

    # Draw bounding boxes for detected people (optional for visualization)
    for (x, y, w, h) in rects:
        cv2.rectangle(open_cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with bounding boxes and blurred faces
    st.image(open_cv_image, caption="Detected People with Blurred Faces", channels="BGR", use_column_width=True)

    # Return the count of detected people
    return len(rects)

# Function to generate a line image using OpenAI's DALL-E
def generate_line_image(restaurant, location, day_of_week, time_of_day):
    prompt = f"A realistic image of people waiting in line at {restaurant} in {location} during {time_of_day} on a {day_of_week}. Depict the line length typical for this time and place."
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512"
    )
    image_url = response['data'][0]['url']
    return image_url

# Streamlit app layout
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Estimate Wait Time", "Recommendations", "Line Generator"])

with tab1:
    st.image("DineTimelogo.png", width=300)
    st.title("Welcome to DineTime.ai")
    st.write("At DineTime.ai, our mission is to make your life easier by saving you time and reducing the hassle of waiting in line for your favorite meals. "
             "We know that every minute counts, and with our innovative line analysis and wait-time estimation, we empower you to make the most of your dining experience. "
             "Simply snap a photo of the line, and let us provide a quick, accurate estimate of the wait time so you can make an informed choice. Whether you’re on a lunch break or on the go, we’re here to help you plan ahead and enjoy your meal sooner. "
             "Dine smarter with DineTime.ai – because your time is valuable.")

with tab2:
    st.header("Estimate Wait Time")

    # File uploader for the image
    uploaded_image = st.file_uploader("Upload an image of the line", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Detect and count people in the image, blur faces
        with st.spinner("Analyzing the image..."):
            try:
                people_count = detect_people(image)
                st.write(f"Detected People Count: {people_count}")

                # User selects restaurant details
                restaurant = st.selectbox("Select the Restaurant:", wait_time_data['Restaurant Name'].unique())
                location = st.selectbox("Select the Location:", wait_time_data['Location'].unique())
                time_of_day = st.selectbox("Select the Time of Day:", wait_time_data['Time of Day'].unique())
                
                # Get current day of the week
                current_day = datetime.now().strftime("%A")

                # Get dynamic wait time based on dataset
                dynamic_service_time = get_dynamic_wait_time(restaurant, location, current_day, time_of_day, people_count)

                # Display estimated wait time
                st.subheader(f"Estimated Wait Time: {dynamic_service_time} minutes")
                st.write(f"Based on selected conditions and detected line length.")
            except Exception as e:
                st.error(f"Error in analyzing the image: {e}")

with tab3:
    st.header("Recommendations")
    st.write("Looking for the best place to grab a bite nearby? Tell us your location and food preference, and we’ll suggest some great options!")

    # User inputs for location and food preference
    location = st.text_input("Enter your location (e.g., city or neighborhood)")
    food_preference = st.selectbox("Select your food preference", ["Fast Food", "Chinese", "Italian", "Mexican", "Vegan", "Dessert", "Other"])

    if location and food_preference:
        # Modify the prompt to include opening and closing times
        prompt = f"Please suggest some popular {food_preference} restaurants in {location}. Include a brief description, why they’re recommended, and their typical opening and closing hours."
        
        # Get recommendations with opening and closing times using the get_completion function
        with st.spinner('Fetching recommendations...'):
            try:
                recommendations = get_completion(prompt)
                st.subheader("Recommended Places with Operating Hours:")
                st.write(recommendations)
            except Exception as e:
                st.error(f"Error fetching recommendations: {e}")

with tab4:
    st.header("Line Generator")
    st.write("Generate an image of a line at a selected restaurant and location for a specific time and day.")

    # User text inputs for generating a line image
    restaurant = st.text_input("Enter the Restaurant Name:", key="line_gen_restaurant")
    location = st.text_input("Enter the Location:", key="line_gen_location")
    time_of_day = st.text_input("Enter the Time of Day (e.g., morning, afternoon, evening):", key="line_gen_time_of_day")
    day_of_week = st.text_input("Enter the Day of the Week:", key="line_gen_day_of_week")

    # Generate and display line image
    if st.button("Generate Line Image"):
        with st.spinner("Generating image..."):
            try:
                # Generate prompt based on free-form input
                prompt = f"A realistic image of people waiting in line at {restaurant} in {location} during {time_of_day} on a {day_of_week}. Depict the line length typical for this time and place."
                response = openai.Image.create(
                    prompt=prompt,
                    n=1,
                    size="512x512"
                )
                line_image_url = response['data'][0]['url']
                st.image(line_image_url, caption=f"Line at {restaurant} in {location} during {time_of_day} on a {day_of_week}")
            except Exception as e:
                st.error(f"Error generating image: {e}")


