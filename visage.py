import cv2
import streamlit as st

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in an image
def detect_faces(image, min_neighbors, scale_factor, rectangle_color):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), rectangle_color, 2)

    return image

# Main Streamlit app function
def app():
    st.title("Face Detection using Viola-Jones Algorithm")

    # Instructions
    st.write("## Instructions")
    st.write("1. Press the 'Detect Faces' button to start detecting faces from your webcam.")
    st.write("2. Adjust the parameters using the sliders and color picker.")
    st.write("3. Press 'Save Image' to save the image with detected faces to your device.")

    # Button to start detecting faces
    if st.button("Detect Faces"):
        st.write("Detecting faces. Press 'q' to stop.")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            frame_with_faces = detect_faces(frame, st.slider("minNeighbors", 1, 10, 5), st.slider("scaleFactor", 1.01, 2.0, 1.1, step=0.01), st.color_picker("Select Rectangle Color", "#ff5733"))
            cv2.imshow('Face Detection using Viola-Jones Algorithm', frame_with_faces)

            # Exit the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # Save the image with detected faces
    if st.button("Save Image"):
        if 'frame_with_faces' in locals():
            cv2.imwrite('detected_faces_image.jpg', frame_with_faces)
            st.success("Image with detected faces saved successfully!")

if __name__ == "__main__":
    app()
