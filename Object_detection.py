import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import pandas as pd
import face_recognition
import requests
from PIL import Image
import io
import time

class AttendanceSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.attendance_df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
        
    def load_student_data(self, student_images_dict):
        """Load and encode student face data"""
        for name, image in student_images_dict.items():
            face_encoding = face_recognition.face_encodings(image)[0]
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
    
    def mark_attendance(self, name):
        """Mark attendance for a recognized student"""
        now = datetime.now()
        date = now.strftime('%Y-%m-%d')
        time = now.strftime('%H:%M:%S')
        
        # Check if student already marked attendance today
        today_records = self.attendance_df[
            (self.attendance_df['Name'] == name) & 
            (self.attendance_df['Date'] == date)
        ]
        
        if len(today_records) == 0:
            self.attendance_df = pd.concat([
                self.attendance_df,
                pd.DataFrame([[name, date, time]], columns=['Name', 'Date', 'Time'])
            ], ignore_index=True)
            return True
        return False

def main():
    st.title("Student Attendance System")
    
    # Initialize session state
    if 'attendance_system' not in st.session_state:
        st.session_state.attendance_system = AttendanceSystem()
        
    if 'processing_active' not in st.session_state:
        st.session_state.processing_active = False

    # ESP32 Camera IP input
    esp_ip = st.text_input("Enter ESP32 Camera IP Address", "http://192.168.1.100")
    
    # Layout columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Video feed display
        st.subheader("Live Feed")
        video_placeholder = st.empty()
        
    with col2:
        # Attendance controls and display
        st.subheader("Attendance Controls")
        start_button = st.button("Start Processing")
        stop_button = st.button("Stop Processing")
        take_attendance = st.button("Take Attendance")
        
        st.subheader("Today's Attendance")
        attendance_placeholder = st.empty()

    def process_video_feed():
        try:
            while st.session_state.processing_active:
                # Get video feed from ESP32
                response = requests.get(f"{esp_ip}/stream")
                img_array = np.array(Image.open(io.BytesIO(response.content)))
                
                # Convert to RGB for face_recognition library
                rgb_frame = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                
                # Find faces in frame
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                # Draw boxes around faces
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(
                        st.session_state.attendance_system.known_face_encodings,
                        face_encoding
                    )
                    
                    if True in matches:
                        match_index = matches.index(True)
                        name = st.session_state.attendance_system.known_face_names[match_index]
                        
                        # Draw box and name
                        cv2.rectangle(img_array, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(img_array, name, (left, top - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display processed frame
                video_placeholder.image(img_array, channels="RGB")
                
                # Update attendance display
                today = datetime.now().strftime('%Y-%m-%d')
                today_attendance = st.session_state.attendance_system.attendance_df[
                    st.session_state.attendance_system.attendance_df['Date'] == today
                ]
                attendance_placeholder.dataframe(today_attendance)
                
                time.sleep(0.1)  # Prevent excessive CPU usage
                
        except Exception as e:
            st.error(f"Error processing video feed: {str(e)}")
            st.session_state.processing_active = False

    # Handle button clicks
    if start_button:
        st.session_state.processing_active = True
        process_video_feed()
        
    if stop_button:
        st.session_state.processing_active = False
        
    if take_attendance and st.session_state.processing_active:
        try:
            # Get current frame and process for attendance
            response = requests.get(f"{esp_ip}/stream")
            img_array = np.array(Image.open(io.BytesIO(response.content)))
            rgb_frame = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    st.session_state.attendance_system.known_face_encodings,
                    face_encoding
                )
                
                if True in matches:
                    match_index = matches.index(True)
                    name = st.session_state.attendance_system.known_face_names[match_index]
                    if st.session_state.attendance_system.mark_attendance(name):
                        st.success(f"Attendance marked for {name}")
                    else:
                        st.info(f"Attendance already marked for {name} today")
                        
        except Exception as e:
            st.error(f"Error taking attendance: {str(e)}")

if __name__ == "__main__":
    main()