import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd
from PIL import Image
import time
import openpyxl
from openpyxl.utils import get_column_letter
import face_recognition
import requests
import threading
from queue import Queue
import urllib.request

NUM_REGISTRATION_IMAGES = 10

# Global queue for frames
frame_queue = Queue(maxsize=2)
# Global flag to control streaming
stop_streaming = False

def stream_frames(esp32_ip):
    """Stream frames continuously from ESP32-CAM"""
    global stop_streaming
    while not stop_streaming:
        try:
            esp32_ip = esp32_ip.replace('http://', '').replace('/', '')
            url = f"http://{esp32_ip}/capture"
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(imgnp, -1)
            
            if frame is not None:
                # Keep only the latest frame
                if frame_queue.full():
                    frame_queue.get()
                frame_queue.put(frame)
            time.sleep(0.1)  # Small delay to prevent overwhelming the ESP32
        except Exception as e:
            print(f"Streaming error: {e}")
            time.sleep(1)

def start_streaming(esp32_ip):
    """Start the streaming thread"""
    global stop_streaming
    stop_streaming = False
    stream_thread = threading.Thread(target=stream_frames, args=(esp32_ip,))
    stream_thread.daemon = True
    stream_thread.start()
    return stream_thread

def stop_streaming_thread():
    """Stop the streaming thread"""
    global stop_streaming
    stop_streaming = True

# Fixed function for loading registered faces
def get_registered_faces():
    """Load all registered faces and their encodings"""
    registered_faces = []
    if os.path.exists('data/students.csv'):
        students_df = pd.read_csv('data/students.csv')
        # Drop empty rows to avoid issues
        students_df = students_df.dropna(subset=['student_id', 'student_name'])
        
        for _, row in students_df.iterrows():
            student_folder = f"registered_faces/{row['student_id']}_{row['student_name']}"
            if os.path.exists(student_folder):
                face_encodings = []
                for image_file in os.listdir(student_folder):
                    if image_file.endswith(('.jpg', '.jpeg', '.png')):
                        face_path = os.path.join(student_folder, image_file)
                        try:
                            face_img = face_recognition.load_image_file(face_path)
                            encoding = face_recognition.face_encodings(face_img)
                            if len(encoding) > 0:
                                face_encodings.append(encoding[0])
                        except Exception as e:
                            st.error(f"Error loading face image {face_path}: {e}")
                
                if face_encodings:
                    registered_faces.append({
                        'student_id': row['student_id'],
                        'student_name': row['student_name'],
                        'face_encodings': face_encodings
                    })
                    print(f"Loaded face encodings for {row['student_name']}")
    
    print(f"Total registered faces loaded: {len(registered_faces)}")
    return registered_faces

# Fixed attendance data loading function
def load_attendance_data():
    """Load attendance data from Excel file"""
    if os.path.exists('data/attendance.xlsx'):
        try:
            df = pd.read_excel('data/attendance.xlsx')
            # Clean up empty rows properly
            df = df.dropna(how='all').reset_index(drop=True)
            return df
        except Exception as e:
            st.error(f"Error loading attendance data: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# Cache ESP32 frame getter with TTL
@st.cache_data(ttl=1)  # Cache for 1 second
def get_esp32_frame(esp32_ip):
    """Get a frame from ESP32-CAM"""
    try:
        esp32_ip = esp32_ip.replace('http://', '').replace('/', '')
        url = f"http://{esp32_ip}/capture"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            image_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return True, frame
        return False, None
    except Exception as e:
        return False, None

# Initialize session state variables with better caching control
def init_session_state():
    if 'capture_count' not in st.session_state:
        st.session_state.capture_count = 0
    # Always refresh registered faces on app startup
    st.session_state.registered_faces = get_registered_faces()
    # Always refresh attendance data on app startup
    st.session_state.attendance_data = load_attendance_data()

def create_folders():
    """Create necessary folders for the system"""
    base_folders = ['registered_faces', 'attendance_captures', 'data']
    for folder in base_folders:
        os.makedirs(folder, exist_ok=True)

def register_student():
    """Register a new student with multiple face captures"""
    st.header("Register New Student")
    
    esp32_ip = st.text_input("Enter ESP32-CAM IP Address", "192.168.1.XXX")
    
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("Enter Student ID")
    with col2:
        student_name = st.text_input("Enter Student Name")
    
    if student_id and student_name and esp32_ip:
        if os.path.exists('data/students.csv'):
            existing_students = pd.read_csv('data/students.csv')
            # Clean the DataFrame from any empty rows
            existing_students = existing_students.dropna(subset=['student_id'])
            if student_id in existing_students['student_id'].values:
                st.error("Student ID already exists!")
                return

        student_folder = f"registered_faces/{student_id}_{student_name}"
        os.makedirs(student_folder, exist_ok=True)
        
        st.info(f"We'll capture {NUM_REGISTRATION_IMAGES} photos. Please move your head slightly between captures.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            frame_placeholder = st.empty()
        with col2:
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            capture_button = st.button("Capture Face")

        # Start streaming when IP is entered
        if esp32_ip and esp32_ip != "192.168.1.XXX":
            stream_thread = start_streaming(esp32_ip)
            
            try:
                while True:
                    if not frame_queue.empty():
                        frame = frame_queue.get()
                        if frame is not None:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            face_locations = face_recognition.face_locations(rgb_frame)
                            
                            display_frame = frame.copy()
                            for (top, right, bottom, left) in face_locations:
                                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            
                            frame_placeholder.image(display_frame, channels="BGR")
                            
                            if face_locations:
                                status_placeholder.success(f"Face detected! ({st.session_state.capture_count}/{NUM_REGISTRATION_IMAGES})")
                                
                                if capture_button and st.session_state.capture_count < NUM_REGISTRATION_IMAGES:
                                    for (top, right, bottom, left) in face_locations:
                                        face_img = frame[top:bottom, left:right]
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        face_path = f"{student_folder}/face_{st.session_state.capture_count + 1}_{timestamp}.jpg"
                                        cv2.imwrite(face_path, face_img)
                                        
                                        st.session_state.capture_count += 1
                                        progress_bar.progress(st.session_state.capture_count / NUM_REGISTRATION_IMAGES)
                                        
                                        if st.session_state.capture_count == NUM_REGISTRATION_IMAGES:
                                            df = pd.DataFrame({
                                                'student_id': [student_id],
                                                'student_name': [student_name],
                                                'registration_date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                                            })
                                            
                                            # Properly handle CSV file for student data
                                            if os.path.exists('data/students.csv'):
                                                existing_df = pd.read_csv('data/students.csv')
                                                # Clean up any empty rows first
                                                existing_df = existing_df.dropna(subset=['student_id'])
                                                combined_df = pd.concat([existing_df, df], ignore_index=True)
                                                combined_df.to_csv('data/students.csv', index=False)
                                            else:
                                                df.to_csv('data/students.csv', index=False)
                                            
                                            st.success(f"Successfully registered {student_name}!")
                                            st.session_state.capture_count = 0
                                            # Refresh the registered faces in session state
                                            st.session_state.registered_faces = get_registered_faces()
                                            stop_streaming_thread()
                                            return
                            else:
                                status_placeholder.warning("No face detected")
                                
                    time.sleep(0.1)
            except Exception as e:
                st.error(f"Error in video stream: {e}")
            finally:
                stop_streaming_thread()

def take_attendance():
    """Take attendance with face recognition"""
    st.header("Take Attendance")
    
    # Debug info about registered faces
    if not st.session_state.registered_faces:
        st.info("Checking for registered faces...")
        # Force refresh registered faces
        st.session_state.registered_faces = get_registered_faces()
    
    esp32_ip = st.text_input("Enter ESP32-CAM IP Address", "192.168.1.XXX")
    
    periods = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]
    current_hour = datetime.now().hour
    default_period = periods[min(max(current_hour - 8, 0), 7)] if 8 <= current_hour <= 16 else "1st"
    
    period = st.selectbox("Select Period", periods, index=periods.index(default_period))
    
    if not st.session_state.registered_faces:
        st.error("No registered faces found! Please register students first.")
        return
    else:
        st.success(f"Found {len(st.session_state.registered_faces)} registered students")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        frame_placeholder = st.empty()
    with col2:
        status_placeholder = st.empty()
        process_button = st.button("Mark Attendance")

    if esp32_ip and esp32_ip != "192.168.1.XXX":
        stream_thread = start_streaming(esp32_ip)
        
        try:
            while True:
                if not frame_queue.empty():
                    frame = frame_queue.get()
                    if frame is not None:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_frame)
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        
                        display_frame = frame.copy()
                        
                        # Process each detected face
                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            matched = False
                            # Check against registered faces
                            for reg_face in st.session_state.registered_faces:
                                if any(face_recognition.compare_faces([encoding], face_encoding, tolerance=0.6)[0] 
                                      for encoding in reg_face['face_encodings']):
                                    matched = True
                                    # Draw green box for recognized face
                                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                                    
                                    # Prepare label with name and ID
                                    label = f"{reg_face['student_name']} (ID: {reg_face['student_id']})"
                                    
                                    # Calculate label position and background
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 0.6
                                    thickness = 2
                                    (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                                    
                                    # Draw label background
                                    cv2.rectangle(display_frame, 
                                                (left, top - label_height - 10), 
                                                (left + label_width, top), 
                                                (0, 255, 0), 
                                                cv2.FILLED)
                                    
                                    # Draw label text
                                    cv2.putText(display_frame, 
                                              label, 
                                              (left, top - 5), 
                                              font, 
                                              font_scale, 
                                              (0, 0, 0), 
                                              thickness)
                                    
                                    if process_button:
                                        mark_attendance(reg_face['student_id'], reg_face['student_name'], period)
                                        st.success(f"Attendance marked for {reg_face['student_name']}")
                                        stop_streaming_thread()
                                        return
                                    break
                            
                            if not matched:
                                # Draw red box for unrecognized face
                                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                                label = "Unknown"
                                
                                # Calculate label position and background
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.6
                                thickness = 2
                                (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                                
                                # Draw label background
                                cv2.rectangle(display_frame, 
                                            (left, top - label_height - 10), 
                                            (left + label_width, top), 
                                            (0, 0, 255), 
                                            cv2.FILLED) 
                                # Draw label text
                                cv2.putText(display_frame, 
                                          label, 
                                          (left, top - 5), 
                                          font, 
                                          font_scale, 
                                          (0, 0, 0), 
                                          thickness)
                        
                        frame_placeholder.image(display_frame, channels="BGR")
                        
                        if face_locations:
                            status_placeholder.success("Face detected!")
                        else:
                            status_placeholder.warning("No face detected")
                
                time.sleep(0.1)
        except Exception as e:
            st.error(f"Error in video stream: {e}")
        finally:
            stop_streaming_thread()

def mark_attendance(student_id, student_name, period):
    """Mark attendance for a recognized student"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    attendance_file = 'data/attendance.xlsx'
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Get a fresh copy of attendance data to avoid working with stale data
    if os.path.exists(attendance_file):
        try:
            df = pd.read_excel(attendance_file)
            df = df.dropna(how='all').reset_index(drop=True)  # Clean up empty rows
        except Exception as e:
            st.error(f"Error reading attendance file: {e}")
            df = pd.DataFrame(columns=['Date', 'Student ID', 'Student Name', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th'])
    else:
        df = pd.DataFrame(columns=['Date', 'Student ID', 'Student Name', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th'])
    
    mask = (df['Date'] == date_str) & (df['Student ID'] == student_id)
    
    if len(df[mask]) > 0:
        df.loc[mask, period] = timestamp
    else:
        new_record = pd.DataFrame([{
            'Date': date_str,
            'Student ID': student_id,
            'Student Name': student_name,
            period: timestamp
        }])
        df = pd.concat([df, new_record], ignore_index=True)
    
    # Save the updated attendance data and refresh the session state
    try:
        df.to_excel(attendance_file, index=False)
        st.session_state.attendance_data = df  # Update session state with new data
    except Exception as e:
        st.error(f"Error saving attendance data: {e}")

def view_attendance():
    """View and display attendance records"""
    st.header("View Attendance")
    
    # Refresh button to reload attendance data
    if st.button("Refresh Attendance Data"):
        st.session_state.attendance_data = load_attendance_data()
        st.success("Attendance data refreshed!")
    
    if os.path.exists('data/attendance.xlsx'):
        # Always get fresh data when viewing attendance
        df = load_attendance_data()
        st.session_state.attendance_data = df
    else:
        st.warning("No attendance records file found.")
        return
    
    if st.session_state.attendance_data.empty:
        st.warning("No attendance records found.")
        return
    
    df = st.session_state.attendance_data
    
    col1, col2 = st.columns(2)
    with col1:
        dates = sorted(df['Date'].unique(), reverse=True)
        selected_date = st.selectbox("Select Date", ['All'] + list(dates))
    
    with col2:
        students = sorted(df['Student ID'].unique())
        selected_student = st.selectbox("Select Student", ['All'] + list(students))
    
    filtered_df = df.copy()
    if selected_date != 'All':
        filtered_df = filtered_df[filtered_df['Date'] == selected_date]
    if selected_student != 'All':
        filtered_df = filtered_df[filtered_df['Student ID'] == selected_student]
    
    # Sort by date and student ID for better readability
    filtered_df = filtered_df.sort_values(by=['Date', 'Student ID'])
    
    st.dataframe(filtered_df)
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Attendance Records",
        data=csv,
        file_name=f"attendance_records_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def main():
    st.set_page_config(page_title="Face Recognition Attendance", layout="wide")
    
    # Create needed folders
    create_folders()
    
    # Clear any stale cache data on startup
    st.cache_data.clear()
    
    # Initialize session state
    init_session_state()
    
    st.title("Face Recognition Attendance System")
    
    menu = ["Register Student", "Take Attendance", "View Attendance"]
    choice = st.sidebar.selectbox("Select Option", menu)
    
    # Show debug information in sidebar (can be removed in production)
    with st.sidebar.expander("Debug Info"):
        st.write(f"Registered Faces: {len(st.session_state.registered_faces)}")
        if st.session_state.registered_faces:
            for face in st.session_state.registered_faces:
                st.write(f"- {face['student_name']} (ID: {face['student_id']})")
        
        if st.button("Refresh Registered Faces"):
            st.session_state.registered_faces = get_registered_faces()
            st.success(f"Refreshed! Found {len(st.session_state.registered_faces)} registered faces")
    
    if choice == "Register Student":
        register_student()
    elif choice == "Take Attendance":
        take_attendance()
    else:
        view_attendance()

if __name__ == "__main__":
    main()
                                    