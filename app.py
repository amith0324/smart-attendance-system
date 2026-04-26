import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import cv2
import numpy as np
import time

from database import get_attendance_logs, get_all_users, add_user
from recognition import FaceRecognizer

st.set_page_config(page_title="Smart Attendance Dashboard", layout="wide", page_icon="📊")

@st.cache_resource
def load_recognizer():
    return FaceRecognizer()

def render_dashboard():
    st.title("📊 Smart Attendance Dashboard")
    
    # Metrics
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    logs = get_attendance_logs(date=today)
    all_users = get_all_users()
    
    total_users = len(all_users)
    present_today = len(set([log[1] for log in logs])) # Unique names present
    absent_today = total_users - present_today
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Registered Users", total_users)
    col2.metric("Present Today", present_today)
    col3.metric("Absent Today", absent_today)
    
    st.markdown("---")
    
    # Data Table
    st.subheader(f"Attendance Logs for {today}")
    if logs:
        df = pd.DataFrame(logs, columns=["Log ID", "Name", "Date", "Time", "Status"])
        st.dataframe(df, use_container_width=True)
        
        # Export CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Export as CSV", csv, f"attendance_{today}.csv", "text/csv")
    else:
        st.info("No attendance records found for today.")
        
    st.markdown("---")
    
    # Analytics
    st.subheader("Monthly Analytics")
    all_logs = get_attendance_logs()
    if all_logs:
        df_all = pd.DataFrame(all_logs, columns=["Log ID", "Name", "Date", "Time", "Status"])
        # Group by Date and count unique persons
        df_trend = df_all.groupby('Date')['Name'].nunique().reset_index()
        df_trend.rename(columns={'Name': 'Total Present'}, inplace=True)
        
        fig = px.bar(df_trend, x='Date', y='Total Present', title="Daily Attendance Trend")
        st.plotly_chart(fig, use_container_width=True)
        
        # Pie chart for today's attendance
        fig_pie = px.pie(names=['Present', 'Absent'], values=[present_today, absent_today], title="Today's Attendance Ratio")
        st.plotly_chart(fig_pie)
    else:
        st.info("No data available for analytics.")


def render_add_user():
    st.title("👤 Register New User")
    
    name = st.text_input("Full Name")
    
    # WebRTC or cv2 based capture. Streamlit native camera input is simplest.
    img_file_buffer = st.camera_input("Take a picture")
    
    if img_file_buffer is not None and name:
        if st.button("Register User"):
            # Convert to opencv image
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Simple Face Detection for Registration
            # Using Haar Cascade for simplicity here, or just assume the face is centered.
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                st.error("No face detected! Please ensure your face is clearly visible.")
            else:
                x, y, w, h = faces[0]
                recognizer = load_recognizer()
                face_img = recognizer.extract_face(cv2_img, [x, y, w, h])
                
                if face_img is not None:
                    with st.spinner('Generating embedding...'):
                        embedding = recognizer.get_embedding(face_img)
                        add_user(name, embedding)
                        st.success(f"User '{name}' registered successfully!")
                        st.balloons()
                else:
                    st.error("Could not process the detected face.")

def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", ["Dashboard", "Register User"])
    
    if choice == "Dashboard":
        render_dashboard()
    elif choice == "Register User":
        render_add_user()

if __name__ == "__main__":
    main()
