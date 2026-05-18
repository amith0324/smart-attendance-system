import os
import sys

import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import cv2
import numpy as np

from database import get_attendance_logs, get_all_users, add_user, delete_user
from recognition import FaceRecognizer

st.set_page_config(
    page_title="Smart Attendance Dashboard",
    page_icon="📊",
    layout="wide",
)

DARK_STYLE = """
<style>
    :root {
        color-scheme: dark;
    }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .stApp {
        background: radial-gradient(circle at top left, rgba(56, 94, 191, 0.25), transparent 30%),
                    radial-gradient(circle at bottom right, rgba(134, 52, 168, 0.18), transparent 30%),
                    #0e1117;
        color: #e3e9f9;
    }
    .css-18e3th9 {
        background-color: transparent;
    }
    .css-1d391kg {
        background: rgba(22, 28, 38, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 16px 50px rgba(0, 0, 0, 0.35);
        border-radius: 18px;
    }
    .css-1v0mbdj, .css-10trblm {
        background: rgba(255, 255, 255, 0.03);
    }
    .stButton>button {
        background-color: #4169e1;
        color: white;
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3050c0;
    }
    .css-1aumxhk {
        background-color: rgba(33, 39, 53, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    header, footer, [data-testid="collapsedControl"] {
        visibility: hidden;
    }
    .css-1avcm0n {
        padding-top: 0;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #f7fbff;
    }
    .stMetricValue {
        color: #f7fbff;
    }
</style>
"""

@st.cache_resource
def load_recognizer():
    return FaceRecognizer()


def inject_style():
    st.markdown(DARK_STYLE, unsafe_allow_html=True)


def render_header():
    st.markdown("### Smart Attendance Dashboard")
    st.markdown(
        "##### A modern dark UI for real-time attendance, face recognition, and analytics."
    )


def render_dashboard():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    logs = get_attendance_logs(date=today)
    all_users = get_all_users()

    total_users = len(all_users)
    present_today = len(set([log[1] for log in logs]))
    absent_today = max(total_users - present_today, 0)

    st.markdown("<div style='display:flex;gap:1rem;flex-wrap:wrap;'>"
                f"<div style='flex:1;min-width:200px;background:rgba(255,255,255,0.04);padding:1.3rem;border-radius:18px;'>"
                f"<h4 style='margin:0;color:#7fb7ff;'>Total Registered</h4>"
                f"<p style='font-size:2rem;font-weight:700;margin:0;'>{total_users}</p>"
                "</div>"
                f"<div style='flex:1;min-width:200px;background:rgba(255,255,255,0.04);padding:1.3rem;border-radius:18px;'>"
                f"<h4 style='margin:0;color:#7cffb9;'>Present Today</h4>"
                f"<p style='font-size:2rem;font-weight:700;margin:0;'>{present_today}</p>"
                "</div>"
                f"<div style='flex:1;min-width:200px;background:rgba(255,255,255,0.04);padding:1.3rem;border-radius:18px;'>"
                f"<h4 style='margin:0;color:#ff8fb1;'>Absent Today</h4>"
                f"<p style='font-size:2rem;font-weight:700;margin:0;'>{absent_today}</p>"
                "</div>"
                "</div>", unsafe_allow_html=True)

    if all_users:
        st.markdown("### Manage registered users")
        delete_col, info_col = st.columns([3, 1])
        user_options = {f"{user[1]} (ID {user[0]})": user[0] for user in all_users}
        selected_user = delete_col.selectbox("Choose user to delete", [""] + list(user_options.keys()))
        if selected_user:
            if delete_col.button("🗑️ Delete user"):
                delete_user(user_options[selected_user])
                st.success(f"Deleted {selected_user} and their attendance history.")
                st.rerun()
        info_col.info("Delete a user only when you want to remove them completely from the system.")

    st.divider()
    st.subheader(f"Attendance Logs for {today}")

    if logs:
        df = pd.DataFrame(logs, columns=["Log ID", "Name", "Date", "Time", "Status"])
        st.dataframe(df, width="stretch")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Export as CSV", csv, f"attendance_{today}.csv", "text/csv")
    else:
        st.error("No attendance records found for today.")

    st.divider()
    st.subheader("Monthly Analytics")

    all_logs = get_attendance_logs()
    if all_logs:
        df_all = pd.DataFrame(all_logs, columns=["Log ID", "Name", "Date", "Time", "Status"])
        df_trend = df_all.groupby("Date")["Name"].nunique().reset_index()
        df_trend.rename(columns={"Name": "Total Present"}, inplace=True)

        fig = px.bar(
            df_trend,
            x="Date",
            y="Total Present",
            title="Daily Attendance Trend",
            template="plotly_dark",
            color_discrete_sequence=["#6db9ff"],
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

        fig_pie = px.pie(
            names=["Present", "Absent"],
            values=[present_today, absent_today],
            title="Today's Attendance Ratio",
            template="plotly_dark",
            color_discrete_sequence=["#57d7a3", "#ff6b8a"],
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, width="stretch")
    else:
        st.info("No analytics data available yet.")


def render_add_user():
    st.markdown("### Register a New User")
    st.markdown(
        "Use the camera to capture a clear face image and register the user with a secure embedding. "
        "For best results, stay centered, allow camera access, and keep consistent lighting."
    )

    name = st.text_input("Full Name")
    st.info("If the camera prompt does not appear, make sure your browser allows camera access for localhost.")
    img_file_buffer = st.camera_input("Open camera to take a picture", key="register_camera")

    if img_file_buffer is not None and name:
        if st.button("Register User"):
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) == 0:
                st.error("No face detected. Please retry with your face centered in the frame.")
            else:
                x, y, w, h = faces[0]
                recognizer = load_recognizer()
                face_img = recognizer.extract_face(cv2_img, [x, y, w, h])

                if face_img is not None:
                    with st.spinner("Generating embedding and saving user..."):
                        embedding = recognizer.get_embedding(face_img)
                        add_user(name, embedding)
                        st.success(f"User '{name}' registered successfully!")
                        st.balloons()
                else:
                    st.error("Unable to extract the face. Please try again.")
    elif name and img_file_buffer is None:
        st.warning("Please capture a photo to register the user.")


def is_running_with_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            return True
    except Exception:
        pass

    if any(key.startswith("STREAMLIT_") for key in os.environ):
        return True

    return "streamlit" in os.path.basename(sys.argv[0]).lower() or any(
        "streamlit" in arg.lower() for arg in sys.argv
    )


def main():
    if not is_running_with_streamlit():
        print("ERROR: Streamlit must launch this app.")
        print("Run this app with: python -m streamlit run app.py --server.port 8502 --server.address 127.0.0.1")
        sys.exit(1)

    inject_style()

    st.sidebar.title("Navigation")
    st.sidebar.markdown("Choose a section to manage attendance and users.")
    choice = st.sidebar.radio("Go to", ["Dashboard", "Register User"], index=0)

    if choice == "Dashboard":
        render_header()
        render_dashboard()
    else:
        render_add_user()


if __name__ == "__main__":
    main()
