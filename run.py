import importlib
import os
import sys

REQUIRED_MODULES = [
    "streamlit",
    "cv2",
    "numpy",
    "pandas",
    "plotly",
    "face_recognition",
    "ultralytics",
]

missing = []
for module in REQUIRED_MODULES:
    try:
        importlib.import_module(module)
    except ImportError:
        missing.append(module)

if missing:
    print("ERROR: Missing required Python packages:", ", ".join(missing))
    print("Install dependencies with: python3.13 -m pip install -r requirements.txt")
    sys.exit(1)

import socket
import subprocess
import time
import webbrowser


def find_free_port(start_port=8502, max_port=8602):
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start_port}-{max_port}")


def wait_for_port(host, port, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((host, port))
                return True
            except OSError:
                time.sleep(0.4)
    return False


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    python_executable = sys.executable

    port = find_free_port(8502, 8602)
    url = f"http://127.0.0.1:{port}"
    cmd = [
        python_executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "false",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
    ]

    print(f"Starting Streamlit on {url}...")
    proc = subprocess.Popen(cmd)

    if wait_for_port("127.0.0.1", port, timeout=20.0):
        print(f"Streamlit running at {url}")
        webbrowser.open(url)
    else:
        print("ERROR: Streamlit did not start in time. Check for issues in the terminal output.")

    proc.wait()
