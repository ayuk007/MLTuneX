"""Launch the MLTuneX Streamlit UI."""
import subprocess, sys, os

def main():
    app = os.path.join(os.path.dirname(__file__), "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app] + sys.argv[1:])

if __name__ == "__main__":
    main()
