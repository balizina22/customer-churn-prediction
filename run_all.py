import subprocess
import time
import requests

# 📌 Ordre des notebooks
NOTEBOOKS = [
    "notebooks/eda.ipynb",
    "notebooks/data_prep.ipynb",
    "notebooks/ml_dl.ipynb"
]

API_URL = "http://127.0.0.1:8000/health"


def run_notebooks():
    print("\n🚀 Lancement des notebooks...\n")

    for nb in NOTEBOOKS:
        print(f"▶️ {nb}")

        result = subprocess.run([
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            nb
        ])

        if result.returncode != 0:
            print(f"❌ Erreur sur {nb}")
            exit(1)

    print("\n✅ Notebooks OK\n")


def run_api():
    print("🌐 Lancement API FastAPI...\n")

    return subprocess.Popen([
        "uvicorn",
        "api:app",
        "--reload"
    ])


def wait_for_api():
    print("⏳ Attente que l'API soit prête...")

    while True:
        try:
            r = requests.get(API_URL)
            if r.status_code == 200:
                print("✅ API prête !\n")
                break
        except:
            pass

        time.sleep(1)


def run_streamlit():
    print("📊 Lancement Streamlit...\n")

    return subprocess.Popen([
        "streamlit", "run", "appapi.py"
    ])


if __name__ == "__main__":
    run_notebooks()

    api_process = run_api()
    wait_for_api()

    streamlit_process = run_streamlit()

    print("🔥 Pipeline complet lancé\n")

    try:
        api_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Arrêt...")
        api_process.terminate()
        streamlit_process.terminate()