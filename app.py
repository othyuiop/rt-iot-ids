import requests
import os

MODEL_URL = "https://github.com/othyuiop/rt-iot-ids/releases/download/v1/model.pkl"

@st.cache_resource(show_spinner=True)
def load_model():
    if not os.path.exists("model.pkl"):
        st.info("Téléchargement du modèle ML...")
        r = requests.get(MODEL_URL)
        r.raise_for_status()
        with open("model.pkl", "wb") as f:
            f.write(r.content)
    return joblib.load("model.pkl")

model = load_model()

# Ces fichiers sont déjà dans le repo GitHub
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")
encoder_cat = joblib.load("encoder_cat.pkl")
cat_cols = joblib.load("cat_cols.pkl")
num_cols = joblib.load("num_cols.pkl")
