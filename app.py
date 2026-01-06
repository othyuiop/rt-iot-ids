import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os

# ==============================
# URLs des fichiers (GitHub Releases)
# ==============================
BASE_URL = "https://github.com/othyuiop/rt-iot-ids/releases/download/v1"

FILES = {
    "model.pkl": f"{BASE_URL}/model.pkl",
    "scaler.pkl": f"{BASE_URL}/scaler.pkl",
    "selector.pkl": f"{BASE_URL}/selector.pkl",
    "encoder_cat.pkl": f"{BASE_URL}/encoder_cat.pkl",
    "cat_cols.pkl": f"{BASE_URL}/cat_cols.pkl",
    "num_cols.pkl": f"{BASE_URL}/num_cols.pkl",
}

# ==============================
# Téléchargement + chargement
# ==============================
@st.cache_resource(show_spinner=True)
def load_artifacts():
    artifacts = {}
    for name, url in FILES.items():
        if not os.path.exists(name):
            st.info(f"Téléchargement de {name} ...")
            r = requests.get(url)
            r.raise_for_status()   # 👈 IMPORTANT
            with open(name, "wb") as f:
                f.write(r.content)
        artifacts[name] = joblib.load(name)
    return artifacts

artifacts = load_artifacts()

model = artifacts["model.pkl"]
scaler = artifacts["scaler.pkl"]
selector = artifacts["selector.pkl"]
encoder_cat = artifacts["encoder_cat.pkl"]
cat_cols = artifacts["cat_cols.pkl"]
num_cols = artifacts["num_cols.pkl"]
