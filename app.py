import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os

BASE_URL = "https://github.com/othyuiop/rt-iot-ids/releases/download/v1"

FILES = [
    "model.pkl",
    "scaler.pkl",
    "selector.pkl",
    "encoder_cat.pkl",
    "cat_cols.pkl",
    "num_cols.pkl"
]

@st.cache_resource(show_spinner=True)
def load_artifacts():
    artifacts = {}
    for file in FILES:
        if not os.path.exists(file):
            st.info(f"Téléchargement de {file}...")
            url = f"{BASE_URL}/{file}"
            r = requests.get(url)
            r.raise_for_status()   # 👈 très important
            with open(file, "wb") as f:
                f.write(r.content)
        artifacts[file] = joblib.load(file)
    return artifacts

artifacts = load_artifacts()

model = artifacts["model.pkl"]
scaler = artifacts["scaler.pkl"]
selector = artifacts["selector.pkl"]
encoder_cat = artifacts["encoder_cat.pkl"]
cat_cols = artifacts["cat_cols.pkl"]
num_cols = artifacts["num_cols.pkl"]
