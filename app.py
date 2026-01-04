# NOTE:
# Les fichiers du modèle (.pkl) sont chargés localement.
# Ils ne sont pas inclus sur GitHub en raison des limitations de taille.

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger les objets sauvegardés
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")
encoder_cat = joblib.load("encoder_cat.pkl")
cat_cols = joblib.load("cat_cols.pkl")
num_cols = joblib.load("num_cols.pkl")

st.title("IoT Intrusion Detection System")
st.write("Classification des types d’attaques réseau (RT-IoT2022)")

uploaded_file = st.file_uploader("Uploader un fichier CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Aperçu des données")
    st.write(df.head())

    # Séparation des colonnes
    X_cat = df[cat_cols]
    X_num = df[num_cols]

    # Encodage et normalisation
    X_cat_enc = encoder_cat.transform(X_cat)
    X_num_scaled = scaler.transform(X_num)

    # Fusion
    X_processed = np.hstack((X_num_scaled, X_cat_enc))

    # Feature selection
    X_final = selector.transform(X_processed)

    if st.button("Prédire"):
        predictions = model.predict(X_final)
        st.subheader("Résultat de la prédiction")
        st.write(predictions)
