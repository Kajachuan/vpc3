# -*- coding: utf-8 -*-

try:
    import streamlit as st
    st.set_page_config(page_title="VPC3-TP", layout="centered")
    st.title("VPC3-TP")
    st.write("Estructura base generada (Streamlit).")
    st.caption("Para ejecutar: streamlit run app/main.py")
except Exception:
    print("Streamlit no está instalado. Instalá: pip install streamlit")
    print("Luego ejecutá: streamlit run app/main.py")


