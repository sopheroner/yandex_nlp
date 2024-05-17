import streamlit as st
from faiss_pred import inference


st.header("Medical QA 💉")
row = st.text_input("Write your question:")

val = st.slider("How many results u want?", min_value=1, max_value=10, step=1)
inference_results = inference(row, val)

click = st.button("Result")

if click:
    for el in inference_results:
        st.write(el)
