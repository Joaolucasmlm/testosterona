import streamlit as st
import numpy as np

st.set_page_config(page_title="Preditor de Testosterona Baixa", layout="centered")
st.title("Preditor de Testosterona Baixa")
st.markdown("Este aplicativo estima a probabilidade de testosterona total baixa com base em fatores da síndrome metabólica, utilizando modelo de regressão logística.")

# Coeficientes da regressão logística
def calcular_probabilidade(idade, diabetes, hipertri, hipertensao, hdl_baixo, obesidade):
    beta = {
        "idade": -0.195,
        "diabetes": 0.394,
        "hipertri": 0.606,
        "hipertensao": 0.184,
        "hdl": 0.289,
        "obesidade": 1.426
    }
    intercepto = 0  # Pode ser ajustado caso disponível

    escore = (
        beta["idade"] * idade +
        beta["diabetes"] * diabetes +
        beta["hipertri"] * hipertri +
        beta["hipertensao"] * hipertensao +
        beta["hdl"] * hdl_baixo +
        beta["obesidade"] * obesidade
    ) + intercepto

    prob = 1 / (1 + np.exp(-escore))
    return prob

# Interface do usuário
st.header("Preencha os dados abaixo")

col1, col2 = st.columns(2)

with col1:
    idade = st.selectbox("Idade ≥ 60 anos?", ["Não", "Sim"])
    diabetes = st.selectbox("Diabetes mellitus tipo 2?", ["Não", "Sim"])
    hipertri = st.selectbox("Hipertrigliceridemia?", ["Não", "Sim"])

with col2:
    hipertensao = st.selectbox("Hipertensão arterial?", ["Não", "Sim"])
    hdl_baixo = st.selectbox("HDL-colesterol baixo?", ["Não", "Sim"])
    obesidade = st.selectbox("Obesidade?", ["Não", "Sim"])

# Codificar como 0 e 1
variaveis = {
    "idade": 1 if idade == "Sim" else 0,
    "diabetes": 1 if diabetes == "Sim" else 0,
    "hipertri": 1 if hipertri == "Sim" else 0,
    "hipertensao": 1 if hipertensao == "Sim" else 0,
    "hdl_baixo": 1 if hdl_baixo == "Sim" else 0,
    "obesidade": 1 if obesidade == "Sim" else 0
}

if st.button("Calcular probabilidade"):
    prob = calcular_probabilidade(**variaveis)
    st.subheader("Resultado")
    st.markdown(f"Probabilidade estimada de testosterona baixa: **{prob*100:.1f}%**")

    if prob >= 0.5:
        st.markdown("Alto risco de testosterona baixa.")
    else:
        st.markdown("Baixo risco de testosterona baixa.")
