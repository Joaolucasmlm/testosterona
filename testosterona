import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

# ====== Configuração do app ======
st.set_page_config(page_title="Preditor de Testosterona", layout="centered")
st.title("🧬 Preditor de Testosterona Baixa")
st.markdown("Faça o upload de um arquivo `.xlsx` com dados metabólicos para prever risco de **testosterona baixa** usando regressão logística.")

# ====== Upload de Arquivo ======
uploaded_file = st.file_uploader("📂 Envie sua planilha Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)

        # ====== Verificar colunas esperadas ======
        colunas_esperadas = ["circ_abdominal", "hdl", "triglicerideos", "pressao_sistolica", "glicemia", "testosterona"]
        if not all(col in df.columns for col in colunas_esperadas):
            st.error(f"O arquivo deve conter as colunas: {', '.join(colunas_esperadas)}")
        else:
            # ====== Variável alvo ======
            df["testo_baixa"] = (df["testosterona"] < 350).astype(int)
            X = df[["circ_abdominal", "hdl", "triglicerideos", "pressao_sistolica", "glicemia"]]
            y = df["testo_baixa"]

            # ====== Treinamento ======
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]

            # ====== Avaliação ======
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)

            # ====== Resultados ======
            st.subheader("📊 Resultados")
            st.markdown(f"**AUC (Área sob a Curva ROC):** `{auc:.3f}`")

            # ====== Gráfico ======
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"Curva ROC (AUC = {auc:.2f})")
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel("Falso-positivo")
            ax.set_ylabel("Verdadeiro-positivo")
            ax.set_title("Curva ROC - Regressão Logística")
            ax.legend(loc="lower right")
            ax.grid(True)
            st.pyplot(fig)

            # ====== Coeficientes ======
            st.subheader("📌 Importância das variáveis")
            coefs = pd.DataFrame({
                "Variável": X.columns,
                "Coeficiente": model.coef_[0]
            }).sort_values(by="Coeficiente", key=abs, ascending=False)
            st.dataframe(coefs.style.format({"Coeficiente": "{:.2f}"}))

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
else:
    st.info("Envie uma planilha Excel para iniciar.")

