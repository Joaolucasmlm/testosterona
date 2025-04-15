import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns

st.set_page_config(page_title="Preditor de Testosterona Baixa", layout="centered")
st.title("🧬 Preditor de Testosterona Baixa")

st.markdown("Faça upload de uma planilha Excel com os seguintes campos:\n\n- circ_abdominal\n- hdl\n- triglicerideos\n- pressao_sistolica\n- glicemia\n- testosterona")

uploaded_file = st.file_uploader("📂 Envie sua planilha Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        colunas_esperadas = ["circ_abdominal", "hdl", "triglicerideos", "pressao_sistolica", "glicemia", "testosterona"]

        if not all(col in df.columns for col in colunas_esperadas):
            st.error(f"O arquivo deve conter as colunas: {', '.join(colunas_esperadas)}")
        else:
            df["testo_baixa"] = (df["testosterona"] < 350).astype(int)
            df["tg_hdl_ratio"] = df["triglicerideos"] / df["hdl"]

            X = df[["circ_abdominal", "hdl", "triglicerideos", "pressao_sistolica", "glicemia", "tg_hdl_ratio"]]
            y = df["testo_baixa"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)

            st.subheader("📊 Resultados do Modelo")
            st.markdown(f"**AUC:** `{auc:.3f}`")
            st.markdown(f"**Melhores Parâmetros:** `{grid.best_params_}`")

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"ROC (AUC = {auc:.2f})")
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_xlabel("Falso-positivo")
            ax_roc.set_ylabel("Verdadeiro-positivo")
            ax_roc.set_title("Curva ROC")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

            st.subheader("📌 Matriz de Confusão")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Baixa"], yticklabels=["Normal", "Baixa"])
            ax_cm.set_xlabel("Predito")
            ax_cm.set_ylabel("Real")
            st.pyplot(fig_cm)

            st.subheader("📄 Relatório de Classificação")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"))

            st.subheader("🔍 Importância das Variáveis")
            importancias = pd.DataFrame({
                "Variavel": X.columns,
                "Importancia": best_model.feature_importances_
            }).sort_values(by="Importancia", ascending=False)
            st.dataframe(importancias.style.format({"Importancia": "{:.3f}"}))

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
else:
    st.info("Envie um arquivo Excel com os dados para iniciar.")
