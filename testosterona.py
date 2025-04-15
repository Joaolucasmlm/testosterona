import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from scipy.stats import randint

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

            # Pontuação tipo nomograma
            df["wc_pts"] = (df["circ_abdominal"] >= 102).astype(int)
            df["hdl_pts"] = (df["hdl"] < 40).astype(int) * 2
            df["has_pts"] = (df["pressao_sistolica"] >= 130).astype(int)
            df["tgl_pts"] = (df["triglicerideos"] >= 150).astype(int)
            df["glu_pts"] = (df["glicemia"] >= 100).astype(int)
            df["score_total"] = df[["wc_pts", "hdl_pts", "has_pts", "tgl_pts", "glu_pts"]].sum(axis=1)

            st.subheader("📏 Escore Simplificado do Paciente")
            fig_score, ax_score = plt.subplots(figsize=(10, 2))
            sns.histplot(df["score_total"], bins=range(0, 13), kde=False, ax=ax_score, color="skyblue", edgecolor="black")
            ax_score.axvline(df["score_total"].mean(), color='blue', linestyle='--', label=f"Média = {df['score_total'].mean():.1f}")
            ax_score.set_xlim(0, 12)
            ax_score.set_xlabel("Total Score")
            ax_score.set_ylabel("Frequência")
            ax_score.set_title("Distribuição do Escore Clínico (WC, HDL, HAS, TGL, Glicemia)")
            ax_score.legend()
            st.pyplot(fig_score)

            X = df[["circ_abdominal", "hdl", "triglicerideos", "pressao_sistolica", "glicemia", "tg_hdl_ratio"]]
            y = df["testo_baixa"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_selected = SelectKBest(score_func=f_classif, k='all').fit_transform(X_scaled, y)

            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

            rf = RandomForestClassifier(random_state=42)
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

            rf.fit(X_train, y_train)
            xgb.fit(X_train, y_train)

            rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
            xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])

            st.subheader("📈 AUC Individual dos Modelos")
            st.markdown(f"**RandomForest AUC:** `{rf_auc:.3f}`")
            st.markdown(f"**XGBoost AUC:** `{xgb_auc:.3f}`")

            param_dist = {
                'n_estimators': randint(100, 300),
                'max_depth': [5, 10, 15, None],
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 5)
            }
            grid = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=30, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)

            # Regressão logística com score_total
            X_score = df[["score_total"]]
            y_score = df["testo_baixa"]
            X_score_train, X_score_test, y_score_train, y_score_test = train_test_split(X_score, y_score, test_size=0.3, random_state=42)
            log_model = LogisticRegression()
            log_model.fit(X_score_train, y_score_train)
            y_log_proba = log_model.predict_proba(X_score_test)[:, 1]
            fpr_log, tpr_log, _ = roc_curve(y_score_test, y_log_proba)
            auc_log = roc_auc_score(y_score_test, y_log_proba)

            cv_score = cross_val_score(best_model, X_scaled, y, cv=5, scoring='roc_auc')

            st.subheader("📊 Resultados do Melhor Modelo (RandomForest)")
            st.markdown(f"**AUC Teste:** `{auc:.3f}`")
            st.markdown(f"**AUC Média (Validação Cruzada):** `{cv_score.mean():.3f}`")
            st.markdown(f"**Melhores Parâmetros:** `{grid.best_params_}`")

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"RandomForest ROC (AUC = {auc:.2f})")
            ax_roc.plot(fpr_log, tpr_log, label=f"Logística Escore ROC (AUC = {auc_log:.2f})", linestyle='--')
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_xlabel("Falso-positivo")
            ax_roc.set_ylabel("Verdadeiro-positivo")
            ax_roc.set_title("Curva ROC - Comparação de Modelos")
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
        st.exception(e)
else:
    st.info("Envie um arquivo Excel com os dados para iniciar.")
