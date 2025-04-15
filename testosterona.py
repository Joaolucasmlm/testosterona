import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

# Título
st.set_page_config(page_title="Preditor de Testosterona Baixa", layout="centered")
st.title("🔬 Preditor de Testosterona Baixa com Vários Modelos")

# Upload da planilha
st.sidebar.header("📂 Carregar Planilha Excel")
arquivo = st.sidebar.file_uploader("Escolha o arquivo .xlsx", type=["xlsx"])

if arquivo:
    df = pd.read_excel(arquivo)
    st.success("Arquivo carregado com sucesso!")

    # Pré-processamento
    df["testo_baixa"] = (df["testosterona"] < 350).astype(int)
    df["tg_hdl_ratio"] = df["triglicerideos"] / df["hdl"]

    feature_cols = ["circ_abdominal", "hdl", "triglicerideos", "pressao_sistolica", "glicemia", "tg_hdl_ratio"]
    X = df[feature_cols]
    y = df["testo_baixa"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(score_func=f_classif, k=4)
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = selector.get_support(indices=True)
    selected_names = X.columns[selected_features]

    # Divisão
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    # Modelos
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)

    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)

    # ROC
    y_rf = rf_model.predict_proba(X_test)[:, 1]
    y_log = log_model.predict_proba(X_test)[:, 1]
    y_xgb = xgb_model.predict_proba(X_test)[:, 1]

    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_rf)
    fpr_log, tpr_log, _ = roc_curve(y_test, y_log)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_xgb)

    auc_rf = roc_auc_score(y_test, y_rf)
    auc_log = roc_auc_score(y_test, y_log)
    auc_xgb = roc_auc_score(y_test, y_xgb)

    # Plot ROC
    st.subheader("📈 Curva ROC - Comparação de Modelos")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.2f})")
    ax.plot(fpr_log, tpr_log, linestyle="--", label=f"Regressão Logística (AUC = {auc_log:.2f})")
    ax.plot(fpr_xgb, tpr_xgb, linestyle=":", label=f"XGBoost (AUC = {auc_xgb:.2f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    ax.set_xlabel("Falso-positivo")
    ax.set_ylabel("Verdadeiro-positivo")
    ax.set_title("Curva ROC - Modelos")
    ax.legend(loc="lower right")
    ax.grid(True)
    st.pyplot(fig)

    # Interface de predição
    st.header("📋 Inserir dados clínicos de um novo paciente")
    circ_abdominal = st.number_input("Circunferência abdominal (cm)", value=102)
    hdl = st.number_input("HDL (mg/dL)", value=40.0)
    triglicerideos = st.number_input("Triglicerídeos (mg/dL)", value=150.0)
    pressao_sistolica = st.number_input("Pressão sistólica (mmHg)", value=130.0)
    glicemia = st.number_input("Glicemia (mg/dL)", value=100.0)

    if st.button("🔍 Prever Testosterona"):
        tg_hdl_ratio = triglicerideos / hdl
        entrada = pd.DataFrame([[circ_abdominal, hdl, triglicerideos, pressao_sistolica, glicemia, tg_hdl_ratio]],
                               columns=feature_cols)

        entrada_scaled = scaler.transform(entrada)
        entrada_selected = selector.transform(entrada_scaled)

        prob = rf_model.predict_proba(entrada_selected)[0][1]
        pred = rf_model.predict(entrada_selected)[0]

        if pred == 1:
            st.error(f"🔴 Predição: Testosterona Baixa (< 350 ng/dL)")
        else:
            st.success(f"🟢 Predição: Testosterona Normal (≥ 350 ng/dL)")

        st.info(f"Probabilidade de testosterona baixa (Random Forest): {prob:.2%}")
else:
    st.warning("Por favor, carregue um arquivo Excel contendo os dados clínicos dos pacientes.")
