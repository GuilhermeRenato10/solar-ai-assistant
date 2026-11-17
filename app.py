"""
Interface Streamlit: painel de perdas + chat com o agente.

    streamlit run app.py
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src import perdas
from src.agente import AgenteSolar

load_dotenv()
st.set_page_config(page_title="Assistente de Usinas Solares", page_icon="☀️", layout="wide")

CSV = Path(__file__).parent / "data" / "geracao_usinas.csv"
if not CSV.exists():
    from data.gerar_dados import gerar

    gerar().to_csv(CSV, index=False)


@st.cache_data
def dados() -> pd.DataFrame:
    return perdas.carregar(CSV)


df = dados()

st.title("☀️ Assistente de Análise de Usinas Solares")
st.caption(
    "Agente LLM com tool use: o modelo decide quais análises rodar, o Python calcula os números. "
    "Dados 100% sintéticos."
)

aba_painel, aba_chat = st.tabs(["📊 Painel de perdas", "💬 Pergunte ao agente"])

with aba_painel:
    rank = pd.DataFrame(perdas.ranking_perdas(df, "total")["ranking"])
    por_tipo = (
        df.groupby("usina")[
            ["perda_indisponibilidade_kwh", "perda_comunicacao_kwh", "perda_subperformance_kwh"]
        ]
        .sum()
        .rename(columns=lambda c: c.replace("perda_", "").replace("_kwh", ""))
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Geração registrada", f"{df['geracao_registrada_kwh'].sum() / 1000:,.0f} MWh")
    c2.metric("Perda total", f"{df['perda_total_kwh'].sum() / 1000:,.1f} MWh")
    c3.metric("Usina com maior perda", rank.iloc[0]["usina"], f"{rank.iloc[0]['perda_pct']:.1f}% da esperada",
              delta_color="inverse")

    st.subheader("Perdas por tipo (kWh)")
    st.bar_chart(por_tipo)

    st.subheader("PR diário por usina")
    st.line_chart(df.pivot(index="data", columns="usina", values="pr_real"))

with aba_chat:
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.warning("Defina ANTHROPIC_API_KEY no arquivo .env (ou nos Secrets do Streamlit Cloud) para usar o chat.")
        st.stop()

    if "agente" not in st.session_state:
        st.session_state.agente = AgenteSolar(df)
        st.session_state.mensagens = []

    exemplos = [
        "Qual usina mais perdeu energia no período e por quê?",
        "A UFV Charlie está com algum problema de performance? Mostre a tendência.",
        "Quanto a UFV Bravo deixou de registrar por falha de comunicação em setembro?",
    ]
    st.write("Exemplos: " + " · ".join(f"*{e}*" for e in exemplos))

    for m in st.session_state.mensagens:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("chamadas"):
                with st.expander("Ferramentas chamadas pelo agente"):
                    st.json(m["chamadas"])

    if pergunta := st.chat_input("Pergunte sobre as usinas..."):
        st.session_state.mensagens.append({"role": "user", "content": pergunta})
        with st.chat_message("user"):
            st.markdown(pergunta)
        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                agente = st.session_state.agente
                resposta = agente.perguntar(pergunta)
            st.markdown(resposta)
            with st.expander("Ferramentas chamadas pelo agente"):
                st.json(agente.ultimas_chamadas)
        st.session_state.mensagens.append(
            {"role": "assistant", "content": resposta, "chamadas": agente.ultimas_chamadas}
        )
