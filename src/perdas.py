"""
Cálculo determinístico de perdas de geração.

Toda a matemática fica aqui, em Python puro, testável. O LLM nunca calcula
números: ele só chama estas funções (via tool use) e interpreta os resultados.
Isso evita alucinação numérica.

Metodologia simplificada (documentada no README):

    esperada        = kWp × irradiação × PR_referência
    perda_indisp    = esperada × (1 − disponibilidade)
    esperada_disp   = esperada − perda_indisp
    perda_com       = parcela estimada não registrada por falha de comunicação
    perda_subperf   = esperada_disp − registrada − perda_com   (mínimo 0)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PR_REFERENCIA = 0.80
HORAS_SOL = 12.0
CAMINHO_PADRAO = Path(__file__).resolve().parent.parent / "data" / "geracao_usinas.csv"

TIPOS_PERDA = ("indisponibilidade", "comunicacao", "subperformance", "total")


def carregar(caminho: str | Path = CAMINHO_PADRAO) -> pd.DataFrame:
    df = pd.read_csv(caminho, parse_dates=["data"])
    df["evento"] = df["evento"].fillna("")
    return calcular_perdas(df)


def calcular_perdas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["geracao_esperada_kwh"] = df["capacidade_kwp"] * df["irradiacao_kwh_m2"] * PR_REFERENCIA
    df["perda_indisponibilidade_kwh"] = df["geracao_esperada_kwh"] * (1 - df["disponibilidade"])
    esperada_disp = df["geracao_esperada_kwh"] - df["perda_indisponibilidade_kwh"]

    # Se houve falha de comunicação, o valor registrado cobre só (1 - h/12) do dia.
    frac_registrada = 1 - df["horas_falha_comunicacao"] / HORAS_SOL
    gerada_estimada = df["geracao_registrada_kwh"] / frac_registrada.where(frac_registrada > 0, 1)
    df["perda_comunicacao_kwh"] = (gerada_estimada - df["geracao_registrada_kwh"]).clip(lower=0)

    df["perda_subperformance_kwh"] = (
        esperada_disp - df["geracao_registrada_kwh"] - df["perda_comunicacao_kwh"]
    ).clip(lower=0)
    df["perda_total_kwh"] = (
        df["perda_indisponibilidade_kwh"]
        + df["perda_comunicacao_kwh"]
        + df["perda_subperformance_kwh"]
    )
    df["pr_real"] = df["geracao_registrada_kwh"] / (
        df["capacidade_kwp"] * df["irradiacao_kwh_m2"]
    )
    return df


def filtrar(df: pd.DataFrame, usina: str | None = None,
            inicio: str | None = None, fim: str | None = None) -> pd.DataFrame:
    out = df
    if usina:
        nomes = {u.lower(): u for u in df["usina"].unique()}
        chave = usina.lower().strip()
        if chave not in nomes:
            # aceita "alfa" para "UFV Alfa"
            candidatos = [u for k, u in nomes.items() if chave in k]
            if len(candidatos) != 1:
                raise ValueError(f"Usina '{usina}' não encontrada. Opções: {sorted(nomes.values())}")
            chave = candidatos[0].lower()
        out = out[out["usina"] == nomes[chave]]
    if inicio:
        out = out[out["data"] >= pd.Timestamp(inicio)]
    if fim:
        out = out[out["data"] <= pd.Timestamp(fim)]
    return out


def listar_usinas(df: pd.DataFrame) -> dict:
    info = df.groupby("usina")["capacidade_kwp"].first().to_dict()
    return {
        "usinas": [{"usina": u, "capacidade_kwp": k} for u, k in info.items()],
        "periodo_disponivel": {
            "inicio": df["data"].min().date().isoformat(),
            "fim": df["data"].max().date().isoformat(),
        },
    }


def resumo_usina(df: pd.DataFrame, usina: str,
                 inicio: str | None = None, fim: str | None = None) -> dict:
    d = filtrar(df, usina, inicio, fim)
    if d.empty:
        return {"erro": "Sem dados para o filtro informado."}
    esperada = d["geracao_esperada_kwh"].sum()
    eventos = d[d["evento"] != ""][["data", "evento"]]
    return {
        "usina": d["usina"].iloc[0],
        "periodo": [d["data"].min().date().isoformat(), d["data"].max().date().isoformat()],
        "geracao_esperada_kwh": round(esperada, 1),
        "geracao_registrada_kwh": round(d["geracao_registrada_kwh"].sum(), 1),
        "perdas_kwh": {
            "indisponibilidade": round(d["perda_indisponibilidade_kwh"].sum(), 1),
            "comunicacao": round(d["perda_comunicacao_kwh"].sum(), 1),
            "subperformance": round(d["perda_subperformance_kwh"].sum(), 1),
            "total": round(d["perda_total_kwh"].sum(), 1),
        },
        "perda_total_pct_da_esperada": round(100 * d["perda_total_kwh"].sum() / esperada, 2),
        "pr_medio": round(d["pr_real"].mean(), 3),
        "eventos": [
            {"data": r.data.date().isoformat(), "evento": r.evento}
            for r in eventos.itertuples()
        ][:20],
    }


def ranking_perdas(df: pd.DataFrame, tipo: str = "total",
                   inicio: str | None = None, fim: str | None = None) -> dict:
    if tipo not in TIPOS_PERDA:
        raise ValueError(f"tipo deve ser um de {TIPOS_PERDA}")
    d = filtrar(df, None, inicio, fim)
    col = f"perda_{tipo}_kwh"
    g = d.groupby("usina").agg(perda_kwh=(col, "sum"), esperada=("geracao_esperada_kwh", "sum"))
    g["perda_pct"] = 100 * g["perda_kwh"] / g["esperada"]
    g = g.sort_values("perda_kwh", ascending=False)
    return {
        "tipo": tipo,
        "ranking": [
            {"usina": u, "perda_kwh": round(r.perda_kwh, 1), "perda_pct": round(r.perda_pct, 2)}
            for u, r in g.iterrows()
        ],
    }


def serie_diaria(df: pd.DataFrame, usina: str,
                 inicio: str | None = None, fim: str | None = None) -> dict:
    d = filtrar(df, usina, inicio, fim)
    cols = ["data", "pr_real", "disponibilidade", "horas_falha_comunicacao",
            "perda_total_kwh", "evento"]
    d = d[cols].tail(62)  # limita o tamanho enviado ao modelo
    return {
        "usina": usina,
        "dias": [
            {
                "data": r.data.date().isoformat(),
                "pr": round(r.pr_real, 3),
                "disp": r.disponibilidade,
                "h_sem_com": r.horas_falha_comunicacao,
                "perda_kwh": round(r.perda_total_kwh, 1),
                "evento": r.evento,
            }
            for r in d.itertuples()
        ],
    }


def detectar_anomalias(df: pd.DataFrame, limiar_pr: float = 0.72,
                       inicio: str | None = None, fim: str | None = None) -> dict:
    """Dias com PR abaixo do limiar, indisponibilidade ou falha de comunicação."""
    d = filtrar(df, None, inicio, fim)
    mask = (d["pr_real"] < limiar_pr) | (d["disponibilidade"] < 1) | (d["horas_falha_comunicacao"] > 0)
    a = d[mask]
    resumo = a.groupby("usina").agg(
        dias_anomalos=("data", "count"),
        dias_pr_baixo=("pr_real", lambda s: int((s < limiar_pr).sum())),
        dias_indisponivel=("disponibilidade", lambda s: int((s < 1).sum())),
        dias_falha_com=("horas_falha_comunicacao", lambda s: int((s > 0).sum())),
    )
    return {
        "limiar_pr": limiar_pr,
        "por_usina": resumo.reset_index().to_dict(orient="records"),
    }
