"""
Gera um dataset SINTÉTICO de geração diária de usinas fotovoltaicas.

Nenhum dado real de cliente é usado. Os cenários injetados imitam ocorrências
comuns em O&M solar, para que o agente tenha o que investigar:

- UFV Alfa:    inversor fora por alguns dias (queda de disponibilidade)
- UFV Bravo:   falhas de comunicação recorrentes (dados não registrados)
- UFV Charlie: sujidade acumulando nos módulos (PR caindo) até uma limpeza
- UFV Delta:   operação normal (referência)
- UFV Echo:    operação normal com um dia de trip de rede

Uso:
    python data/gerar_dados.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
INICIO = "2026-08-01"
DIAS = 92  # agosto a outubro
PR_REFERENCIA = 0.80

USINAS = {
    "UFV Alfa": 1500.0,
    "UFV Bravo": 2200.0,
    "UFV Charlie": 1000.0,
    "UFV Delta": 3000.0,
    "UFV Echo": 1800.0,
}


def gerar(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    datas = pd.date_range(INICIO, periods=DIAS, freq="D")

    # Irradiação diária (kWh/m²) com leve sazonalidade + dias nublados
    base = 4.8 + 0.8 * np.sin(np.linspace(0, np.pi, DIAS))
    nublado = rng.random(DIAS) < 0.15
    irradiacao_regional = np.where(nublado, base * rng.uniform(0.35, 0.65, DIAS), base)

    linhas = []
    for usina, kwp in USINAS.items():
        for i, data in enumerate(datas):
            irr = max(0.5, irradiacao_regional[i] * rng.normal(1.0, 0.04))
            esperada = kwp * irr * PR_REFERENCIA

            disponibilidade = 1.0
            horas_falha_com = 0.0
            fator_performance = rng.normal(0.99, 0.015)  # ruído natural
            evento = ""

            if usina == "UFV Alfa" and 20 <= i <= 26:
                disponibilidade = 0.70
                evento = "Inversor INV-03 fora de operação"
            if usina == "UFV Bravo" and rng.random() < 0.18:
                horas_falha_com = float(rng.integers(2, 9))
                evento = "Falha de comunicação do datalogger"
            if usina == "UFV Charlie":
                if i < 60:
                    fator_performance -= 0.0035 * i  # sujidade acumulando
                elif i == 60:
                    evento = "Limpeza dos módulos realizada"
            if usina == "UFV Echo" and i == 45:
                disponibilidade = 0.0
                evento = "Trip de rede - concessionária"

            fator_performance = float(np.clip(fator_performance, 0.5, 1.02))
            # Energia efetivamente gerada
            gerada = esperada * disponibilidade * fator_performance
            # Energia registrada pelo sistema (perde-se o período sem comunicação;
            # assume ~12h de sol por dia)
            registrada = gerada * (1 - horas_falha_com / 12.0)

            linhas.append(
                {
                    "data": data.date().isoformat(),
                    "usina": usina,
                    "capacidade_kwp": kwp,
                    "irradiacao_kwh_m2": round(irr, 3),
                    "disponibilidade": round(disponibilidade, 3),
                    "horas_falha_comunicacao": horas_falha_com,
                    "geracao_registrada_kwh": round(registrada, 1),
                    "evento": evento,
                }
            )

    return pd.DataFrame(linhas)


if __name__ == "__main__":
    destino = Path(__file__).parent / "geracao_usinas.csv"
    df = gerar()
    df.to_csv(destino, index=False)
    print(f"{len(df)} linhas salvas em {destino}")
