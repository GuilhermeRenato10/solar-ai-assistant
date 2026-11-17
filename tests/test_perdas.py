import json
from types import SimpleNamespace

import pandas as pd
import pytest

from data.gerar_dados import gerar
from src import perdas
from src.agente import AgenteSolar, executar_ferramenta


@pytest.fixture(scope="module")
def df():
    return perdas.calcular_perdas(pd.DataFrame(gerar()).assign(data=lambda d: pd.to_datetime(d["data"])))


def test_balanco_energetico_fecha(df):
    # registrada + perdas == esperada (quando não há sobreperformance cortada pelo clip)
    d = df[df["perda_subperformance_kwh"] > 0]
    soma = d["geracao_registrada_kwh"] + d["perda_total_kwh"]
    assert (soma - d["geracao_esperada_kwh"]).abs().max() < 1e-6


def test_perdas_nao_negativas(df):
    for c in ["perda_indisponibilidade_kwh", "perda_comunicacao_kwh", "perda_subperformance_kwh"]:
        assert (df[c] >= 0).all()


def test_cenarios_injetados_sao_detectados(df):
    assert perdas.ranking_perdas(df, "indisponibilidade")["ranking"][0]["usina"] == "UFV Alfa"
    assert perdas.ranking_perdas(df, "comunicacao")["ranking"][0]["usina"] == "UFV Bravo"
    assert perdas.ranking_perdas(df, "subperformance")["ranking"][0]["usina"] == "UFV Charlie"


def test_nome_parcial_de_usina(df):
    assert perdas.resumo_usina(df, "charlie")["usina"] == "UFV Charlie"


def test_usina_inexistente_vira_erro_para_o_modelo(df):
    saida = json.loads(executar_ferramenta(df, "resumo_usina", {"usina": "Zulu"}))
    assert "erro" in saida


def test_loop_do_agente_com_cliente_falso(df):
    """Simula o modelo: 1ª resposta pede uma ferramenta, 2ª responde em texto."""
    chamada = SimpleNamespace(type="tool_use", id="t1", name="ranking_perdas", input={"tipo": "total"})
    texto = SimpleNamespace(type="text", text="A UFV Alfa teve a maior perda.")
    respostas = iter([
        SimpleNamespace(stop_reason="tool_use", content=[chamada]),
        SimpleNamespace(stop_reason="end_turn", content=[texto]),
    ])
    cliente = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: next(respostas)))

    agente = AgenteSolar(df, client=cliente)
    assert agente.perguntar("Qual usina mais perdeu?") == "A UFV Alfa teve a maior perda."
    assert agente.ultimas_chamadas == [{"ferramenta": "ranking_perdas", "argumentos": {"tipo": "total"}}]
    # histórico: user, assistant(tool_use), user(tool_result), assistant(texto)
    assert agente.historico[2]["content"][0]["type"] == "tool_result"
