"""
Agente LLM com tool use (function calling) para análise de usinas solares.

Fluxo:
    1. O usuário pergunta em linguagem natural.
    2. O modelo decide quais ferramentas chamar (e com quais argumentos).
    3. As ferramentas rodam em Python (src/perdas.py) sobre o DataFrame.
    4. Os resultados voltam ao modelo, que pode chamar mais ferramentas
       ou responder ao usuário com base nos números retornados.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

import pandas as pd

from src import perdas

MODELO_PADRAO = os.getenv("MODELO_LLM", "claude-haiku-5-5")
MAX_ITERACOES = 8

SYSTEM_PROMPT = """Você é um analista de performance de usinas fotovoltaicas que apoia uma equipe de O&M.

Regras:
- NUNCA invente ou calcule números de cabeça. Todo número da resposta deve vir de uma ferramenta.
- Se a pergunta for ambígua quanto ao período, use o período completo disponível e diga isso.
- Responda em português, de forma objetiva: primeiro a resposta direta, depois os números que a sustentam,
  e por fim a provável causa e uma recomendação prática para o time de campo.
- Diferencie claramente: indisponibilidade (equipamento parado), comunicação (energia gerada mas não
  registrada) e subperformance (equipamento operando abaixo do esperado, ex.: sujidade, degradação).
- Use kWh com separador de milhar e no máximo 1 casa decimal.
"""

_PROPS_PERIODO = {
    "inicio": {"type": "string", "description": "Data inicial AAAA-MM-DD (opcional)"},
    "fim": {"type": "string", "description": "Data final AAAA-MM-DD (opcional)"},
}

FERRAMENTAS: list[dict[str, Any]] = [
    {
        "name": "listar_usinas",
        "description": "Lista as usinas disponíveis, a capacidade de cada uma e o período de dados.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "resumo_usina",
        "description": (
            "Resumo de uma usina no período: geração esperada vs registrada, perdas por tipo "
            "(indisponibilidade, comunicação, subperformance), PR médio e eventos registrados."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"usina": {"type": "string"}, **_PROPS_PERIODO},
            "required": ["usina"],
        },
    },
    {
        "name": "ranking_perdas",
        "description": "Ranking das usinas por perda (kWh e % da esperada) de um tipo específico.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tipo": {"type": "string", "enum": list(perdas.TIPOS_PERDA)},
                **_PROPS_PERIODO,
            },
            "required": ["tipo"],
        },
    },
    {
        "name": "serie_diaria",
        "description": (
            "Série diária de uma usina (PR, disponibilidade, horas sem comunicação, perda, evento). "
            "Use para investigar tendências, como queda gradual de PR por sujidade."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"usina": {"type": "string"}, **_PROPS_PERIODO},
            "required": ["usina"],
        },
    },
    {
        "name": "detectar_anomalias",
        "description": "Conta, por usina, os dias com PR baixo, indisponibilidade ou falha de comunicação.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limiar_pr": {"type": "number", "description": "PR abaixo do qual o dia é anômalo (padrão 0.72)"},
                **_PROPS_PERIODO,
            },
        },
    },
]


def _registro_ferramentas(df: pd.DataFrame) -> dict[str, Callable[..., dict]]:
    return {
        "listar_usinas": lambda **kw: perdas.listar_usinas(df),
        "resumo_usina": lambda **kw: perdas.resumo_usina(df, **kw),
        "ranking_perdas": lambda **kw: perdas.ranking_perdas(df, **kw),
        "serie_diaria": lambda **kw: perdas.serie_diaria(df, **kw),
        "detectar_anomalias": lambda **kw: perdas.detectar_anomalias(df, **kw),
    }


def executar_ferramenta(df: pd.DataFrame, nome: str, argumentos: dict) -> str:
    """Executa uma ferramenta e devolve JSON. Erros viram mensagem para o modelo corrigir."""
    registro = _registro_ferramentas(df)
    if nome not in registro:
        return json.dumps({"erro": f"Ferramenta desconhecida: {nome}"})
    try:
        resultado = registro[nome](**argumentos)
    except (ValueError, TypeError) as e:
        resultado = {"erro": str(e)}
    return json.dumps(resultado, ensure_ascii=False, default=str)


class AgenteSolar:
    def __init__(self, df: pd.DataFrame, client=None, modelo: str = MODELO_PADRAO):
        if client is None:
            import anthropic  # import tardio: testes não precisam da lib/chave

            client = anthropic.Anthropic()  # lê ANTHROPIC_API_KEY do ambiente
        self.client = client
        self.df = df
        self.modelo = modelo
        self.historico: list[dict] = []
        self.ultimas_chamadas: list[dict] = []  # para mostrar na UI (transparência)

    def perguntar(self, pergunta: str) -> str:
        self.historico.append({"role": "user", "content": pergunta})
        self.ultimas_chamadas = []

        for _ in range(MAX_ITERACOES):
            resposta = self.client.messages.create(
                model=self.modelo,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                tools=FERRAMENTAS,
                messages=self.historico,
            )
            self.historico.append({"role": "assistant", "content": resposta.content})

            if resposta.stop_reason != "tool_use":
                return "".join(b.text for b in resposta.content if b.type == "text")

            resultados = []
            for bloco in resposta.content:
                if bloco.type == "tool_use":
                    saida = executar_ferramenta(self.df, bloco.name, bloco.input)
                    self.ultimas_chamadas.append({"ferramenta": bloco.name, "argumentos": bloco.input})
                    resultados.append(
                        {"type": "tool_result", "tool_use_id": bloco.id, "content": saida}
                    )
            self.historico.append({"role": "user", "content": resultados})

        return "Não consegui concluir a análise dentro do limite de etapas. Tente reformular a pergunta."
