# ☀️ Assistente de Análise de Usinas Solares (Agente LLM + Tool Use)

Agente de IA que responde perguntas em linguagem natural sobre a performance de usinas fotovoltaicas —
*"qual usina mais perdeu energia e por quê?"*, *"a UFV Charlie está com sujidade?"* — investigando os dados
por conta própria através de ferramentas Python.

> **Motivação:** no meu trabalho com dados de usinas solares, investigar perdas de geração exige cruzar
> disponibilidade, falhas de comunicação e PR usina por usina, dia por dia. Este projeto mostra como um
> agente LLM pode fazer essa primeira triagem e entregar ao time de O&M uma resposta com causa provável e
> recomendação. **Os dados são 100% sintéticos** — nenhum dado de cliente é usado.

## Como funciona

```
Pergunta do usuário
        │
        ▼
┌──────────────────┐   decide qual ferramenta chamar   ┌───────────────────────────┐
│   LLM (Claude)   │ ───────────────────────────────▶  │  Ferramentas em Python    │
│  raciocina e     │                                   │  (src/perdas.py, pandas)  │
│  interpreta      │ ◀───────────────────────────────  │  calculam os números      │
└──────────────────┘     resultados em JSON            └───────────────────────────┘
        │  (repete até ter informação suficiente)
        ▼
Resposta: diagnóstico + números + recomendação
```

**Decisão de design principal:** o LLM *nunca* calcula números. Toda a matemática fica em funções Python
determinísticas e testadas; o modelo só escolhe quais análises rodar e interpreta os resultados. Isso elimina
alucinação numérica — um requisito básico quando a resposta vai para um cliente ou para o time de campo.

### Ferramentas disponíveis para o agente

| Ferramenta | O que faz |
|---|---|
| `listar_usinas` | Usinas, capacidade e período de dados |
| `resumo_usina` | Esperada vs registrada, perdas por tipo, PR médio, eventos |
| `ranking_perdas` | Ranking por tipo de perda (indisponibilidade, comunicação, subperformance, total) |
| `serie_diaria` | Série diária para investigar tendências (ex.: PR caindo por sujidade) |
| `detectar_anomalias` | Dias com PR baixo, indisponibilidade ou falha de comunicação |

### Metodologia de perdas (simplificada)

```
esperada        = kWp × irradiação × PR_referência (0,80)
perda_indisp    = esperada × (1 − disponibilidade)
perda_com       = energia gerada mas não registrada durante falha de comunicação
perda_subperf   = (esperada − perda_indisp) − registrada − perda_com
```

### Cenários injetados nos dados sintéticos

| Usina | Cenário | O agente deve concluir |
|---|---|---|
| UFV Alfa | Inversor fora por 7 dias | Perda por **indisponibilidade** |
| UFV Bravo | Falhas recorrentes do datalogger | Energia **não registrada** (comunicação) |
| UFV Charlie | Sujidade acumulando até uma limpeza | **Subperformance** com PR caindo e recuperação |
| UFV Delta | Normal | Referência |
| UFV Echo | 1 dia de trip de rede | Indisponibilidade pontual |

Os testes (`tests/`) verificam que o cálculo detecta cada cenário e que o balanço energético fecha.

## Rodando localmente

```bash
git clone https://github.com/GuilhermeRenato10/solar-ai-assistant.git
cd solar-ai-assistant
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # coloque sua ANTHROPIC_API_KEY
python data/gerar_dados.py  # gera data/geracao_usinas.csv
pytest -q                   # 6 testes
streamlit run app.py
```

O painel de perdas funciona sem chave de API; o chat precisa dela.

## Stack

Python · Pandas · Anthropic API (tool use / function calling) · Streamlit · Pytest

## Próximos passos

- [ ] Ler dados reais via API REST (ex.: modelo semântico do Power BI) em vez de CSV
- [ ] Gerar relatório mensal automático em PDF/Excel a partir da conversa
- [ ] Avaliação automatizada do agente (conjunto de perguntas com respostas esperadas)
- [ ] Suporte a outros provedores de LLM

---

Desenvolvido por [Guilherme Macedo](https://linkedin.com/in/guilherme-macedo01) com apoio de assistente de IA
no desenvolvimento.
