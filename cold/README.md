# 🔍 Fraud Detection POC — Amazon Bedrock AgentCore

POC didática de detecção de fraudes usando **Amazon Bedrock** (Claude 3 Sonnet) como agente classificador, com análise final via **Matriz de Confusão**.

---

## 📁 Estrutura

```
fraud_detection_poc/
├── generate_data.py     # Gera dados sintéticos (transactions + merchants)
├── fraud_agent.py       # Agente que classifica as transações via Bedrock
├── requirements.txt
├── data/
│   ├── transactions.json   # gerado pelo script
│   └── merchants.json      # gerado pelo script
├── confusion_matrix.png    # gerado após a execução
└── results.json            # resultados detalhados por transação
```

---

## ⚙️ Pré-requisitos

### 1. Python 3.10+
```bash
pip install -r requirements.txt
```

### 2. AWS — configurar credenciais
```bash
aws configure
```
Informe:
- `AWS Access Key ID`
- `AWS Secret Access Key`
- `Default region name`: `us-east-1`

> **Importante:** sua conta precisa ter acesso habilitado ao modelo  
> `global.anthropic.claude-sonnet-4-6` no **Amazon Bedrock**.  
> Verifique em: AWS Console → Bedrock → Model Access → habilitar Claude 3 Sonnet.

---

## 🚀 Como executar

### Passo 1 — Gerar dados sintéticos
```bash
python generate_data.py
```
Saída esperada:
```
✅  30 merchants gerados → data/merchants.json
✅  300 transações geradas → data/transactions.json
   legitimate : 142
   suspicious : 108
   fraud      : 50
```

### Passo 2 — Executar o agente de classificação
```bash
python fraud_agent.py
```

O script processa cada transação, chama o Bedrock e exibe:
```
[  1/300] T0001 | score= 15 | truth=legitimate  pred=legitimate  ✅
[  2/300] T0002 | score= 55 | truth=fraud        pred=fraud       ✅
...
```

Ao final, exibe o **classification report** e salva `confusion_matrix.png`.

---

## 📐 Regras de Negócio & Pesos

| Fator de Risco                          | Peso |
|-----------------------------------------|------|
| Valor > 5× avg_ticket do merchant       | 35   |
| Merchant classificado como high_risk    | 20   |
| Valor entre 2× e 5× avg_ticket         | 15   |
| Horário entre 01h e 04h                 | 15   |
| Horário tardio (23h, 04h, 05h)          | 10   |
| Alta velocidade: > 3 txns/1h no cartão  | 20   |
| Valor > R$500 + horário de madrugada    | 25   |

**Limiares de decisão (score /100):**
- `≥ 50` → **fraud**
- `≥ 25` → **suspicious**
- `< 25` → **legitimate**

O score calculado é enviado ao agente junto com os fatores acionados, permitindo que o LLM valide ou ajuste a classificação.

---

## 🗺️ Arquitetura da POC

```
generate_data.py
      │
      ▼
 data/*.json  ──►  fraud_agent.py
                        │
                        ├── compute_risk_score()   (regras locais)
                        │         │ score + fatores
                        ▼
               Amazon Bedrock (Claude 3 Sonnet)
                        │
                        ▼
               classification: legitimate|suspicious|fraud
                        │
                        ▼
             Matriz de Confusão + results.json
```

---

## 💡 Dicas para a aula

- Altere os `WEIGHTS` em `fraud_agent.py` e veja como a acurácia muda.
- Reduza `NUM_TRANSACTIONS` em `generate_data.py` para testes rápidos.
- Substitua `MODEL_ID` por `global.anthropic.claude-sonnet-4-6` para chamadas mais rápidas e baratas.
- Os `results.json` podem ser abertos no Excel/Pandas para análises exploratórias adicionais.
