# PRD — cavaquinho v0.3.0

**Status:** Draft  
**Data:** 2026-05-26  
**Autor:** Felipe Tomé Pereira  

---

## Contexto

A v0.2.0 estabeleceu a fundação: pipeline estável, 65 testes unitários, 92.8% de cobertura, CI/CD, benchmark ASSIN2-PT e publicação no PyPI. A v0.2.1 corrigiu o publish workflow e atualizou o README no PyPI.

O benchmark HaluEval (rodado parcialmente em 50 amostras QA durante o desenvolvimento da v0.3) revelou o principal gap da versão atual:

- **Accuracy: 74%** no subset QA (baseline: 52%) — funciona
- **F1-hal (recall de alucinações): 0.698** — o modelo perde ~42% das alucinações reais
- **F1-faith: 0.772** — poucos falsos positivos

O problema central é que o pipeline é conservador demais. O threshold default (0.5) favorece precision, e o extrator não lida bem com respostas curtas (respostas de 1–2 palavras do HaluEval QA).

---

## Objetivo da v0.3.0

Melhorar o recall de alucinações sem degradar precision, através de:

1. Threshold calibrado e documentado por tipo de tarefa
2. Extração de claims mais robusta para respostas curtas
3. Benchmark HaluEval completo integrado à suite de benchmarks
4. Relatório de calibração de score (precision/recall curve)

---

## Requisitos funcionais

### RF-01 — Threshold por task type (must have)

Expor thresholds recomendados por tipo de tarefa como constante documentada:

```python
from cavaquinho import THRESHOLDS

THRESHOLDS = {
    "qa": 0.3,           # respostas curtas, recall mais crítico
    "summarization": 0.4,
    "default": 0.5,
}

validator = caco.caco(threshold=THRESHOLDS["qa"])
```

**Critério de aceitação:** F1-hal no HaluEval QA sobe de 0.698 para ≥ 0.75 com threshold 0.3.

---

### RF-02 — Extração melhorada para respostas curtas (must have)

O `RuleExtractor` atual usa `sent_tokenize`, que retorna respostas de 1–2 palavras como um único claim sem contexto semântico suficiente para o NLI.

**Solução:** adicionar lógica de expansão de claim curto — quando um claim tem menos de N tokens, concatenar com a pergunta/prompt original se disponível, ou manter como está mas ajustar o input NLI.

```python
# novo parâmetro opcional em validate()
result = validator.validate(
    response="Delhi",
    context="The Oberoi Group has its head office in Delhi.",
    query="Where is the head office of the Oberoi Group?"  # novo
)
```

O `query` é passado para o extrator para enriquecer claims curtos:  
`"Delhi"` → `"The head office is in Delhi."` (reformulação via template simples)

**Critério de aceitação:** Claims com menos de 5 tokens são expandidos com o query quando disponível. Testes unitários cobrem os casos com e sem query.

---

### RF-03 — Benchmark HaluEval integrado (must have)

Rodar o benchmark completo (10.000 amostras por subset) e adicionar os resultados ao README e ao `benchmarks/halueval_benchmark.py`.

**Critério de aceitação:**
- `python -m benchmarks.halueval_benchmark --n 10000 --subset all` roda sem erros
- Resultados documentados no README na seção Benchmark
- Tempo estimado de execução documentado no script

---

### RF-04 — Precision/Recall curve por threshold (should have)

Adicionar ao benchmark a opção de varrer thresholds e gerar a curva PR:

```bash
python -m benchmarks.halueval_benchmark --calibrate --subset qa
```

Saída: tabela threshold × precision × recall × F1, para orientar usuários na escolha.

**Critério de aceitação:** `--calibrate` varre thresholds de 0.1 a 0.9 em steps de 0.1 e imprime a tabela.

---

### RF-05 — `ValidationResult.threshold` exposto (should have)

O threshold usado na inferência não está no resultado. Usuários que inspecionam `result` não sabem qual threshold gerou o `is_hallucination`.

```python
@dataclass(frozen=True)
class ValidationResult:
    ...
    threshold: float   # novo campo
```

**Critério de aceitação:** `result.threshold` retorna o valor configurado no `caco`. Não é breaking change (campo novo em dataclass frozen → usar `field(default=0.5)`).

---

## Requisitos não-funcionais

- Nenhuma dependência nova obrigatória
- API pública (`caco`, `validate`, `ClaimResult`, `ValidationResult`, `Labels`) não tem breaking changes
- Cobertura de testes mantida em ≥ 90%
- `query` é parâmetro opcional em `validate()` — código existente não precisa mudar

---

## O que fica fora do escopo (v0.4+)

- Self-consistency via response sampling (v0.3 original do roadmap — movido para v0.4)
- Verificação factual com busca externa (Tavily, Wikipedia)
- Renomear `caco` → `Validator` com deprecation warning
- CLI: `cavaquinho validate --response X --context Y`

---

## Estimativa de esforço

| Item | Esforço estimado |
|---|---|
| RF-01 — THRESHOLDS constant + docs | 1h |
| RF-02 — query param + claim expansion | 3–4h |
| RF-03 — HaluEval benchmark completo | 2h (principalmente tempo de execução) |
| RF-04 — calibration flag | 2h |
| RF-05 — threshold no ValidationResult | 1h |
| Testes + cobertura | 2h |
| README + CHANGELOG | 1h |

**Total estimado:** ~12h de desenvolvimento

---

## Referências

- HaluEval: Li et al., 2023 — `pminervini/HaluEval` no HuggingFace
- Resultado parcial HaluEval QA (50 amostras, threshold=0.5): Acc=0.74, F1-hal=0.698, F1-faith=0.772
- ASSIN2 benchmark (v0.2.0): DeBERTa EN acc=88.2%, mDeBERTa acc=87.6%
