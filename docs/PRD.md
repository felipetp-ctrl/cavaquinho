# PRD — cavaquinho v1.0

**Versão do documento:** 1.0  
**Data:** 2026-05-26  
**Status:** Rascunho

---

## 1. Contexto e Motivação

`cavaquinho` é uma biblioteca Python para detecção de alucinações de fidelidade em respostas de LLMs. A abordagem central — decompor a resposta em afirmações atômicas, verificar cada uma via NLI local e devolver rastreabilidade por afirmação — é diferenciada e resolve um problema real em pipelines RAG.

Uma análise crítica da versão 0.2.4 identificou os seguintes problemas que impedem adoção séria:

1. Convenção de naming quebrada (`caco` minúsculo viola PEP 8).
2. Lógica de agregação potencialmente falha (score diluído sem normalização).
3. Código duplicado no `Aggregator` (dois métodos quase idênticos de summary).
4. Dependências sem ceiling — compatibilidade futura em risco.
5. URLs do PyPI incorretas (repositório errado nos metadados).
6. Ausência de benchmarks de precisão publicados.
7. Ausência de documentação de API gerada.
8. Suíte de testes construída para passar no CI, não para garantir comportamento.
9. Ausência de dados de avaliação em datasets padronizados (HaluEval, FaithDial).

O objetivo deste PRD é planejar as entregas que levam o projeto do estado alpha atual até v1.0 — estável, confiável, adotável.

---

## 2. Objetivos de Produto

| Objetivo | Critério de sucesso |
|---|---|
| Qualidade de código | Zero violações de PEP 8 no CI; cobertura de testes ≥ 85% excluindo entry points |
| Correção lógica | Aggregator documentado e testado com casos de borda (0 claims, 100% contradições) |
| Confiabilidade de dependências | Todas as dependências com versão pinada em upper bound; zero breaking changes não planejados |
| Credibilidade técnica | Resultados publicados em ao menos um dataset padronizado (HaluEval ou FaithDial) |
| Descobribilidade | Metadados PyPI corretos; documentação de API navegável publicada |
| Adoção | ≥ 1 projeto externo usando a biblioteca (medido via GitHub dependency graph) |

---

## 3. Escopo — v0.3 (qualidade interna)

**Prazo sugerido:** 2–3 semanas

### 3.1 Correções de naming e estilo

- Renomear a classe `caco` para `Caco` em `core.py` e atualizar todas as referências.
- Executar `ruff` ou `flake8` no CI como gate obrigatório.
- Adicionar `ruff` às dependências de desenvolvimento.

**Arquivos afetados:** `cavaquinho/core.py`, `cavaquinho/__init__.py`, `tests/`.

### 3.2 Correção do Aggregator

- Revisar o método `aggregate` e documentar explicitamente a semântica do score (média ponderada? proporção de contradições?).
- Escrever testes parametrizados cobrindo:
  - 0 afirmações (edge case).
  - Todas as afirmações como `CONTRADICTION`.
  - Uma contradição em dez afirmações (o caso problemático identificado).
  - Threshold no limite exato.
- Se o comportamento atual for intencional, documentar o racional no docstring. Se for falha, corrigir.

**Arquivos afetados:** `cavaquinho/aggregator.py`, `tests/test_aggregator.py`.

### 3.3 Refatoração do summary no Aggregator

- Substituir `_build_summary_en` e `_build_summary_pt` por um único método com dicionário de templates por idioma.
- Manter interface pública inalterada.

**Arquivos afetados:** `cavaquinho/aggregator.py`.

### 3.4 Pinagem de dependências com upper bound

- Adicionar upper bound às dependências principais:
  - `transformers>=4.40.0,<5.0.0`
  - `torch>=2.0.0,<3.0.0`
  - `nltk>=3.8.0,<4.0.0`
- Adicionar `max_workers` explícito no `ThreadPoolExecutor` (ou expor como parâmetro de configuração).

**Arquivos afetados:** `pyproject.toml`, `cavaquinho/core.py`.

### 3.5 Correção dos metadados do PyPI

- Verificar e corrigir as URLs do `pyproject.toml` para apontar ao repositório correto.
- Adicionar `Changelog` como URL adicional.

---

## 4. Escopo — v0.4 (testes e cobertura)

**Prazo sugerido:** 2–3 semanas

### 4.1 Suíte de testes orientada a comportamento

- Remover ou justificar explicitamente qualquer exclusão de cobertura que não seja entry point ou modelo pesado.
- Escrever testes de integração leves (com modelo stub/mock) para o pipeline completo `core.py → extractor → classifier → aggregator`.
- Meta: cobertura ≥ 85% no relatório de CI sem exclusões desnecessárias.

### 4.2 Testes de regressão para casos documentados

- Para cada limitação listada no README, criar ao menos um teste que falhe se a limitação virar bug silencioso.
- Exemplos: afirmações com negação implícita, contexto vazio, resposta sem afirmações verificáveis.

---

## 5. Escopo — v0.5 (benchmarks e credibilidade)

**Prazo sugerido:** 3–4 semanas

### 5.1 Avaliação em dataset padronizado

- Implementar script de avaliação em `benchmarks/` contra HaluEval (subconjunto aberto) ou FaithDial.
- Publicar resultados no README com tabela de precisão/recall/F1 por label (`ENTAILMENT`, `NEUTRAL`, `CONTRADICTION`).
- Comparar com baseline simples (ex: BERTScore) para contextualizar os números.

### 5.2 Documentação da metodologia de avaliação

- Descrever o protocolo de avaliação (dataset, split, métricas) em `docs/evaluation.md`.
- Incluir instruções para reproduzir os resultados localmente.

---

## 6. Escopo — v1.0 (estabilidade e documentação)

**Prazo sugerido:** 2–3 semanas

### 6.1 Documentação de API gerada

- Configurar MkDocs com `mkdocstrings` (ou Sphinx) para gerar documentação a partir dos docstrings.
- Publicar via GitHub Pages.
- Cobrir ao menos: `Caco`, `Validator`, contratos abstratos (`BaseExtractor`, `BaseClassifier`, `BaseAggregator`), modelos de dados (`models.py`).

### 6.2 Guia de contribuição

- Criar `CONTRIBUTING.md` com: setup do ambiente, como rodar testes, convenções de commit, processo de PR.
- Adicionar template de issue no GitHub (bug report, feature request).

### 6.3 CHANGELOG estruturado

- Adotar formato Keep a Changelog para todas as releases a partir de v0.3.
- Retroativamente documentar as mudanças principais de v0.1 → v0.2.4.

### 6.4 Classificador como `Development Status :: 4 - Beta` → `5 - Production/Stable`

- Atualizar o classificador do PyPI somente após os critérios das seções anteriores estarem atendidos.

---

## 7. O que está fora do escopo

- Suporte a modelos de NLI além dos já integrados (DeBERTa, MiniCheck) — extensão via interface abstrata é suficiente.
- Interface gráfica ou serviço HTTP próprio.
- Suporte a Python < 3.10.
- Substituição de `torch`/`transformers` por alternativas mais leves — avaliado separadamente se houver demanda.

---

## 8. Riscos

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| Renomear `caco` → `Caco` quebra usuários existentes | Baixa (0 adoção externa confirmada) | Médio | Manter alias com `DeprecationWarning` por uma versão |
| Correção do Aggregator muda comportamento observado | Média | Alto | Testes antes da correção; bump de minor version |
| Benchmarks revelam precisão baixa | Média | Alto | Publicar resultados com contexto e comparação justa |
| `torch` lança v3 com breaking changes | Baixa | Médio | Upper bound já cobre; CI com matriz de versões |

---

## 9. Sequência de entregas recomendada

```
v0.3  →  v0.4  →  v0.5  →  v1.0
qualidade   testes   benchmarks   estabilidade
interna     robustos  publicados   documentada
```

Cada versão deve ser publicada no PyPI para manter o ritmo de releases e sinalizar progresso para potenciais adotantes.
