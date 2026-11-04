---
title: RAG Document QA
emoji: 📄
sdk: gradio
app_file: space_app.py
pinned: false
---

# RAG Document QA — perguntas e respostas sobre PDFs (LangChain + ChromaDB + Gemini)

Responde perguntas sobre um PDF usando **apenas o conteúdo do documento** e cita as páginas usadas.
Se a resposta não está no texto, diz `Não sei com base no documento.`

Caso de uso testado: **LGPD (Lei nº 13.709/2018)**.

## Como funciona

PDF → limpeza (junta palavras hifenizadas no fim da linha) → chunks de 1000 caracteres com 200 de sobreposição
→ embeddings Gemini → ChromaDB (uma coleção por PDF, identificada por hash) → busca dos 5 trechos mais próximos
→ Gemini responde só com esse contexto e indica as páginas.

Decisões que vieram de problemas reais durante o desenvolvimento:

- **Resposta vazia por `RECITATION`**: em perguntas como "bases legais", o modelo devolvia vazio ao tentar citar o texto da lei. O app tenta primeiro o prompt literal e, se vier vazio, refaz com um prompt que pede resumo com as próprias palavras.
- **Modelo reserva**: se um modelo estiver indisponível (erro 503), tenta o próximo da lista.
- **Limite do plano gratuito**: a indexação é feita em lotes, com pausas, para evitar erro 429.
- **Hifenização do PDF**: a lei traz `cin-` / `quenta` quebrado em duas linhas. Isso atrapalhou minha avaliação (a âncora "cinquenta milhões" não existia no texto extraído) e pode prejudicar a busca. A causa que eu suspeitava (chunking) estava errada; os chunks mostraram o motivo real.

## Avaliação da busca

9 perguntas sobre a LGPD, cada uma com um trecho-âncora que deve aparecer nos resultados:

| Métrica | Acertos |
|---|---|
| top-3 | 8/9 |
| top-5 | 9/9 |
| top-10 | 9/9 |

Amostra pequena: serve como verificação de sanidade, não como benchmark.

## Exemplo de saída (execução real no Colab, PDF da LGPD)

```
PERGUNTA: Qual é o valor máximo da multa simples por infração?
PÁGINAS RECUPERADAS: [35, 35, 38, 37, 36]
RESPOSTA: ... limitada, no total, a R$ 50.000.000,00 (cinquenta milhões de reais) por infração.
(páginas: 35)

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não sei com base no documento.
```

A primeira pergunta passa pelo caso da palavra hifenizada (`cin-` / `quenta`); a segunda mostra a recusa quando a resposta não está no documento.

## Como rodar

```bash
git clone <URL-DO-REPOSITORIO>
cd <PASTA>
pip install -r requirements.txt
cp .env.example .env     # coloque sua GOOGLE_API_KEY (https://aistudio.google.com/apikey)

python app.py documento.pdf -q "Quais são os direitos do titular?"
python app.py documento.pdf                      # modo interativo
python app.py documento.pdf --persist-dir .chroma  # reaproveita o índice
```

Os nomes dos modelos mudam com frequência: ajuste `EMBEDDING_MODEL` e `GEMINI_MODELS` no `.env`.
O PDF não está incluído no repositório.

## Limitações

- Plano gratuito: indexar um PDF de ~120 chunks leva cerca de 2 a 3 minutos.
- A qualidade da resposta depende do trecho recuperado; perguntas amplas podem precisar de `-k` maior.
- Avaliação pequena (9 perguntas, só LGPD).
