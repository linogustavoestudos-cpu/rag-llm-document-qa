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
git clone https://github.com/linogustavoestudos-cpu/rag-llm-document-qa
cd rag-llm-document-qa
pip install -r requirements.txt
cp .env.example .env     # (Windows: copy .env.example .env) e coloque sua GOOGLE_API_KEY
                         # chave em https://aistudio.google.com/apikey

# linha de comando
python app.py documento.pdf -q "Quais são os direitos do titular?"
python app.py documento.pdf                        # modo interativo
python app.py documento.pdf --persist-dir .chroma  # reaproveita o índice

# interface web (Streamlit, precisa de Python 3.11+)
streamlit run streamlit_app.py
```

Os nomes dos modelos mudam com frequência: ajuste `EMBEDDING_MODEL` e `GEMINI_MODELS` no `.env`.
Para a interface web usar um PDF de exemplo automaticamente, coloque-o em `data/lgpd.pdf`.

## Publicar a interface no Streamlit Community Cloud

1. Em share.streamlit.io, conecte sua conta do GitHub e escolha este repositório, o branch `main` e o arquivo `streamlit_app.py`.
2. Em *Advanced settings*, escolha Python 3.12 ou 3.13 e, em *Secrets*, cole: `GOOGLE_API_KEY = "sua_chave"`.
3. Clique em *Deploy*. A chave fica só nas configurações do Streamlit, nunca no repositório.

## Limitações

- Plano gratuito do Gemini: indexar um PDF de ~120 chunks leva cerca de 2 a 3 minutos.
- A demo pública compartilha uma cota gratuita: há limite de tamanho (10 MB / 250 trechos) e ela pode ficar indisponível se a cota acabar.
- A qualidade da resposta depende do trecho recuperado; perguntas amplas podem precisar de `-k` maior.
- O "º" do PDF às vezes sai como letra solta na extração (ex.: "Art. 7 o").
- Avaliação pequena (9 perguntas, só LGPD).
