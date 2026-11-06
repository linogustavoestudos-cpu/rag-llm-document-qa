# RAG Document QA — perguntas e respostas sobre PDFs (LangChain + ChromaDB + Gemini)

**Demo ao vivo:** https://rag-llm-document.streamlit.app

Responde perguntas sobre um PDF usando **apenas o conteúdo do documento** e cita as páginas usadas.
Se a resposta não está no texto, diz `Não sei com base no documento.`

Caso de uso testado: **LGPD (Lei nº 13.709/2018)**.

## O que é (e o que não é)

É uma **aplicação RAG** (Retrieval-Augmented Generation). Eu não treinei nenhum modelo: o texto é gerado pelo **Gemini via API** (família Flash-Lite, com modelos reserva) e a busca usa embeddings `gemini-embedding-001`. O trabalho do projeto está no pipeline em volta do modelo: leitura e limpeza do PDF, chunking, indexação no ChromaDB, recuperação, prompts que limitam a resposta ao documento, tratamento de falhas e avaliação da busca.

## Funcionalidades da interface

- Upload de PDF (até 10 MB) ou PDF de exemplo (LGPD) com perguntas sugeridas.
- Resposta com as páginas usadas e **"Ver trechos usados"**, que mostra os trechos recuperados para conferir a fonte.
- Recusa quando a informação não está no documento, sem inventar.
- Histórico da conversa na tela; **cada pergunta é independente** (não há memória entre perguntas).
- Limites de demo para proteger a cota gratuita: 15 perguntas por sessão e PDFs de até 250 trechos.

## Como funciona

PDF → limpeza (junta palavras hifenizadas no fim da linha) → chunks de 1000 caracteres com 200 de sobreposição
→ embeddings Gemini → ChromaDB (uma coleção por PDF, identificada por hash) → busca dos 5 trechos mais próximos
→ Gemini responde só com esse contexto e indica as páginas.

Decisões que vieram de problemas reais durante o desenvolvimento:

- **Resposta vazia por `RECITATION`**: em perguntas como "bases legais", o modelo devolvia vazio ao tentar citar o texto da lei. O app tenta primeiro o prompt literal e, se vier vazio, refaz com um prompt que pede resumo com as próprias palavras.
- **Modelo reserva**: se um modelo estiver indisponível (erro 503), tenta o próximo da lista.
- **Limite do plano gratuito**: a indexação é feita em lotes, com pausas, para evitar erro 429.
- **Índice incompleto**: se a indexação for interrompida, o índice parcial é recriado em vez de reaproveitado.
- **Hifenização do PDF**: a lei traz `cin-` / `quenta` quebrado em duas linhas. Isso atrapalhou minha avaliação (a âncora "cinquenta milhões" não existia no texto extraído) e pode prejudicar a busca. A causa que eu suspeitava (chunking) estava errada; os chunks mostraram o motivo real.

## Avaliação da busca

9 perguntas sobre a LGPD, cada uma com um trecho-âncora que deve aparecer nos resultados:

| Métrica | Acertos |
|---|---|
| top-3 | 8/9 |
| top-5 | 9/9 |
| top-10 | 9/9 |

Amostra pequena: serve como verificação de sanidade, não como benchmark.

## Exemplos de saída (PDF da LGPD)

```
PERGUNTA: Qual é o valor máximo da multa simples por infração?
RESPOSTA: ... limitada, no total, a R$ 50.000.000,00 (cinquenta milhões de reais) por infração.
(páginas: 35)

PERGUNTA: Posso criar uma IA para ajudar a população sem violar a LGPD?
RESPOSTA: O documento não fornece informações sobre a criação de inteligência artificial.
Ele menciona apenas que a LGPD dispõe sobre o tratamento de dados pessoais ... (páginas: 8, 49)

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não sei com base no documento.
```

A primeira passa pelo caso da palavra hifenizada (`cin-` / `quenta`). A segunda mostra o modelo limitado ao que o texto diz, sem opinar além dele. A terceira mostra a recusa quando a resposta não está no documento.

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

- Plano gratuito do Gemini: indexar um PDF de ~120 chunks leva cerca de 2 a 3 minutos, e o índice fica em memória (some quando o app reinicia ou "dorme").
- A demo pública compartilha uma cota gratuita: pode ficar indisponível se a cota acabar.
- Sem memória de conversa: perguntas de continuação ("e a multa?") não herdam o contexto da anterior.
- A qualidade da resposta depende do trecho recuperado; perguntas amplas podem precisar de `-k` maior.
- Só PDFs com texto (sem OCR para PDFs escaneados).
- Avaliação pequena (9 perguntas, só LGPD).
