# 🤖 RAG Document QA Engine (LangChain + Gemini API)

Este repositório contém uma implementação prática de **RAG (Retrieval-Augmented Generation)** desenvolvida em Python. O sistema permite carregar documentos no formato PDF, realizar o processo de *chunking*, gerar embeddings vetoriais com a API do Google Gemini e responder a perguntas do utilizador com base no contexto extraído.

---

## 📌 Funcionalidades
- **Carregamento de PDFs:** Processamento automático de documentos via `PyPDFLoader`.
- **Chunking Eficiente:** Divisão inteligente de textos com `RecursiveCharacterTextSplitter`.
- **Banco de Dados Vetorial:** Armazenamento e busca por similaridade utilizando `ChromaDB`.
- **LLM & Embeddings:** Integração com os modelos `GoogleGenerativeAIEmbeddings` e `gemini-1.5-flash` via LangChain.

---

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python
* **Orquestração de LLMs:** LangChain
* **Vector Store:** ChromaDB
* **Modelos:** Google Gemini (Embeddings e LLM)

---

## 🚀 Como Executar

### 1. Clonar o repositório
```bash
git clone [https://github.com/linogustavoestudos-cpu/rag-llm-document-qa.git](https://github.com/linogustavoestudos-cpu/rag-llm-document-qa.git)
cd rag-llm-document-qa
