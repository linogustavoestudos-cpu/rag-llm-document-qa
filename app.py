"""
RAG Document QA Engine (LangChain + ChromaDB + Gemini)

Responde perguntas sobre um PDF usando apenas o conteúdo do documento e
cita as páginas usadas na resposta.

Uso:
    python app.py documento.pdf -q "Quais são os direitos do titular?"
    python app.py documento.pdf                 # modo interativo
    python app.py documento.pdf --persist-dir .chroma   # reaproveita o índice
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

# ---------------------------------------------------------------------------
# Configuração (os nomes de modelo mudam com frequência: ajuste pelo .env)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
GEN_MODELS = [
    m.strip()
    for m in os.getenv(
        "GEMINI_MODELS",
        "gemini-3.1-flash-lite,gemini-flash-lite-latest,gemini-3.5-flash-lite,gemini-3.5-flash",
    ).split(",")
    if m.strip()
]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_REGRAS_COMUNS = (
    "Você é um assistente de análise de documentos.\n"
    "Responda usando EXCLUSIVAMENTE as informações dos trechos de contexto abaixo. "
    "Não use conhecimento externo e não complete lacunas com explicações próprias.\n"
    "Se o contexto apenas menciona o assunto sem explicá-lo, diga isso claramente "
    "e informe só o que está escrito.\n"
    "Se não houver nada relevante no contexto, responda exatamente: "
    "'Não sei com base no documento.'\n"
    "No final, indique as páginas usadas no formato (páginas: X, Y).\n"
)

PROMPT_LITERAL = ChatPromptTemplate.from_messages(
    [("system", _REGRAS_COMUNS + "\nContexto:\n{context}"), ("human", "{input}")]
)

# Usado quando o modelo devolve resposta vazia (finish_reason=RECITATION).
PROMPT_PARAFRASE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            _REGRAS_COMUNS
            + "Reescreva com suas próprias palavras, em formato de resumo, sem copiar o "
            "texto literalmente. Mantenha números de artigos, incisos e prazos "
            "exatamente como estão.\n\nContexto:\n{context}",
        ),
        ("human", "{input}"),
    ]
)


# ---------------------------------------------------------------------------
# Carregamento e indexação
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Junta palavras hifenizadas no fim da linha ('cin-\\nquenta' -> 'cinquenta')."""
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def load_pdf(path: str) -> list[Document]:
    """Lê o PDF página a página; 'page' nos metadados começa em 1."""
    reader = PdfReader(path)
    docs: list[Document] = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"page": number, "source": os.path.basename(path)},
                )
            )
    return docs


def pdf_digest(path: str) -> str:
    """Hash curto do arquivo: cada PDF tem sua própria coleção (evita duplicatas)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def build_vectorstore(
    chunks: list[Document],
    embeddings,
    digest: str,
    persist_dir: str | None = None,
    batch_size: int = 50,
    pause: float = 62.0,
) -> Chroma:
    """Indexa em lotes para respeitar o limite de requisições do plano gratuito."""
    store = Chroma(
        collection_name=f"rag_{digest}",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    if store._collection.count() > 0:
        print(f"Índice existente reaproveitado ({store._collection.count()} chunks).")
        return store

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        for attempt in range(4):
            try:
                store.add_documents(batch)
                break
            except Exception as exc:  # limite da API (429) ou instabilidade
                if attempt == 3:
                    raise
                print(f"  aguardando limite da API... ({str(exc)[:60]})", file=sys.stderr)
                time.sleep(30)
        print(f"{min(start + batch_size, len(chunks))}/{len(chunks)} chunks indexados")
        if start + batch_size < len(chunks):
            time.sleep(pause)
    return store


# ---------------------------------------------------------------------------
# Resposta
# ---------------------------------------------------------------------------
def format_context(chunks: list[Document]) -> str:
    return "\n\n".join(f"[página {d.metadata['page']}]\n{d.page_content}" for d in chunks)


def answer(question: str, retriever, models: list[str]) -> dict | None:
    """Tenta cada modelo; se a resposta vier vazia, tenta o prompt parafraseado."""
    chunks = retriever.invoke(question)
    pages = [d.metadata["page"] for d in chunks]
    context = format_context(chunks)

    for model_name in models:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, max_retries=2)
        for mode, prompt in (("literal", PROMPT_LITERAL), ("parafraseado", PROMPT_PARAFRASE)):
            try:
                text = (prompt | llm | StrOutputParser()).invoke(
                    {"context": context, "input": question}
                )
            except Exception as exc:
                print(f"  (falhou com {model_name}: {str(exc)[:80]})", file=sys.stderr)
                break  # passa para o próximo modelo
            if text.strip():
                return {"answer": text.strip(), "model": model_name, "mode": mode, "pages": pages}
            print(f"  (resposta vazia com {model_name} no modo {mode})", file=sys.stderr)
    return None


def show(question: str, result: dict | None) -> None:
    print(f"\nPERGUNTA: {question}")
    if result is None:
        print("Nenhum modelo conseguiu responder. Tente novamente em alguns minutos.")
        return
    print(f"PÁGINAS RECUPERADAS: {result['pages']}")
    print(f"MODELO: {result['model']} ({result['mode']})")
    print(f"RESPOSTA:\n{result['answer']}\n" + "-" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Perguntas e respostas sobre um PDF (RAG).")
    parser.add_argument("pdf", help="caminho do arquivo PDF")
    parser.add_argument("-q", "--question", action="append", help="pergunta (pode repetir)")
    parser.add_argument("-k", type=int, default=5, help="trechos recuperados (padrão: 5)")
    parser.add_argument("--persist-dir", default=None, help="pasta para guardar o índice")
    args = parser.parse_args()

    if not os.getenv("GOOGLE_API_KEY"):
        sys.exit("Defina GOOGLE_API_KEY no arquivo .env (veja .env.example).")
    if not os.path.isfile(args.pdf):
        sys.exit(f"Arquivo não encontrado: {args.pdf}")

    docs = load_pdf(args.pdf)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"{len(docs)} páginas com texto -> {len(chunks)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    store = build_vectorstore(chunks, embeddings, pdf_digest(args.pdf), args.persist_dir)
    retriever = store.as_retriever(search_kwargs={"k": args.k})

    if args.question:
        for q in args.question:
            show(q, answer(q, retriever, GEN_MODELS))
        return

    print("Digite uma pergunta (linha vazia para sair).")
    while True:
        q = input("> ").strip()
        if not q:
            break
        show(q, answer(q, retriever, GEN_MODELS))


if __name__ == "__main__":
    main()
