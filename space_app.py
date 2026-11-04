"""Interface web (Gradio) para o Hugging Face Space. Reutiliza a lógica do app.py."""
import os

import gradio as gr
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    GEN_MODELS,
    answer,
    build_vectorstore,
    load_pdf,
    pdf_digest,
)

DEFAULT_PDF = "data/lgpd.pdf"  # opcional: se existir, é usado quando nenhum PDF é enviado
_retrievers: dict = {}


def get_retriever(path: str):
    digest = pdf_digest(path)
    if digest not in _retrievers:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(load_pdf(path))
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        store = build_vectorstore(chunks, embeddings, digest)
        _retrievers[digest] = store.as_retriever(search_kwargs={"k": 5})
    return _retrievers[digest]


def ask(pdf_path, question):
    if not os.getenv("GOOGLE_API_KEY"):
        return "Chave GOOGLE_API_KEY não configurada nos Secrets do Space."
    if not question or not question.strip():
        return "Digite uma pergunta."
    path = pdf_path or (DEFAULT_PDF if os.path.isfile(DEFAULT_PDF) else None)
    if not path:
        return "Envie um PDF."
    result = answer(question, get_retriever(path), GEN_MODELS)
    if result is None:
        return "Nenhum modelo conseguiu responder agora. Tente de novo em alguns minutos."
    return f"{result['answer']}\n\nPáginas recuperadas: {result['pages']}"


demo = gr.Interface(
    fn=ask,
    inputs=[
        gr.File(label="PDF (a primeira pergunta indexa o documento: leva alguns minutos)",
                type="filepath", file_types=[".pdf"]),
        gr.Textbox(label="Pergunta", placeholder="Quais são as bases legais para o tratamento de dados pessoais?"),
    ],
    outputs=gr.Textbox(label="Resposta"),
    title="RAG Document QA (LangChain + ChromaDB + Gemini)",
    description="Responde só com base no PDF e cita as páginas. Se não estiver no documento, diz que não sabe.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
