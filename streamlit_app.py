"""
Interface web (Streamlit) do RAG Document QA. Reutiliza a lógica do app.py.

Rodar localmente:   streamlit run streamlit_app.py
No Streamlit Community Cloud: informe GOOGLE_API_KEY em Settings > Secrets.
"""
import os
import tempfile
import threading

import streamlit as st
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

SAMPLE_PDF = "data/lgpd.pdf"  # opcional: se existir, é usado quando nenhum PDF é enviado
MAX_MB = 10        # limites para proteger a cota gratuita de uma demo pública
MAX_CHUNKS = 250


@st.cache_resource(show_spinner=False)
def _state() -> dict:
    """Índices já prontos (compartilhados entre visitantes) e um cadeado contra indexação dupla."""
    return {"retrievers": {}, "lock": threading.Lock()}


def _load_key() -> None:
    if os.getenv("GOOGLE_API_KEY"):
        return
    try:
        key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:  # sem arquivo de secrets
        key = None
    if key:
        os.environ["GOOGLE_API_KEY"] = key


def get_retriever(path: str, on_progress=None):
    state = _state()
    digest = pdf_digest(path)
    with state["lock"]:
        if digest in state["retrievers"]:
            return state["retrievers"][digest]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(load_pdf(path))
        if not chunks:
            raise ValueError("Não encontrei texto neste PDF (pode ser um PDF escaneado).")
        if len(chunks) > MAX_CHUNKS:
            raise ValueError(
                f"PDF grande demais para esta demo ({len(chunks)} trechos; limite {MAX_CHUNKS})."
            )
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        store = build_vectorstore(chunks, embeddings, digest, on_progress=on_progress)
        state["retrievers"][digest] = store.as_retriever(search_kwargs={"k": 5})
        return state["retrievers"][digest]


def main() -> None:
    st.set_page_config(page_title="RAG Document QA", page_icon="📄")
    _load_key()

    st.title("📄 RAG Document QA")
    st.caption("LangChain + ChromaDB + Gemini · responde só com base no PDF e cita as páginas")
    with st.sidebar:
        st.markdown(
            "**Como usar**\n\n1. Envie um PDF (ou use o exemplo, se houver).\n"
            "2. Escreva a pergunta.\n3. A primeira pergunta indexa o documento "
            "(2 a 3 minutos no plano gratuito do Gemini).\n\n"
            "Esta demo usa uma cota gratuita compartilhada. "
            "**Não envie documentos sensíveis.**"
        )

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY não configurada (Secrets do Streamlit ou arquivo .env).")
        st.stop()

    uploaded = st.file_uploader("PDF", type="pdf")
    path, temp_path = None, None
    if uploaded is not None:
        if uploaded.size > MAX_MB * 1024 * 1024:
            st.error(f"Arquivo maior que {MAX_MB} MB.")
            st.stop()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getvalue())
        path = temp_path = tmp.name
    elif os.path.isfile(SAMPLE_PDF):
        path = SAMPLE_PDF
        st.info("Usando o PDF de exemplo (LGPD). Envie outro PDF para trocar.")
    else:
        st.info("Envie um PDF para começar.")
        st.stop()

    with st.form("qa"):
        question = st.text_input(
            "Pergunta", placeholder="Quais são as bases legais para o tratamento de dados pessoais?"
        )
        submitted = st.form_submit_button("Perguntar")

    if not submitted:
        return
    if not question.strip():
        st.warning("Digite uma pergunta.")
        return

    try:
        bar = st.progress(0.0, text="Preparando o documento (a primeira vez demora)...")
        retriever = get_retriever(
            path,
            on_progress=lambda done, total: bar.progress(
                done / total, text=f"Indexando: {done}/{total} trechos"
            ),
        )
        bar.empty()
        with st.spinner("Consultando o Gemini..."):
            result = answer(question, retriever, GEN_MODELS)
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(
            "Não consegui concluir agora (provável limite da API). "
            f"Tente de novo em alguns minutos. ({type(exc).__name__})"
        )
        return
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    if result is None:
        st.error("Nenhum modelo conseguiu responder agora. Tente de novo em alguns minutos.")
        return
    st.markdown(result["answer"])
    st.caption(f"Páginas recuperadas: {result['pages']} · Modelo: {result['model']} ({result['mode']})")


if __name__ == "__main__":
    main()
