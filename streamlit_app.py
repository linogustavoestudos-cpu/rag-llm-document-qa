"""
Interface web (Streamlit) do RAG Document QA. Reutiliza a lógica do app.py.

Rodar localmente:   streamlit run streamlit_app.py
No Streamlit Community Cloud: informe GOOGLE_API_KEY em Settings > Secrets.
"""
import contextlib
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
MAX_MB = 10            # limites para proteger a cota gratuita de uma demo pública
MAX_CHUNKS = 250
MAX_QUESTIONS = 15     # por sessão
MAX_QUESTION_CHARS = 500
EXAMPLES = [
    "Quais são as bases legais para o tratamento de dados pessoais?",
    "Qual é o valor máximo da multa simples por infração?",
    "Quais são os direitos do titular?",
]


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


@contextlib.contextmanager
def pdf_path(uploaded):
    """PDF a usar: o enviado (arquivo temporário, apagado depois) ou o de exemplo."""
    if uploaded is None:
        yield SAMPLE_PDF
        return
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.getvalue())
    try:
        yield tmp.name
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


def render_answer(item: dict) -> None:
    # "$" abriria fórmula (LaTeX) no markdown do Streamlit: valores como "R$ 50.000.000,00" quebrariam
    st.markdown(item["a"].replace("$", "\\$"))
    st.caption(item["meta"])
    if item.get("sources"):
        with st.expander("Ver trechos usados"):
            for src in item["sources"]:
                st.markdown(f"**Página {src['page']}**")
                st.text(src["text"][:700])


def _ask_example(question: str) -> None:
    st.session_state["pending"] = question


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
            "Cada pergunta é independente (não há memória da conversa). "
            "Abra *Ver trechos usados* para conferir a fonte da resposta.\n\n"
            "Esta demo usa uma cota gratuita compartilhada. "
            "**Não envie documentos sensíveis.**"
        )

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY não configurada (Secrets do Streamlit ou arquivo .env).")
        st.stop()

    uploaded = st.file_uploader("PDF", type="pdf")
    using_sample = uploaded is None and os.path.isfile(SAMPLE_PDF)
    if uploaded is None and not using_sample:
        st.info("Envie um PDF para começar.")
        st.stop()
    if uploaded is not None and uploaded.size > MAX_MB * 1024 * 1024:
        st.error(f"Arquivo maior que {MAX_MB} MB.")
        st.stop()

    doc_id = f"upload:{uploaded.name}:{uploaded.size}" if uploaded is not None else "sample"
    if st.session_state.get("doc_id") != doc_id:  # trocou de documento: zera a conversa
        st.session_state["doc_id"] = doc_id
        st.session_state["history"] = []
    history = st.session_state["history"]

    if using_sample:
        st.info("Usando o PDF de exemplo (LGPD). Envie outro PDF para trocar.")
    else:
        st.info(f"Documento: {uploaded.name}")

    for item in history:
        with st.chat_message("user"):
            st.markdown(item["q"])
        with st.chat_message("assistant"):
            render_answer(item)

    pending = st.session_state.pop("pending", None)
    if using_sample and not history and not pending:
        st.caption("Experimente:")
        for i, example in enumerate(EXAMPLES):
            st.button(example, key=f"example_{i}", on_click=_ask_example, args=(example,))

    typed = st.chat_input("Pergunte algo sobre o documento", max_chars=MAX_QUESTION_CHARS)
    question = (pending or typed or "").strip()
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if st.session_state.get("asked", 0) >= MAX_QUESTIONS:
            st.warning(
                f"Limite de {MAX_QUESTIONS} perguntas por sessão nesta demo "
                "(para proteger a cota gratuita). Recarregue a página para continuar."
            )
            return
        st.session_state["asked"] = st.session_state.get("asked", 0) + 1
        try:
            with pdf_path(uploaded) as path:
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
        if result is None:
            st.error("Nenhum modelo conseguiu responder agora. Tente de novo em alguns minutos.")
            return

        pages = sorted(set(result["pages"]))
        item = {
            "q": question,
            "a": result["answer"],
            "meta": f"Páginas recuperadas: {pages} · Modelo: {result['model']} ({result['mode']})",
            "sources": result["sources"],
        }
        history.append(item)
        render_answer(item)


if __name__ == "__main__":
    main()
