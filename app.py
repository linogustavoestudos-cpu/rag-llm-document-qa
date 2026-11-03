"""
Projeto RAG (Retrieval-Augmented Generation) para Consulta de Documentos
Desenvolvido para demonstração de integração com LLMs (LangChain + Gemini).
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Carregar variáveis de ambiente (.env)
load_dotenv()

def run_rag_pipeline(pdf_path: str, user_query: str):
    """
    Carrega um documento PDF, realiza o chunking, gera embeddings,
    armazena no banco vetorial Chroma e responde à pergunta do utilizador.
    """
    print(f"📄 Carregando o documento: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Division do texto em blocos (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Criação do Vector Store com Embeddings do Google Gemini
    print("🧠 Gerando embeddings e armazenando no ChromaDB...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Definição do Prompt do Sistema
    system_prompt = (
        "Você é um assistente especializado em análise de documentos.\n"
        "Use os seguintes trechos de contexto recuperados para responder à pergunta.\n"
        "Se você não souber a resposta, diga que não sabe com base no documento.\n"
        "Mantenha a resposta clara e objetiva.\n\n"
        "Contexto:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Inicialização da LLM (Gemini 1.5 Flash / Pro)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    # Criação e Execução da Chain de RAG
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print(f"❓ Pergunta: {user_query}")
    response = rag_chain.invoke({"input": user_query})

    print("\n💡 Resposta do Modelo:")
    print(response["answer"])

if __name__ == "__main__":
    # Exemplo de execução (necessita de chave GOOGLE_API_KEY no ficheiro .env)
    # run_rag_pipeline("exemplo_manual.pdf", "Quais são os pontos principais do documento?")
    print("Pipeline RAG pronto para execução!")
