import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🤖",
    layout="centered"
)
st.title("🤖 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about HR policies!")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

@st.cache_resource(show_spinner="Loading HR documents...")
def build_pipeline():
    loader = PyPDFDirectoryLoader("hr_docs/")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    rag_prompt = ChatPromptTemplate.from_template(
        "You are an HR assistant for Zyro Dynamics.\n"
        "Answer using ONLY the context below. If not found, say you don't have enough information.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    oos_prompt = ChatPromptTemplate.from_template(
        "Is this question related to HR topics like leave, salary, WFH, performance, onboarding, travel, IT security?\n"
        "Answer only yes or no.\n\nQuestion: {question}\nAnswer:"
    )
    return retriever, llm, rag_prompt, oos_prompt

retriever, llm, rag_prompt, oos_prompt = build_pipeline()

REFUSAL_MESSAGE = "I can only answer HR-related questions from Zyro Dynamics policy documents."

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

@traceable(name="zyro-ask-bot")
def ask_bot(question: str):
    classification = (oos_prompt | llm | StrOutputParser()).invoke({"question": question}).strip().lower()
    if "no" in classification:
        return {"answer": REFUSAL_MESSAGE, "sources": []}
    docs = retriever.invoke(question)
    answer = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser()).invoke(question)
    return {"answer": answer, "sources": docs}

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            result = ask_bot(question)
        answer = result["answer"]
        docs = result["sources"]
        st.markdown(answer)
        if docs:
            with st.expander("📄 Sources"):
                for i, doc in enumerate(docs, 1):
                    try:
                        source = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Source {i}:** {os.path.basename(str(source))} — Page {page}")
                        st.caption(str(doc.page_content)[:200] + "...")
                    except Exception:
                        st.caption(f"Source {i}: unavailable")
    st.session_state.messages.append({"role": "assistant", "content": answer})
