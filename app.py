import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
 
load_dotenv()
 
st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢", layout="centered")
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about our HR policies — leave, payroll, benefits, and more.")
 
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
CORPUS_PATH  = os.getenv("CORPUS_PATH", "./hr_corpus/")
 
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "zyro-rag-challenge"
 
 
@st.cache_resource(show_spinner="Loading HR policy documents...")
def build_pipeline():
    loader    = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()
    splitter  = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    chunks    = splitter.split_documents(documents)
    emb       = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={"device": "cpu"},
                                       encode_kwargs={"normalize_embeddings": True})
    vs        = FAISS.from_documents(documents=chunks, embedding=emb)
    ret       = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
 
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        model = ChatGroq(model=LLM_MODEL, temperature=0.1, max_tokens=512)
    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1, max_output_tokens=512)
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=LLM_MODEL, temperature=0.1, max_tokens=512)
    else:
        raise ValueError("Unsupported LLM provider")
    return ret, model
 
 
retriever, llm = build_pipeline()
 
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an HR assistant for Zyro Dynamics. Answer using ONLY the provided context.
If not found, say: "I don't have information about that in our HR policies."
Be concise and professional.
 
Context:
{context}
 
Question: {question}
 
Answer:""")
 
OOS_PROMPT = ChatPromptTemplate.from_template("""
You are a classifier. Is this question HR-related (leave, payroll, benefits, policies, onboarding, etc.)?
Question: {question}
Reply ONLY "HR" or "OOS".""")
 
REFUSAL = ("I'm the Zyro Dynamics HR assistant and can only help with HR-related questions. "
           "Please ask about leave, payroll, benefits, or workplace policies.")
 
 
def format_docs(docs):
    return "\n\n".join(f"[{doc.metadata.get('source', 'Policy')}]\n{doc.page_content}" for doc in docs)
 
 
@traceable(name="ask_bot")
def ask_bot(question):
    label = StrOutputParser().invoke(llm.invoke(OOS_PROMPT.invoke({"question": question}))).strip().upper()
    if "OOS" in label:
        return {"answer": REFUSAL, "source_docs": [], "guardrail": "blocked"}
    docs    = retriever.invoke(question)
    answer  = StrOutputParser().invoke(llm.invoke(RAG_PROMPT.invoke({"context": format_docs(docs), "question": question})))
    return {"answer": answer, "source_docs": docs, "guardrail": "passed"}
 
 
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
        st.markdown(result["answer"])
        if result["source_docs"]:
            with st.expander(f"📄 Sources ({len(result['source_docs'])} chunks)"):
                for i, doc in enumerate(result["source_docs"], 1):
                    st.caption(f"Chunk {i} — {doc.metadata.get('source', 'HR Policy')}")
                    st.text(doc.page_content[:300] + "...")
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
 
with st.sidebar:
    st.header("About")
    st.info("RAG-powered HR Help Desk for Zyro Dynamics.")
    st.caption(f"Model: `{LLM_MODEL}`")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
