import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

# ── Load .env FIRST so all os.getenv() calls below see the values ──────────────
load_dotenv()

# ── LangSmith tracing (requires LANGCHAIN_API_KEY in .env) ─────────────────────
# FIX 1: These must be set AFTER load_dotenv() so the API key is already in env.
#         Previously they were set before dotenv could populate LANGCHAIN_API_KEY,
#         causing silent tracing failures.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "zyro-rag-challenge"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about our HR policies — leave, payroll, benefits, and more.")

# ── Runtime config from environment (or sensible defaults) ────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL    = os.getenv("LLM_MODEL",    "llama-3.3-70b-versatile")
CORPUS_PATH  = os.getenv("CORPUS_PATH",  "./hr_corpus/")


# ── Pipeline builder (cached so it only runs once per session) ─────────────────
@st.cache_resource(show_spinner="Loading HR policy documents...")
def build_pipeline():
    # Load + split documents
    loader    = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()
    splitter  = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    chunks = splitter.split_documents(documents)

    # Embeddings + vector store
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs  = FAISS.from_documents(documents=chunks, embedding=emb)
    ret = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # LLM — provider-aware
    # FIX 2: Groq's langchain wrapper accepts `model` (not `model_name`).
    #         All three providers are handled cleanly here.
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        model = ChatGroq(model=LLM_MODEL, temperature=0.1, max_tokens=512)
    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = ChatGoogleGenerativeAI(
            model=LLM_MODEL, temperature=0.1, max_output_tokens=512
        )
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=LLM_MODEL, temperature=0.1, max_tokens=512)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{LLM_PROVIDER}'. Use groq | gemini | openai.")

    return ret, model


retriever, llm = build_pipeline()

# ── Prompts ────────────────────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an HR assistant for Zyro Dynamics. Answer using ONLY the provided context.
If the answer is not in the context, say: "I don't have information about that in our HR policies."
Be concise and professional.

Context:
{context}

Question: {question}

Answer:""")

OOS_PROMPT = ChatPromptTemplate.from_template("""
You are a classifier. Is the following question related to HR topics such as \
leave, payroll, benefits, policies, onboarding, performance, or workplace conduct?
Question: {question}
Reply ONLY with "HR" or "OOS". No explanation.""")

REFUSAL = (
    "I'm the Zyro Dynamics HR assistant and can only help with HR-related questions. "
    "Please ask about leave, payroll, benefits, or workplace policies."
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def format_docs(docs: list) -> str:
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'HR Policy')}]\n{doc.page_content}"
        for doc in docs
    )


# ── Core RAG function (LangSmith-traced) ───────────────────────────────────────
@traceable(name="ask_bot")
def ask_bot(question: str) -> dict:
    """
    Returns {"answer": str, "source_docs": list, "guardrail": str}

    FIX 3: Previously used llm.invoke(PROMPT.invoke({...})) which passes a
           ChatPromptValue directly to the LLM — this works but skips the
           output parser, so `label` was an AIMessage object, not a string.
           The `.strip().upper()` call on an AIMessage silently returned the
           repr of the object, meaning "OOS" was never detected.
           Fixed by using the proper LCEL pipe chain: prompt | llm | parser.

    FIX 4: Wrapped in try/except so LLM failures show a friendly message
           instead of crashing the whole Streamlit app.
    """
    try:
        # Guardrail: classify question as HR-related or out-of-scope
        label = (
            OOS_PROMPT
            | llm
            | StrOutputParser()
        ).invoke({"question": question})

        label = label.strip().upper()

        if "OOS" in label:
            return {"answer": REFUSAL, "source_docs": [], "guardrail": "blocked"}

        # RAG: retrieve relevant chunks then generate answer
        docs = retriever.invoke(question)

        answer = (
            RAG_PROMPT
            | llm
            | StrOutputParser()
        ).invoke({
            "context":  format_docs(docs),
            "question": question,
        })

        return {"answer": answer, "source_docs": docs, "guardrail": "passed"}

    except Exception as exc:
        error_msg = f"⚠️ Something went wrong while fetching the answer: {exc}"
        return {"answer": error_msg, "source_docs": [], "guardrail": "error"}


# ── Chat UI ────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
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


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.info("RAG-powered HR Help Desk for Zyro Dynamics.")
    st.caption(f"**Provider:** `{LLM_PROVIDER}`")
    st.caption(f"**Model:** `{LLM_MODEL}`")
    st.caption(f"**Corpus:** `{CORPUS_PATH}`")

    # FIX 5: Resets full session state (not just messages) to avoid stale state
    if st.button("🗑️ Clear chat"):
        st.session_state.clear()
        st.rerun()
