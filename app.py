import os
import hashlib
import tempfile

import streamlit as st

from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_pinecone import PineconeVectorStore


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GMRIT RAG Chatbot",
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_NAME = "gmrit-rag"
TOP_K = 4
CHUNK_SIZE = 750
CHUNK_OVERLAP = 150


# ============================================================
# API KEYS FROM STREAMLIT SECRETS
# ============================================================

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

except Exception:
    st.error(
        "API keys are not configured. "
        "Add GOOGLE_API_KEY and PINECONE_API_KEY "
        "in Streamlit Cloud → Settings → Secrets."
    )
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🎓 GMRIT Information RAG Chatbot")

st.write(
    "Ask questions about GMRIT using a "
    "Retrieval-Augmented Generation system."
)

st.caption(
    "Gemini 3.8 Flash + Gemini Embeddings + Pinecone"
)


# ============================================================
# GEMINI EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )


# ============================================================
# GEMINI LLM
# ============================================================

@st.cache_resource
def get_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
        max_output_tokens=700,
    )


# ============================================================
# PINECONE
# ============================================================

@st.cache_resource
def get_pinecone():

    return Pinecone(
        api_key=PINECONE_API_KEY
    )


# ============================================================
# CHECK PINECONE INDEX
# ============================================================

def check_pinecone_index():

    pc = get_pinecone()

    existing_indexes = [
        index["name"]
        for index in pc.list_indexes()
    ]

    if INDEX_NAME not in existing_indexes:

        st.error(
            f"Pinecone index '{INDEX_NAME}' was not found."
        )

        st.stop()

    return pc


# ============================================================
# LOAD PDF
# ============================================================

def load_pdf(pdf_bytes):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp:

        temp.write(pdf_bytes)
        pdf_path = temp.name

    try:

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

    finally:

        if os.path.exists(pdf_path):
            os.unlink(pdf_path)

    return documents


# ============================================================
# SPLIT DOCUMENT
# ============================================================

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    return splitter.split_documents(documents)


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vector_store(pdf_bytes):

    file_hash = hashlib.md5(
        pdf_bytes
    ).hexdigest()

    namespace = (
        f"gmrit-{file_hash[:12]}"
    )

    check_pinecone_index()

    embeddings = get_embeddings()

    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=namespace,
        pinecone_api_key=PINECONE_API_KEY,
    )

    if (
        st.session_state.get("indexed_hash")
        != file_hash
    ):

        documents = load_pdf(
            pdf_bytes
        )

        chunks = split_documents(
            documents
        )

        with st.spinner(
            f"Processing {len(chunks)} document chunks..."
        ):

            vector_store.add_documents(
                chunks
            )

        st.session_state[
            "indexed_hash"
        ] = file_hash

        st.success(
            f"Document processed successfully "
            f"({len(chunks)} chunks)."
        )

    return vector_store


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📄 Upload GMRIT Information Profile PDF",
    type=["pdf"],
)


if uploaded_file is None:

    st.info(
        "Upload the GMRIT Information Profile PDF "
        "to start chatting."
    )

    st.stop()


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

try:

    pdf_bytes = uploaded_file.getvalue()

    vector_store = create_vector_store(
        pdf_bytes
    )

except Exception as e:

    st.error(
        f"Error while processing PDF:\n\n{e}"
    )

    st.stop()


# ============================================================
# RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": TOP_K
    }
)


# ============================================================
# LLM
# ============================================================

llm = get_llm()


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY OLD MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📖 Retrieved Sources"
            ):

                for source in message["sources"]:

                    st.markdown(
                        f"**Page {source['page']}**  \n"
                        f"{source['preview']}"
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about GMRIT..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the GMRIT document..."
        ):

            try:

                # RETRIEVE DOCUMENTS

                docs = retriever.invoke(
                    question
                )

                # BUILD CONTEXT

                context_parts = []

                for doc in docs:

                    page = (
                        doc.metadata.get(
                            "page",
                            0
                        ) + 1
                    )

                    context_parts.append(
                        f"[Page {page}]\n"
                        f"{doc.page_content}"
                    )

                context = "\n\n".join(
                    context_parts
                )

                # RAG PROMPT

                prompt = f"""
You are an AI assistant for GMRIT.

Answer the user's question ONLY using
the provided GMRIT document context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not available in
the context, say exactly:

"I don't know based on the provided GMRIT document."

4. Keep the answer clear and concise.
5. Use bullet points when appropriate.

GMRIT DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

                # GENERATE

                response = llm.invoke(
                    prompt
                )

                content = response.content

                if isinstance(content, str):

                    answer = content

                elif isinstance(content, list):

                    parts = []

                    for item in content:

                        if isinstance(item, str):

                            parts.append(item)

                        elif isinstance(item, dict):

                            if "text" in item:

                                parts.append(
                                    item["text"]
                                )

                    answer = "\n".join(parts)

                else:

                    answer = str(content)

                answer = answer.strip()

                # SOURCES

                sources = []

                seen_pages = set()

                for doc in docs:

                    page = (
                        doc.metadata.get(
                            "page",
                            0
                        ) + 1
                    )

                    if page in seen_pages:
                        continue

                    seen_pages.add(page)

                    preview = " ".join(
                        doc.page_content.split()
                    )

                    sources.append(
                        {
                            "page": page,
                            "preview": (
                                preview[:300]
                                + (
                                    "..."
                                    if len(preview) > 300
                                    else ""
                                )
                            )
                        }
                    )

                # DISPLAY ANSWER

                st.markdown(answer)

                # DISPLAY SOURCES

                if sources:

                    with st.expander(
                        "📖 Retrieved Sources"
                    ):

                        for source in sources:

                            st.markdown(
                                f"**Page {source['page']}**  \n"
                                f"{source['preview']}"
                            )

                # SAVE

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )

            except Exception as e:

                st.error(
                    f"Error generating answer:\n\n{e}"
                )
