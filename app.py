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
# CONFIGURATION
# ============================================================

INDEX_NAME = "gmrit-rag"
TOP_K = 4
CHUNK_SIZE = 750
CHUNK_OVERLAP = 150


# ============================================================
# GET STREAMLIT SECRETS
# ============================================================

def get_secret(name):

    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""

    if value:
        return value

    return os.getenv(name, "")


GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    if GOOGLE_API_KEY:
        st.success("Gemini API Key Loaded")
    else:
        st.error("Gemini API Key Missing")

    if PINECONE_API_KEY:
        st.success("Pinecone API Key Loaded")
    else:
        st.error("Pinecone API Key Missing")

    st.write("**Pinecone Index**")
    st.code(INDEX_NAME)

    st.write("**Embedding Model**")
    st.code("gemini-embedding-001")

    st.write("**LLM**")
    st.code("gemini-3.8-flash")

    st.write("**Retrieved Chunks**")
    st.code(str(TOP_K))

    st.write("**Chunk Size**")
    st.code(str(CHUNK_SIZE))

    st.write("**Chunk Overlap**")
    st.code(str(CHUNK_OVERLAP))


# ============================================================
# CHECK API KEYS
# ============================================================

if not GOOGLE_API_KEY:

    st.error(
        "GOOGLE_API_KEY is missing.\n\n"
        "Add it in Streamlit Cloud → Settings → Secrets."
    )

    st.stop()


if not PINECONE_API_KEY:

    st.error(
        "PINECONE_API_KEY is missing.\n\n"
        "Add it in Streamlit Cloud → Settings → Secrets."
    )

    st.stop()


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
# PINECONE CONNECTION
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

        st.info(
            "Create the Pinecone index with:\n\n"
            "Index Name: gmrit-rag\n"
            "Dimension: 3072\n"
            "Metric: cosine\n"
            "Cloud: AWS\n"
            "Region: us-east-1"
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

    # Create unique hash for uploaded PDF

    file_hash = hashlib.md5(
        pdf_bytes
    ).hexdigest()

    # Create unique namespace

    namespace = (
        f"gmrit-{file_hash[:12]}"
    )

    # Check Pinecone index

    check_pinecone_index()

    # Get Gemini embeddings

    embeddings = get_embeddings()

    # Connect to Pinecone

    vector_store = PineconeVectorStore(

        index_name=INDEX_NAME,

        embedding=embeddings,

        namespace=namespace,

        pinecone_api_key=PINECONE_API_KEY,
    )

    # Prevent duplicate indexing
    # during Streamlit reruns

    if (
        st.session_state.get(
            "indexed_hash"
        )
        != file_hash
    ):

        documents = load_pdf(
            pdf_bytes
        )

        chunks = split_documents(
            documents
        )

        with st.spinner(
            f"Indexing {len(chunks)} chunks..."
        ):

            vector_store.add_documents(
                chunks
            )

        st.session_state[
            "indexed_hash"
        ] = file_hash

        st.session_state[
            "chunk_count"
        ] = len(chunks)

        st.success(
            f"Successfully indexed "
            f"{len(chunks)} chunks."
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
# READ PDF
# ============================================================

pdf_bytes = uploaded_file.getvalue()


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

try:

    vector_store = create_vector_store(
        pdf_bytes
    )

except Exception as e:

    st.error(
        f"Error while indexing PDF:\n\n{e}"
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

    # ========================================================
    # USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # ========================================================
    # ASSISTANT MESSAGE
    # ========================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the GMRIT document..."
        ):

            try:

                # ====================================================
                # RETRIEVAL
                # ====================================================

                docs = retriever.invoke(
                    question
                )


                # ====================================================
                # CONTEXT
                # ====================================================

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


                # ====================================================
                # RAG PROMPT
                # ====================================================

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


                # ====================================================
                # GENERATE ANSWER
                # ====================================================

                response = llm.invoke(
                    prompt
                )

                content = response.content


                # ====================================================
                # EXTRACT TEXT
                # ====================================================

                if isinstance(
                    content,
                    str
                ):

                    answer = content

                elif isinstance(
                    content,
                    list
                ):

                    text_parts = []

                    for item in content:

                        if isinstance(
                            item,
                            str
                        ):

                            text_parts.append(
                                item
                            )

                        elif isinstance(
                            item,
                            dict
                        ):

                            if (
                                item.get("type")
                                == "text"
                            ):

                                text_parts.append(
                                    item.get(
                                        "text",
                                        ""
                                    )
                                )

                            elif "text" in item:

                                text_parts.append(
                                    item["text"]
                                )

                    answer = "\n".join(
                        text_parts
                    )

                elif isinstance(
                    content,
                    dict
                ):

                    answer = content.get(
                        "text",
                        str(content)
                    )

                else:

                    answer = str(
                        content
                    )


                answer = answer.strip()


                # ====================================================
                # SOURCES
                # ====================================================

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


                # ====================================================
                # DISPLAY ANSWER
                # ====================================================

                st.markdown(
                    answer
                )


                # ====================================================
                # DISPLAY SOURCES
                # ====================================================

                if sources:

                    with st.expander(
                        "📖 Retrieved Sources"
                    ):

                        for source in sources:

                            st.markdown(
                                f"**Page {source['page']}**  \n"
                                f"{source['preview']}"
                            )


                # ====================================================
                # SAVE MESSAGE
                # ====================================================

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