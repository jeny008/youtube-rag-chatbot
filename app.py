import re
from urllib.parse import parse_qs, urlparse

import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="YouTube Content Chatbot",
    page_icon="🎥",
    layout="wide",
)

st.title("🎥 YouTube Content Chatbot")
st.caption(
    "Ask questions about a YouTube video's transcript using RAG."
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_video_url" not in st.session_state:
    st.session_state.processed_video_url = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "transcript_length" not in st.session_state:
    st.session_state.transcript_length = 0


# ---------------------------------------------------------
# YouTube URL validation and ID extraction
# ---------------------------------------------------------

def extract_video_id(url: str) -> str:
    """Extract and validate the 11-character YouTube video ID."""
    parsed_url = urlparse(url.strip())
    hostname = (parsed_url.hostname or "").lower()
    video_id = ""

    if hostname == "youtu.be":
        video_id = parsed_url.path.strip("/").split("/")[0]

    elif hostname in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed_url.path == "/watch":
            video_id = parse_qs(parsed_url.query).get("v", [""])[0]
        elif parsed_url.path.startswith(("/shorts/", "/embed/", "/live/")):
            parts = parsed_url.path.strip("/").split("/")
            if len(parts) >= 2:
                video_id = parts[1]

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError(
            "Valid YouTube URL nahi mila. Normal watch, youtu.be, "
            "shorts, ya embed URL try karo."
        )

    return video_id


# ---------------------------------------------------------
# Model initialization
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the free local Hugging Face embedding model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_llm() -> ChatGroq:
    """Initialize the Groq chat model using GROQ_API_KEY from .env/secrets."""
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2,
    )


# ---------------------------------------------------------
# Transcript extraction
# ---------------------------------------------------------

def fetch_transcript(video_id: str) -> str:
    """Fetch an available English transcript for the YouTube video."""
    youtube_api = YouTubeTranscriptApi()

    transcript = youtube_api.fetch(
        video_id,
        languages=["en"],
    )

    transcript_text = " ".join(
        snippet.text for snippet in transcript
    ).strip()

    if not transcript_text:
        raise ValueError("Video ka transcript empty hai.")

    return transcript_text


# ---------------------------------------------------------
# Build vector store
# ---------------------------------------------------------

def create_vector_store(
    transcript_text: str,
    video_url: str,
) -> tuple[FAISS, int]:
    """Split transcript, embed chunks, and index them in FAISS."""
    document = Document(
        page_content=transcript_text,
        metadata={"source": video_url},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents([document])

    if not chunks:
        raise ValueError(
            "Transcript se text chunks create nahi ho paaye."
        )

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    return vector_store, len(chunks)


# ---------------------------------------------------------
# RAG answer generation
# ---------------------------------------------------------

def answer_question(question: str) -> tuple[str, list[Document]]:
    """Retrieve relevant transcript chunks and generate a grounded answer."""
    vector_store = st.session_state.vector_store

    if vector_store is None:
        raise ValueError("Pehle YouTube video process karo.")

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return (
            "Is question ke liye transcript mein relevant information nahi mili.",
            [],
        )

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful assistant that answers questions
                about a YouTube video's transcript.

                Follow these rules:
                - Answer using the supplied transcript context.
                - Do not invent video details or unsupported facts.
                - If the context does not contain the answer, say that
                  the available transcript does not provide enough
                  information.
                - Be clear and concise. Use bullets when useful.

                Transcript context:
                {context}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    messages = prompt.format_messages(
        context=context,
        question=question,
    )

    response = get_llm().invoke(messages)
    answer = response.content

    if not isinstance(answer, str):
        answer = str(answer)

    return answer, retrieved_docs


# ---------------------------------------------------------
# Video input and processing
# ---------------------------------------------------------

video_url = st.text_input(
    "YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

process_clicked = st.button(
    "Process Video",
    type="primary",
    use_container_width=False,
)

if process_clicked:
    if not video_url.strip():
        st.warning("Please YouTube video URL enter karo.")
    else:
        try:
            with st.spinner(
                "Fetching transcript and creating the searchable vector store..."
            ):
                video_id = extract_video_id(video_url)
                transcript_text = fetch_transcript(video_id)
                vector_store, chunk_count = create_vector_store(
                    transcript_text,
                    video_url.strip(),
                )

                # Only replace the previous video after processing succeeds.
                st.session_state.vector_store = vector_store
                st.session_state.processed_video_url = video_url.strip()
                st.session_state.chat_history = []
                st.session_state.transcript_length = len(transcript_text)

            st.success(
                f"Video processed successfully! "
                f"Created {chunk_count} searchable chunks."
            )
            st.rerun()

        except Exception as error:
            st.error(
                "Video process nahi ho paaya. Check karo ki video ka "
                "English transcript available hai aur YouTube request "
                "block nahi kar raha.\n\n"
                f"Details: {error}"
            )


# ---------------------------------------------------------
# Processed video status
# ---------------------------------------------------------

if st.session_state.processed_video_url:
    st.info(
        f"**Processed video:** {st.session_state.processed_video_url}\n\n"
        f"**Transcript characters:** {st.session_state.transcript_length:,}"
    )


# ---------------------------------------------------------
# Display previous chat messages
# ---------------------------------------------------------

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View retrieved transcript chunks"):
                for index, source_text in enumerate(
                    message["sources"],
                    start=1,
                ):
                    st.markdown(f"**Retrieved chunk {index}**")
                    st.write(source_text)


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about the processed video...",
    disabled=st.session_state.vector_store is None,
)

if question:
    st.session_state.chat_history.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Searching the transcript and generating an answer..."
            ):
                answer, retrieved_docs = answer_question(question)

            st.markdown(answer)

            source_texts = [
                doc.page_content for doc in retrieved_docs
            ]

            if source_texts:
                with st.expander("View retrieved transcript chunks"):
                    for index, source_text in enumerate(
                        source_texts,
                        start=1,
                    ):
                        st.markdown(f"**Retrieved chunk {index}**")
                        st.write(source_text)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": source_texts,
                }
            )

        except Exception as error:
            error_message = (
                "Answer generate nahi ho paaya. Groq API key, model "
                "availability, aur API usage limits check karo.\n\n"
                f"Details: {error}"
            )
            st.error(error_message)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )
