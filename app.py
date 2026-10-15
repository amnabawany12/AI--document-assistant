
import streamlit as st
import os
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚"
)

st.title("📚 AI-Powered Document Question Answering System")
st.write("Search and find relevant information in your documents.")

# Read text from files
def read_file(path):
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


# Split text into smaller chunks
def split_text(text, words_per_chunk=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i:i + words_per_chunk])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


# Load documents from the documents folder
def load_documents():
    folder = "documents"
    all_chunks = []
    all_sources = []

    if not os.path.exists(folder):
        return all_chunks, all_sources

    for filename in os.listdir(folder):
        if filename.lower().endswith((".txt", ".md", ".pdf")):
            path = os.path.join(folder, filename)

            try:
                text = read_file(path)

                for chunk in split_text(text):
                    all_chunks.append(chunk)
                    all_sources.append(filename)

            except Exception as error:
                st.warning(f"Could not read {filename}: {error}")

    return all_chunks, all_sources


# Process documents
if st.button("Process My Documents"):
    chunks, sources = load_documents()

    if not chunks:
        st.error("No documents found. Check your documents folder.")
    else:
        vectorizer = TfidfVectorizer()
        document_vectors = vectorizer.fit_transform(chunks)

        st.session_state["chunks"] = chunks
        st.session_state["sources"] = sources
        st.session_state["vectorizer"] = vectorizer
        st.session_state["vectors"] = document_vectors

        st.success(f"Processed {len(chunks)} text chunks!")


# Ask questions
st.subheader("Ask a Question")

question = st.text_input("Enter your question:")

if st.button("Find Relevant Documents"):
    if not question.strip():
        st.warning("Please enter a question.")

    elif "vectors" not in st.session_state:
        st.warning("First click 'Process My Documents'.")

    else:
        vectorizer = st.session_state["vectorizer"]
        chunks = st.session_state["chunks"]
        sources = st.session_state["sources"]
        document_vectors = st.session_state["vectors"]

        question_vector = vectorizer.transform([question])

        scores = cosine_similarity(
            question_vector,
            document_vectors
        ).flatten()

        top_indices = scores.argsort()[::-1][:3]

        st.subheader("Top 3 Relevant Chunks")

        for rank, index in enumerate(top_indices, start=1):
            st.markdown(
                f"**Result {rank}: {sources[index]}**"
            )
            st.write(chunks[index])
            st.write(f"Similarity score: {scores[index]:.4f}")
            st.divider()

st.caption(
    "This version uses TF-IDF keyword-based search, "
    "not transformer embeddings."
)
