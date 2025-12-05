import streamlit as st
import os
from rag_core import add_pdf_to_vector_db, answer_question, clear_vector_db

# ----------------------------------------------------
# Initialize chat history
# ----------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"q":..., "a":...}

# ----------------------------------------------------
# Streamlit App
# ----------------------------------------------------
def main():
    st.set_page_config(page_title="Mini RAG Assistant", layout="wide")
    st.title("📚 Mini RAG-based AI Knowledge Assistant")
    st.write("Upload PDFs and ask questions based on their content. Uses RAG + ChromaDB + Groq Llama 3.1.")

    # ---------------- Sidebar: PDF Upload ----------------
    st.sidebar.header("📄 Upload PDF")

    uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        # Save PDF locally
        save_path = os.path.join("data/docs", uploaded_file.name)
        os.makedirs("data/docs", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.sidebar.success(f"Saved: {uploaded_file.name}")

        # Index PDF into vector DB
        if st.sidebar.button("Index this PDF"):
            add_pdf_to_vector_db(save_path, source_name=uploaded_file.name)
            st.sidebar.success("PDF indexed into vector DB ✅")

    # ---------------- Sidebar: Clear Database ----------------
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear Vector Database"):
        clear_vector_db()
        st.sidebar.success("Vector DB cleared!")

    # ---------------- Main UI: Q/A Section ----------------
    st.write("### ❓ Ask a question based on the uploaded PDFs")
    query = st.text_input("Your question")

    if st.button("Get Answer") and query.strip():
        with st.spinner("Thinking..."):
            answer, retrieved_chunks = answer_question(query)

        # Store in history
        st.session_state.chat_history.append({"q": query, "a": answer})

        # Show Answer
        st.write("### 🤖 Answer")
        st.info(answer)

        # Show Retrieved Context for transparency
        st.write("### 📚 Retrieved Context Chunks")
        for i, chunk in enumerate(retrieved_chunks, start=1):
            with st.expander(f"Chunk {i}"):
                st.write(chunk)

    # ---------------- Chat History ----------------
    st.write("### 🧵 Conversation History")

    if len(st.session_state.chat_history) == 0:
        st.caption("No chat history yet.")
    else:
        for turn in st.session_state.chat_history:
            st.markdown(f"**🧑 You:** {turn['q']}")
            st.markdown(f"**🤖 Assistant:** {turn['a']}")
            st.markdown("---")

# ----------------------------------------------------
# Run App
# ----------------------------------------------------
if __name__ == "__main__":
    main()
