import streamlit as st
import os
try:
    import pymupdf as fitz  # PyMuPDF modern import for fast layout-aware text extraction
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_community.vectorstores.faiss import FAISS
except Exception:
    from langchain.vectorstores import FAISS

# Self-contained text splitter that avoids heavy/conflicting ML dependencies
class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size=10000, chunk_overlap=1000, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str):
        if not text:
            return []
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            split_idx = -1
            for sep in self.separators:
                if sep:
                    pos = text.rfind(sep, start, end)
                    if pos > start:
                        split_idx = pos + len(sep)
                        break
            if split_idx == -1:
                split_idx = end
            chunk = text[start:split_idx].strip()
            if chunk:
                chunks.append(chunk)
            start = max(start + 1, split_idx - self.chunk_overlap)
        return chunks

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            if fitz is not None:
                # Fast & layout-aware extraction with PyMuPDF
                doc = fitz.open(stream=pdf.read(), filetype="pdf")
                for page in doc:
                    text += page.get_text() + "\n"
                pdf.seek(0)
            else:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
        except Exception:
            pdf.seek(0)
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text



def get_text_chunks(text):
    # Optimized chunking: 1000 characters (~250 tokens) with 150-char overlap for optimal RAG precision
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    # Upgraded to Google's state-of-the-art text-embedding-004 model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    # Clear cache so next query picks up the newly saved index
    load_vector_store.clear()


@st.cache_resource(show_spinner=False)
def load_vector_store():
    """Cached FAISS vector store — loads once, reused on every query (5x faster)."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


def get_conversational_chain(detailed=True):
    if detailed:
        style = "Answer the question as detailed as possible from the provided context, make sure to provide all the details"
    else:
        style = "Provide a concise, direct, and clear summary answer based only on the provided context"

    prompt_template = f"""
    {style}, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {{context}}?\n
    Question: \n{{question}}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt | model | StrOutputParser()



def user_input(user_question, show_sources=False, detailed=True):
    if not os.path.exists("faiss_index"):
        st.warning("Please upload and process PDF files first before asking questions.")
        return

    # Use cached vector store — no reload on every question
    new_db = load_vector_store()
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(detailed=detailed)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    with st.spinner("Generating answer..."):
        response = chain.invoke({"context": context, "question": user_question})

    if show_sources and docs:
        source_snippets = []
        for i, doc in enumerate(docs, 1):
            snippet = doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else "")
            source_snippets.append(f"**Snippet {i}:** {snippet}")
        response += "\n\n---\n📄 **Sources:**\n" + "\n\n".join(source_snippets)

    # Save to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": response})




def main():
    st.set_page_config("Multi PDF Chatbot", page_icon = ":scroll:")

    if "night_mode" not in st.session_state:
        st.session_state.night_mode = True

    # Initialize chat history in session
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    col_title, col_theme = st.columns([3.5, 1.5])
    with col_title:
        st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")
    with col_theme:
        night_mode = st.toggle(
            "🌙 Night Mode" if st.session_state.night_mode else "☀️ Day Mode",
            value=st.session_state.night_mode,
            key="theme_toggle"
        )
        st.session_state.night_mode = night_mode

    # Dynamic Day / Night Theme Styles
    if not night_mode:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #F8FAFC !important;
                color: #0F172A !important;
            }
            [data-testid="stSidebar"] {
                background-color: #FFFFFF !important;
                border-right: 1px solid #E2E8F0 !important;
            }
            [data-testid="stSidebar"] * {
                color: #0F172A !important;
            }
            .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, label {
                color: #0F172A !important;
            }
            div[data-baseweb="input"] input {
                background-color: #FFFFFF !important;
                color: #0F172A !important;
                border: 1px solid #CBD5E1 !important;
            }
            div[data-testid="stExpander"] {
                background-color: #FFFFFF !important;
                border: 1px solid #E2E8F0 !important;
            }
            .app-footer {
                background-color: #FFFFFF !important;
                color: #475569 !important;
                border-top: 1px solid #E2E8F0 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #0E1117 !important;
                color: #F1F5F9 !important;
            }
            [data-testid="stSidebar"] {
                background-color: #161B22 !important;
                border-right: 1px solid #30363D !important;
            }
            [data-testid="stSidebar"] * {
                color: #F1F5F9 !important;
            }
            .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, label {
                color: #F1F5F9 !important;
            }
            div[data-baseweb="input"] input {
                background-color: #0D1117 !important;
                color: #F1F5F9 !important;
                border: 1px solid #30363D !important;
            }
            div[data-testid="stExpander"] {
                background-color: #161B22 !important;
                border: 1px solid #30363D !important;
            }
            .app-footer {
                background-color: #0E1117 !important;
                color: #94A3B8 !important;
                border-top: 1px solid #30363D !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        show_sources = st.toggle("🔍 Show PDF Sources", value=False, help="Toggle to view extracted source passages used for the answer")
    with col2:
        detailed_mode = st.toggle("📝 Detailed Answers", value=True, help="Toggle between detailed and concise response modes")
    with col3:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Display full chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    # Chat input at the bottom
    user_question = st.chat_input("Ask a Question from the PDF Files uploaded .. ✍️📝")

    if user_question:
        # Show user message immediately
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_question)
        user_input(user_question, show_sources=show_sources, detailed=detailed_mode)
        # Show latest assistant response
        if st.session_state.chat_history:
            latest = st.session_state.chat_history[-1]
            if latest["role"] == "assistant":
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(latest["content"])

    with st.sidebar:

        if os.path.exists("img/pdf_logo.jpg"):
            st.image("img/pdf_logo.jpg")
        st.write("---")
        
        current_api_key = os.getenv("GOOGLE_API_KEY")
        if not current_api_key:
            input_key = st.text_input("Enter Google API Key 🔑", type="password", help="Enter your Gemini API key if not set in .env")
            if input_key:
                os.environ["GOOGLE_API_KEY"] = input_key
                genai.configure(api_key=input_key)
                current_api_key = input_key
        else:
            st.success("Google API Key loaded from environment ✅")

        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not current_api_key:
                st.error("Please provide a Google API Key above or in a .env file.")
            elif not pdf_docs:
                st.warning("Please upload at least one PDF file first.")
            else:
                with st.spinner("Processing..."): # user friendly message.
                    raw_text = get_pdf_text(pdf_docs) # get the pdf text
                    text_chunks = get_text_chunks(raw_text) # get the text chunks
                    get_vector_store(text_chunks) # create vector store
                    st.success("Done")
        
        st.write("---")
        if os.path.exists("img/gkj.jpg"):
            st.image("img/gkj.jpg")
        st.write("AI App created by @ Priyanshu")


    st.markdown(
        """
        <div class="app-footer" style="position: fixed; bottom: 0; left: 0; width: 100%; padding: 12px; text-align: center; z-index: 999;">
            © Priyanshu | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
