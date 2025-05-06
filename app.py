import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import io
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

load_dotenv()

# Custom HTML for the animated text loader
loader_html = """
<style>
html, body {
    height: 100%;
    width: 100%;
    margin: 0;
    padding: 0;
    font-size: 100%;
    background: transparent;
    text-align: center;
}

h1 {
    margin: 0;
    padding: 0;
    font-family: 'Arial Narrow', sans-serif;
    font-weight: 100;
    font-size: 1.1em;
    color: #a3e1f0;
}

span {
    position: relative;
    top: 0.63em;  
    display: inline-block;
    text-transform: uppercase;  
    opacity: 0;
    transform: rotateX(-90deg);
}

.let1 { animation: drop 1.2s ease-in-out infinite; animation-delay: 1.2s; }
.let2 { animation: drop 1.2s ease-in-out infinite; animation-delay: 1.3s; }
.let3 { animation: drop 1.2s ease-in-out infinite; animation-delay: 1.4s; }
.let4 { animation: drop 1.2s ease-in-out infinite; animation-delay: 1.5s; }
.let5 { animation: drop 1.2s ease-in-out infinite; animation-delay: 1.6s; }
.let6 { animation: drop 1.2s ease-in-out infinite; animation-delay: 1.7s; }
.let7 { animation: drop 1.2s ease-in-out infinite; animation-delay: 1.8s; }

@keyframes drop {
    10% {
        opacity: 0.5;
    }
    20% {
        opacity: 1;
        top: 3.78em;
        transform: rotateX(-360deg);
    }
    80% {
        opacity: 1;
        top: 3.78em;
        transform: rotateX(-360deg);
    }
    90% {
        opacity: 0.5;
    }
    100% {
        opacity: 0;
        top: 6.94em
    }
}
</style>

<h1>
    <span class="let1">l</span>  
    <span class="let2">o</span>  
    <span class="let3">a</span>  
    <span class="let4">d</span>  
    <span class="let5">i</span>  
    <span class="let6">n</span>  
    <span class="let7">g</span>  
</h1>
"""

## loading the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']

# Initialize session state if it doesn't exist
if "vectors" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Initialize empty variables for PDF documents and vectors
    st.session_state.pdf_docs = None
    st.session_state.pdf_vectors = None
    st.session_state.website_docs = None
    st.session_state.website_vectors = None


# Website URL input
website_url = st.text_input("Enter Website URL:")
if website_url:
    try:
        loader = WebBaseLoader(website_url)
        st.session_state.website_docs = loader.load()
        st.session_state.website_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.website_final_documents = st.session_state.website_text_splitter.split_documents(st.session_state.website_docs)
        st.session_state.website_vectors = FAISS.from_documents(st.session_state.website_final_documents, st.session_state.embeddings)
    except Exception as e:
        st.error(f"Error loading website: {e}")
# PDF Upload section - MODIFIED
pdf_file = st.file_uploader("Upload PDF File", type="pdf")
if pdf_file is not None:
    try:
        pdf_bytes = pdf_file.read()
        pdf_file_io = io.BytesIO(pdf_bytes)
        pdf_reader = pypdf.PdfReader(pdf_file_io)
        text = ""
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(text)
        # Correctly create a list of Document objects
        docs = [Document(page_content=t) for t in texts]
        st.session_state.pdf_final_documents = docs
        st.session_state.pdf_vectors = FAISS.from_documents(st.session_state.pdf_final_documents, st.session_state.embeddings)
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
st.title("Chat Groq")
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="gemma2-9b-it")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions base on the provided context only. 
please provide the more accurate response based on the question 
<context>
{context}  
<context>
Questions:{input}
"""
)

document_chain = create_stuff_documents_chain(llm,prompt)


# Radio button to select data source
data_source = st.radio("Select Data Source:", ("Website", "PDF"))

prompt = st.text_input("Please Input Your Prompt Here")
submit_button = st.button("Submit", type="primary")

if prompt and submit_button:
    loader_placeholder = st.empty()
    loader_placeholder.markdown(loader_html, unsafe_allow_html=True)
    
    start = time.process_time()
    try:
        if data_source == "Website" and st.session_state.website_vectors:
            website_retriever = st.session_state.website_vectors.as_retriever()
            website_retrieval_chain = create_retrieval_chain(website_retriever, document_chain)
            response = website_retrieval_chain.invoke({"input": prompt})
        elif data_source == "PDF" and st.session_state.pdf_vectors:
            pdf_retriever = st.session_state.pdf_vectors.as_retriever()
            pdf_retrieval_chain = create_retrieval_chain(pdf_retriever, document_chain)
            response = pdf_retrieval_chain.invoke({"input": prompt})
        else:
            st.error("Please select a data source and provide valid data.")
            response = None # prevents error further down
    except Exception as e:
        st.exception(e)

    process_time = time.process_time() - start
    
    loader_placeholder.empty()
    if response:
        st.write(response['answer'])
        st.info(f"Response Time: {process_time:.2f} seconds")

        with st.expander("Document Similarity Search"):
            for i, docs in enumerate(response["context"]):
                st.write(docs.page_content)
                st.write("------------------------------------")
