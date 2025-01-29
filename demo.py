import os
import faiss
import pickle
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

# Load environment variable
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key :
    st.error("API key not found...Check .env file")
    st.stop()

# Enable tracing/logging (LANGCHAIN_TRACING_V2) and make the API key available globally via os.environ.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["GROQ_API_KEY"] = api_key

client = Groq(api_key=api_key)


def extract_text_from_pdf(filepath):
    try:
        reader = PdfReader(filepath)
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text() + " "
        extracted_text = " ".join(extracted_text.split())
        return extracted_text
    except Exception as e:
        st.error(f"An error occurred while extracting PDF: {e}")
        return None

def EmbeddingToVectorDB(text, file_path):
    if not text:
        st.error("No text to process")
        return False
        
    # Split data into chunks with larger size for better context
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=512,  # Increased for better context
        chunk_overlap=100  # Increased for better coherence
    )
    
    docs = text_splitter.split_text(text=text)
    
    if not docs:
        st.error("Failed to split documents. No valid text content found.")
        return False

    try:
        # Load HuggingFace embedding model
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Show progress for long documents
        with st.spinner('Generating embeddings...'):
            embeddings = embedding_model.encode(docs, show_progress_bar=True)
        
        # Create FAISS index with normalized vectors for better similarity search
        dimension = embeddings.shape[1]
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_array)
        index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity for normalized vectors
        index.add(embeddings_array)

        # Save FAISS index
        faiss.write_index(index, file_path)
        
        # Save metadata (text chunks) to file
        metadata_path = file_path.replace('.pkl', '_metadata.pkl')
        with open(metadata_path, "wb") as f:
            pickle.dump(docs, f)
        
        return True
        
    except Exception as e:
        st.error(f"Error in embedding process: {e}")
        return False

def AskQuery(query, file_path, top_k=5, similarity_threshold=0.5):
    if not query or not os.path.exists(file_path):
        return None
        
    try:
        # Load FAISS index and metadata
        index = faiss.read_index(file_path)
        metadata_path = file_path.replace('.pkl', '_metadata.pkl')
        
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # Initialize embedding model
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Generate query embedding and normalize
        query_embedding = embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Perform similarity search
        distances, indices = index.search(query_embedding, k=top_k)
        
        # Filter results based on similarity threshold
        filtered_results = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist > similarity_threshold:  # Higher score = more similar for inner product
                filtered_results.append(metadata[idx])
                
        return filtered_results[:4]
        
    except Exception as e:
        st.error(f"Error in query process: {e}")
        return None

def getLLMOutPut():
    results = AskQuery(query, file_path)
    try :
        # send to groq
        prompt = str(query) + " " + str(results)
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt }],
                temperature=0.3,
                top_p=1,
                max_tokens=1024,
                stream=False,
                stop=None,
        )
        # Retrieve the AI-generated response
        bot_response = completion.choices[0].message.content
        st.header("Combined output")
        st.write(bot_response)
    
    except Exception as e:
        st.error(f"An Error Occurred: {e}")



# Streamlit UI
st.title("PDF Question Answering System")

# File uploader
pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])
file_path = "faiss_store_huggingface.pkl"

if pdf_file:
    with st.spinner('Processing PDF...'):
        text = extract_text_from_pdf(pdf_file)
        if text and EmbeddingToVectorDB(text, file_path):
            st.success("PDF processed successfully! 🎉")


# Query input
query = st.text_input("Enter your question:")
if query:
    with st.spinner('Searching for relevant information...'):
        results = AskQuery(query, file_path)
        if results:
            st.header("Relevant Passages")
            for i, text in enumerate(results, 1):
                st.markdown(f"**Passage {i}:**")
                st.write(text)
                st.markdown("---")
            getLLMOutPut()
        else:
            st.warning("No relevant information found.")


    




        
