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

if not api_key:
    st.error("API key not found...Check .env file")
    st.stop()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = api_key

client = Groq(api_key=api_key)

# Using BAAI/bge-large-en-v1.5 which is specifically trained for semantic similarity
try:
    embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
except Exception as e:
    st.error(f"Failed to load main embedding model: {e}")
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  # Strong fallback model

def process_query(query):
    """Enhanced query processing to improve semantic matching"""
    # Add instruction prefix for better semantic understanding
    return f"Represent this sentence for searching relevant passages: {query}"

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
    
    # Enhanced text splitting for better semantic chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?"],  # More natural text boundaries
        chunk_size=512,
        chunk_overlap=150,  # Increased overlap to maintain context
        length_function=len
    )
    docs = text_splitter.split_text(text=text)
    
    if not docs:
        st.error("Failed to split documents. No valid text content found.")
        return False
    
    try:
        # Process chunks with instruction prefix for better embedding
        processed_docs = [f"Represent this sentence for retrieval: {doc}" for doc in docs]
        embeddings = embedding_model.encode(processed_docs, 
                                         normalize_embeddings=True,
                                         batch_size=8,
                                         show_progress_bar=True)
        
        dimension = embeddings.shape[1]
        # Enhanced HNSW index configuration for better semantic search
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 128  # Increased accuracy during index building
        index.hnsw.efSearch = 128  # Increased accuracy during search
        index.add(np.array(embeddings, dtype=np.float32))
        
        faiss.write_index(index, file_path)
        with open(file_path.replace('.pkl', '_metadata.pkl'), "wb") as f:
            pickle.dump(docs, f)
        
        return True
    except Exception as e:
        st.error(f"Error in embedding process: {e}")
        return False

def expand_query(query):
    """Generate semantic variations of the query"""
    try:
        prompt = """Given the query, generate 2-3 semantic variations that capture the same meaning. 
        Format: Original query followed by variations, one per line. Keep it concise.
        Query: """ + query
        
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )
        variations = completion.choices[0].message.content.strip().split('\n')
        return variations
    except Exception:
        return [query]

def AskQuery(query, file_path, top_k=10, similarity_threshold=0.5):
    if not query or not os.path.exists(file_path):
        return None
    
    try:
        # Get query variations for better semantic coverage
        query_variations = expand_query(query)
        
        index = faiss.read_index(file_path)
        with open(file_path.replace('.pkl', '_metadata.pkl'), "rb") as f:
            metadata = pickle.load(f)
        
        all_results = []
        for variation in query_variations:
            # Process query with instruction prefix
            processed_query = process_query(variation)
            query_embedding = embedding_model.encode([processed_query], 
                                                  normalize_embeddings=True)
            
            distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k=top_k)
            
            # Collect results from each variation
            for idx, distance in zip(indices[0], distances[0]):
                if distance > similarity_threshold:
                    all_results.append((metadata[idx], distance))
        
        # Remove duplicates and sort by similarity
        unique_results = {}
        for text, score in all_results:
            if text not in unique_results or score > unique_results[text]:
                unique_results[text] = score
        
        sorted_results = [text for text, _ in sorted(unique_results.items(), 
                                                   key=lambda x: x[1], 
                                                   reverse=True)]
        
        return sorted_results[:4] if sorted_results else None
        
    except Exception as e:
        st.error(f"Error in query process: {e}")
        return None

def getLLMOutPut(query, file_path):
    results = AskQuery(query, file_path)
    try:
        # Enhanced prompt for better context utilization
        prompt = f"""Based on the following context, provide a comprehensive answer to the question. 
        If the context doesn't contain enough information, say so.
        
        Question: {query}
        
        Context: {results}
        
        Answer:"""
        
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024
        )
        bot_response = completion.choices[0].message.content
        st.header("Combined Output")
        st.write(bot_response)
    except Exception as e:
        st.error(f"An Error Occurred: {e}")

# Streamlit UI
st.title("Enhanced PDF Question Answering System")
st.markdown("""
This system uses advanced semantic search to understand the meaning behind your questions 
and find relevant information in the PDF document.
""")

pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])
file_path = "faiss_store_hnsw.pkl"

if pdf_file:
    with st.spinner('Processing PDF... This may take a moment for larger documents.'):
        text = extract_text_from_pdf(pdf_file)
        if text and EmbeddingToVectorDB(text, file_path):
            st.success("PDF processed successfully! 🎉")

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
            getLLMOutPut(query, file_path)
        else:
            st.warning("No relevant information found.")