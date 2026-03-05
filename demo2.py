import os
import faiss
import pickle
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
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


@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-large-en-v1.5")


# Using BAAI/bge-large-en-v1.5 which is specifically trained for semantic similarity
try:
    embedding_model = load_model()
except Exception as e:
    st.error(f"Failed to load main embedding model: {e}")
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  # Strong fallback model

def process_query(query):
    """Enhanced query processing to improve semantic matching"""
    # Add instruction prefix for better semantic understanding
    return f"Represent this sentence for searching relevant passages: {query}"

def extract_text_with_metadata(filepath):
    """
    Extracts text while preserving page numbers.
    Returns a list of LangChain Document objects.
    """
    try:
        reader = PdfReader(filepath)
        documents = []
        
        if hasattr(filepath, 'name'):
            source_name = filepath.name
        else:
            source_name = str(filepath)
            
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text :
                doc = Document(
                    page_content=text,
                    metadata = {"page":page_num+1, "source":source_name}
                )
                documents.append(doc)
        return documents

    except Exception as e:
        st.error(f"Error : {e}")
        return []
    
def EmbeddingToVectorDB(documents, file_path):
    if not documents:
        st.error("No documents to process")
        return False
    
    # Enhanced text splitting for better semantic chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?"],  # More natural text boundaries
        chunk_size=512,
        chunk_overlap=150,  # Increased overlap to maintain context
        length_function=len
    )
    splits_docs = text_splitter.split_documents(documents)
    
    if not splits_docs:
        st.error("Failed to split documents. No valid text content found.")
        return False
    
    try:
        # Embedding
        texts_to_embed = [f"Represent this sentence for retrieval: {doc.page_content}" for doc in splits_docs]
        embeddings = embedding_model.encode(
                                        texts_to_embed, 
                                         normalize_embeddings=True,
                                         batch_size=8,
                                         show_progress_bar=True)
        # Save to FAISS
        dimension = embeddings.shape[1]
        # Enhanced HNSW index configuration for better semantic search
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 128  # Increased accuracy during index building
        index.hnsw.efSearch = 128  # Increased accuracy during search
        index.add(np.array(embeddings, dtype=np.float32))
        
        faiss.write_index(index, file_path)
        
        # Introducing BM25 Hybrid Code - Toeknize the text for Keyword Searching
        tokenized_corpus = [doc.page_content.lower().split() for doc in splits_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Save the full document objects(Content + metadata + BM25 index) to pickle
        with open(file_path.replace('.pkl', '_metadata.pkl'), "wb") as f:
            pickle.dump({'docs' : splits_docs, 'bm25' : bm25}, f)
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
            model="llama-3.3-70b-versatile",        #llama3-8b-8192
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150
        )
        variations = completion.choices[0].message.content.strip().split('\n')
        return variations
    except Exception:
        return [query]

def AskQuery(query, file_path, top_k=5):
    if not query or not os.path.exists(file_path):
        return None
    
    try:
        # Get query variations for better semantic coverage
        query_variations = [query]
        
        index = faiss.read_index(file_path)
        with open(file_path.replace('.pkl', '_metadata.pkl'), "rb") as f:
            data = pickle.load(f)
            metadata = data['docs']
            bm25 = data['bm25']
        
        all_results = {}  # Dictionary is used over array can hold combined scores,
        for variation in query_variations:
            # FAISS Semantic Search
            processed_query = process_query(variation)
            query_embedding = embedding_model.encode([processed_query], 
                                                  normalize_embeddings=True)
            
            distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k=top_k)
            
            # BM25 Keyword Search
            tokenized_query = variation.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            bm25_top_Indices = np.argsort(bm25_scores)[::-1][:top_k]
            
            
            # Combine Scores (Simple Reciprocal Rank Fusion)
            for rank, idx in enumerate(indices[0]):
                if idx not in all_results:
                    all_results[idx] = {'doc':metadata[idx], 'score':0}
                all_results[idx]['score'] += 1.0 / (rank+1) # Adding FAISS rank score
                
            for rank, idx in enumerate(bm25_top_Indices):
                if bm25_scores[idx] > 0 :
                    if idx not in all_results:
                        all_results[idx] = {'doc':metadata[idx], 'score':0}
                    all_results[idx]['score'] += 1.0 / (rank+1) # Adding BM25 rank score
        
        # Sort by combined highest score
        sorted_results = sorted(all_results.values(), key=lambda x: x['score'], reverse = True)           
                    
        formatted_result = [
            f"{item['doc'].page_content} [Source : Page {item['doc'].metadata['page']}]"
            for item in sorted_results[:3]
        ]      
        return formatted_result
    
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
            model="llama-3.3-70b-versatile",       #llama3-8b-8192
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024
        )
        bot_response = completion.choices[0].message.content
        st.header("Combined Output")
        st.write(bot_response)
    except Exception as e:
        st.error(f"An Error Occurred: {e}")

if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None
    
# Streamlit UI
st.title("Enhanced PDF Question Answering System")
st.markdown("""
This system uses advanced semantic search to understand the meaning behind your questions 
and find relevant information in the PDF document, with source tracking.
""")

pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])
file_path = "faiss_store_hnsw.pkl"

if pdf_file:
    if st.session_state.processed_file != pdf_file.name:
        with st.spinner('Processing PDF... This may take a moment for larger documents.'):
            documents = extract_text_with_metadata(pdf_file)
        # Pass the list of Document objects to the VectorDB function
            if documents and EmbeddingToVectorDB(documents, file_path):
                st.session_state.processed_file = pdf_file.name
                st.success("PDF processed successfully and metadata attached")
    else:
        #If pdf is already processed, just show a quiet success message
        st.info("PDF is loaded and ready. Ask away")
        
query = st.text_input("Enter your question:")
if query:
    with st.spinner('Searching...'):
        results = AskQuery(query, file_path)
        if results:
            st.write("Debugging")
            for res in results:
                st.info(res)
            getLLMOutPut(query, file_path)
        else:
            st.warning("No relevant information found in VectorDB.")