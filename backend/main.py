# backend/main.py
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Get the Gemini API key from environment variables
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Configure the Gemini API client
genai.configure(api_key=API_KEY)

# Initialize the generative model
llm = genai.GenerativeModel('gemini-1.5-pro')

# Initialize the embedding model for vector search
# This model converts text into numerical vectors
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Data and Vector DB Setup ---
def create_vector_store():
    """
    Creates a FAISS in-memory vector store from a CSV string.
    In a real-world application, this data would come from a database or a file.
    """
    try:
        # Placeholder FAQ data
        data_content = """
question,answer
What is the shipping policy?,Our standard shipping takes 3-5 business days. We also offer express shipping for an additional fee.
How can I return an item?,You can return an item within 30 days of purchase with the original receipt. Please visit our returns page for more information.
What are your business hours?,We are open Monday to Friday, from 9 AM to 5 PM EST. Our customer support is available 24/7.
Do you offer international shipping?,Yes, we offer international shipping to a wide range of countries. Shipping costs and times vary by destination.
What is the product warranty?,All of our products come with a one-year manufacturer's warranty against defects.
How do I contact customer support?,You can contact our support team via email at support@example.com or by calling 1-800-555-0199.
"""
        # Create a DataFrame from the placeholder content
        df = pd.read_csv(pd.compat.StringIO(data_content))

        # Generate embeddings for the 'question' column
        embeddings = embedding_model.encode(df['question'].tolist())
        dimension = embeddings.shape[1]

        # Create a FAISS index
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        print("FAISS index created successfully.")
        return index, df
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None, None

# Create the vector store and load the DataFrame on startup
vector_store, faq_df = create_vector_store()

if vector_store is None or faq_df is None:
    raise RuntimeError("Failed to initialize vector store. Please check data source.")

# --- FastAPI App Setup ---
app = FastAPI(
    title="FAQ Chatbot Backend",
    description="A RAG system using FastAPI, Gemini API, and FAISS."
)

# Set up CORS to allow communication with the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"], # Allows all HTTP methods
    allow_headers=["*"], # Allows all HTTP headers
)

# Pydantic model for the request body
class QueryRequest(BaseModel):
    query: str

# --- API Endpoints ---
@app.post("/api/query")
async def handle_query(request: QueryRequest):
    """
    Receives a user query, performs a vector search, and generates a grounded answer
    using the Gemini API.
    """
    user_query = request.query
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        # Step 1: Embed the user's query
        query_embedding = embedding_model.encode([user_query])

        # Step 2: Search the FAISS index for the top K relevant documents
        k = 3 # Number of top results to retrieve
        distances, indices = vector_store.search(query_embedding, k)
        
        # Step 3: Retrieve the actual documents (questions and answers)
        relevant_docs = []
        for i in indices[0]:
            if i != -1: # Ensure the index is valid
                relevant_docs.append({
                    "question": faq_df.loc[i, 'question'],
                    "answer": faq_df.loc[i, 'answer']
                })

        # Check if any documents were found
        if not relevant_docs:
            return {
                "answer": "I'm sorry, I couldn't find any relevant information to answer your question.",
                "source": "No relevant documents found."
            }
        
        # Step 4: Construct the prompt for the LLM
        context = ""
        for doc in relevant_docs:
            context += f"Q: {doc['question']}\nA: {doc['answer']}\n\n"
        
        prompt = f"""
        You are a helpful and professional chatbot. Use the following context to answer the user's question. 
        If the answer is not contained in the context, say that you don't have enough information.

        Context:
        {context}

        User Question: {user_query}
        """

        # Step 5: Call the Gemini API to get a grounded answer
        response = llm.generate_content(prompt)
        
        # Step 6: Return the LLM's answer and the source
        answer = response.text.strip()
        source = relevant_docs[0]['question'] # Use the top-ranking question as the source
        
        return {
            "answer": answer,
            "source": source
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Main endpoint for the root URL
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Chatbot Backend! Use /api/query to interact."}

