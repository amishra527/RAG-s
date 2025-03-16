from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import torch
from pymilvus import connections, Collection
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sentence_transformers import CrossEncoder
import re
import spacy
import logging
import numpy as np

# Configurations
PHI_2_MODEL_DIR = "D:\\WorkSpace_0\\CAI\\infrence\\phi-2-local"
RETRIEVER_MODEL_DIR = "D:\\WorkSpace_0\\CAI\\infrence\\retriever_model"
RERANKER_MODEL_DIR = "D:\\WorkSpace_0\\CAI\\infrence\\reranker_model"
MIN_QUERY_LENGTH = 5
RESTRICTED_WORDS_FILE = "restricted_words.txt"
PROFANITY_LIST = ["badword1", "badword2"]  # Expand this list as needed
DEFAULT_TOP_K = 5

# Setup logging
logging.basicConfig(filename='guardrail_logs.log', level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Global variables for models
tokenizer = None
model = None
retriever_model = None
retriever_tokenizer = None
reranker_model = None
nlp = None

# Load models at startup
@app.on_event("startup")
def load_models():
    """Load all the necessary models at startup."""
    global tokenizer, model, retriever_model, retriever_tokenizer, reranker_model, nlp
    
    # Load LLM for answer generation
    tokenizer, model = load_phi2_model()
    
    # Load retriever model
    retriever_tokenizer, retriever_model = load_retriever_model()
    
    # Load reranker model
    reranker_model = load_reranker_model()
    
    # Load NLP model for entity recognition
    nlp = spacy.load("en_core_web_sm")

# Define request model
class QueryRequest(BaseModel):
    username: str
    project_name: str
    query: str
    top_k: int = DEFAULT_TOP_K

# API endpoint
@app.post("/generate")
async def generate_response_endpoint(request: QueryRequest):
    """Generate a response based on the user's query using retrieval and reranking."""
    try:
        response_data = main(
            request.username,
            request.project_name,
            request.query,
            request.top_k,
            tokenizer,
            model,
            retriever_model,
            retriever_tokenizer,
            reranker_model,
            nlp
        )
        return response_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Core processing function
def main(username, project_name, query, top_k, tokenizer, model, retriever_model, retriever_tokenizer, reranker_model, nlp):
    """Process the query and generate a response with context, query, response, and references."""
    connect_to_milvus()
    collection_name = get_collection_name(username, project_name)
    collection = Collection(collection_name)
    collection.load()

    query = validate_query(query)
    
    # Create embeddings using the retriever model
    query_embedding = embed_query(query, retriever_model, retriever_tokenizer)
    
    # Initial retrieval with embeddings
    retrieved_docs, retrieval_scores = retrieve_documents(collection, query_embedding, query, top_k * 2)
    
    # Rerank the retrieved documents
    reranked_docs, confidence_scores = rerank_documents(reranker_model, query, retrieved_docs, top_k)
    
    # Extract sources from the reranked documents
    sources = [extract_source(doc) for doc in reranked_docs]
    unique_sources = list(set(sources))
    
    # Prepare context from the reranked documents
    context = "\n".join(reranked_docs)

    # Truncate context if too long
    if len(tokenizer.encode(context)) > 1500:
        context = tokenizer.decode(tokenizer.encode(context)[:1500])

    prompt = (
        f"Context: {context}\n\n"
        f"Question: {query}\n\n"
        "Only provide the exact answer to the question without additional information.\n"
        "Answer:"
    )
    
    # Generate the response
    answer = generate_response(model, tokenizer, prompt)
    final_response = filter_response(answer, nlp)

    return {
        "reference_text": context,
        "query": query,
        "response": final_response,
        "reference_doc": unique_sources,
        "confidence_score": max(confidence_scores) if confidence_scores else 0.0
    }

# Helper functions
def get_collection_name(username, project_name):
    """Generate a unique collection name based on username and project."""
    return f"NV_{username}_{project_name}"

def connect_to_milvus():
    """Establish a connection to the Milvus server."""
    connections.connect("default", host="127.0.0.1", port="19530")

def load_retriever_model():
    """Load the retriever model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_MODEL_DIR)
    model = AutoModel.from_pretrained(RETRIEVER_MODEL_DIR)
    return tokenizer, model

def load_reranker_model():
    """Load the reranker model."""
    return CrossEncoder(RERANKER_MODEL_DIR)

def embed_query(query, retriever_model, retriever_tokenizer):
    """Create embedding for the query using the retriever model."""
    inputs = retriever_tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = retriever_model(**inputs)
        # Use mean pooling to get the sentence embedding
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Get the embedding as a numpy array
        embedding_np = embedding[0].cpu().numpy()
        
        # Check dimension compatibility with your original model
        # If your original SentenceTransformer model produced 384-dim vectors (common for all-MiniLM-L6-v2)
        expected_dim = 384  # Replace with the dimension of your Milvus collection
        actual_dim = embedding_np.shape[0]
        
        # Handle dimension mismatch
        if actual_dim != expected_dim:
            if actual_dim > expected_dim:
                # Truncate if new embedding is larger
                embedding_np = embedding_np[:expected_dim]
            else:
                # Pad with zeros if new embedding is smaller
                padding = np.zeros(expected_dim - actual_dim)
                embedding_np = np.concatenate([embedding_np, padding])
        
        return embedding_np.tolist()

# def embed_query(query, retriever_model, retriever_tokenizer):
#     """Create embedding for the query using the retriever model."""
#     inputs = retriever_tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = retriever_model(**inputs)
#         # Use mean pooling to get the sentence embedding
#         attention_mask = inputs["attention_mask"]
#         token_embeddings = outputs.last_hidden_state
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#         return embedding[0].cpu().numpy().tolist()

def retrieve_documents(collection, query_embedding, query_text, limit=10):
    """Retrieve documents using vector search."""
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        [query_embedding],
        "embedding",
        search_params,
        limit=limit,
        output_fields=["text"]
    )
    
    retrieved_docs = [hit.entity.get("text") for hit in results[0]]
    retrieval_scores = [hit.score for hit in results[0]]
    
    return retrieved_docs, retrieval_scores

def rerank_documents(reranker_model, query, documents, top_k=5):
    """Rerank the retrieved documents using the reranker model."""
    if not documents:
        return [], []
    
    # Prepare input pairs for reranking
    pairs = [(query, doc) for doc in documents]
    
    # Get scores from the reranker model
    scores = reranker_model.predict(pairs)
    
    # Sort documents by reranker scores
    ranked_results = [(doc, score) for doc, score in zip(documents, scores)]
    ranked_results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k documents and their scores
    top_docs = [doc for doc, _ in ranked_results[:top_k]]
    top_scores = [float(score) for _, score in ranked_results[:top_k]]
    
    return top_docs, top_scores

def extract_source(text):
    """Extract the source name from the text."""
    match = re.match(r"Source:\s*(.+?)\n", text)
    return match.group(1) if match else "Unknown"

def load_restricted_words():
    """Load restricted words from a file or return defaults."""
    if os.path.exists(RESTRICTED_WORDS_FILE):
        with open(RESTRICTED_WORDS_FILE, 'r') as f:
            return [line.strip().lower() for line in f.readlines()]
    return ["hack", "exploit"]

def validate_query(query):
    """Validate the query for length, restricted words, and SQL injection attempts."""
    restricted_words = load_restricted_words()
    if len(query) < MIN_QUERY_LENGTH:
        logging.warning(f"Query too short: {query}")
        raise ValueError("Invalid query: Too short.")
    if any(word in query.lower() for word in restricted_words):
        logging.warning(f"Restricted word found in query: {query}")
        raise ValueError("Invalid query: Contains restricted content.")
    sql_injection_patterns = [
        r"\b(select|insert|update|delete|drop|alter)\b",
        r"\b(union|into|from|where)\b",
        r"--|/\*|\*/"
    ]
    for pattern in sql_injection_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            logging.warning(f"Potential SQL injection attempt: {query}")
            raise ValueError("Invalid query: Potential SQL injection attempt.")
    return query

def filter_response(response, nlp):
    """Filter the response for profanity and sensitive entities."""
    if any(word in response.lower() for word in PROFANITY_LIST):
        logging.info(f"Profanity detected in response: {response}")
        return "[Response blocked due to inappropriate content.]"
    doc = nlp(response)
    redacted_response = response
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
            redacted_response = redacted_response.replace(ent.text, "[REDACTED]")
    logging.info(f"Filtered response: {redacted_response}")
    return redacted_response

def load_phi2_model():
    """Load the Phi-2 model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(PHI_2_MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        PHI_2_MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return tokenizer, model

def generate_response(model, tokenizer, prompt):
    """Generate a response using the Phi-2 model, returning only the generated part."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_length = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=200)
    generated_sequence = outputs[0, input_length:]
    return tokenizer.decode(generated_sequence, skip_special_tokens=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10002)