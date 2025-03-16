import os
import tempfile
from pathlib import Path
from io import BytesIO
import warnings
import multiprocessing
import gradio as gr
import torch
import numpy as np

# Import your model and Milvus related dependencies
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import spacy
import logging
import re

# Import your custom modules
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

# ---- Setup your configuration variables ----
BASE_DIR = r"D:\WorkSpace_0\CAI\infrence\ConvData\MarkerData"
os.makedirs(BASE_DIR, exist_ok=True)
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
RETRIEVER_MODEL_DIR = r"D:\WorkSpace_0\CAI\infrence\retriever_model"
PHI_2_MODEL_DIR = r"D:\WorkSpace_0\CAI\infrence\phi-2-local"
RERANKER_MODEL_DIR = r"D:\WorkSpace_0\CAI\infrence\reranker_model"
RESTRICTED_WORDS_FILE = "restricted_words.txt"
PROFANITY_LIST = ["badword1", "badword2"]
DEFAULT_TOP_K = 5
MIN_QUERY_LENGTH = 5

# Set Hugging Face local cache (if needed)
os.environ["HF_HOME"] = r"D:\WorkSpace_0\CAI\infrence\local_models_marker"

# ---- Load models & initialize global variables ----
# For document conversion, we use your PDF converter (assuming its configuration is as you provided)
# For retrieval and generation, we load the required models
tokenizer_retriever = AutoTokenizer.from_pretrained(RETRIEVER_MODEL_DIR)
model_retriever = AutoModel.from_pretrained(RETRIEVER_MODEL_DIR)

with torch.no_grad():
    sample_inputs = tokenizer_retriever("This is a test", return_tensors="pt")
    sample_outputs = model_retriever(**sample_inputs)
    VECTOR_DIMENSION = sample_outputs.last_hidden_state[:, 0, :].shape[1]
    print(f"Using vector dimension: {VECTOR_DIMENSION}")

# Load LLM for answer generation
tokenizer_phi, model_phi = None, None
def load_phi2_model():
    global tokenizer_phi, model_phi
    tokenizer_phi = AutoTokenizer.from_pretrained(PHI_2_MODEL_DIR, trust_remote_code=True)
    model_phi = AutoModelForCausalLM.from_pretrained(
        PHI_2_MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
load_phi2_model()

# Load reranker model and NLP for filtering
from sentence_transformers import CrossEncoder
reranker_model = CrossEncoder(RERANKER_MODEL_DIR)
nlp = spacy.load("en_core_web_sm")

# ---- Helper functions shared across endpoints ----

def cleanup_resources():
    multiprocessing.active_children()
    warnings.filterwarnings("ignore", category=UserWarning)

# Register cleanup at exit if needed
import atexit
atexit.register(cleanup_resources)

import re

def sanitize_name(name):
    # Replace any non-alphanumeric characters with underscores
    return re.sub(r'\W+', '_', name)

def get_collection_name(username, project_name):
    username = sanitize_name(username)
    project_name = sanitize_name(project_name)
    return f"NV_{username}_{project_name}"


def connect_to_milvus():
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

def create_collection(collection_name):
    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
        ]
        schema = CollectionSchema(fields, description=f"Collection for {collection_name}")
        collection = Collection(collection_name, schema)
        collection.create_index(
            field_name="embedding",
            index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        )
        collection.load()
        return collection
    else:
        collection = Collection(collection_name)
        collection.load()
        return collection

def embed_text(text):
    inputs = tokenizer_retriever(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_retriever(**inputs)
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return embedding[0].cpu().numpy()

def convert_pdf_to_md(pdf_bytes, output_md_path, output_img_dir):
    os.makedirs(output_img_dir, exist_ok=True)
    
    config_dict = {
        "output_format": "markdown",
        "use_llm": False,
        "force_ocr": False,
        "strip_existing_ocr": False
    }
    config_parser = ConfigParser(config_dict)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service()
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_bytes)
        temp_pdf_path = temp_pdf.name

    try:
        rendered = converter(temp_pdf_path)
        text, _, images = text_from_rendered(rendered)

        for img_name, img_obj in images.items():
            img_path = os.path.join(output_img_dir, f"{img_name}.png")
            img_buffer = BytesIO()
            img_obj.save(img_buffer, format="PNG")
            with open(img_path, "wb") as img_file:
                img_file.write(img_buffer.getvalue())
            text = text.replace(f"![]({img_name})", f"![]({img_path})")

        text = fix_markdown_tables(text)

        with open(output_md_path, "w", encoding="utf-8") as md_file:
            md_file.write(text)
        print(f"Converted PDF to Markdown: {output_md_path}")
        return text
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

def fix_markdown_tables(text):
    lines = text.split("\n")
    new_lines = []
    inside_table = False
    for line in lines:
        if "|" in line:
            if not inside_table:
                inside_table = True
                new_lines.append(line)
                new_lines.append("|---" * (line.count("|") - 1) + "|")
            else:
                new_lines.append(line)
        else:
            inside_table = False
            new_lines.append(line)
    return "\n".join(new_lines)

def index_markdown(username, project_name):
    user_dir = Path(BASE_DIR) / username
    parsed_dir = user_dir / f"{project_name}_parsed"
    md_files = list(parsed_dir.glob("*.md"))
    if not md_files:
        return {"error": "No Markdown files found for indexing"}
    
    connect_to_milvus()
    collection_name = get_collection_name(username, project_name)
    
    if utility.has_collection(collection_name):
        old_collection = Collection(collection_name)
        old_dim = None
        for field in old_collection.schema.fields:
            if field.name == "embedding":
                old_dim = field.params.get("dim")
                break
        if old_dim is not None and old_dim != VECTOR_DIMENSION:
            print(f"Dropping collection {collection_name} due to dimension mismatch")
            utility.drop_collection(collection_name)
    
    collection = create_collection(collection_name)
    
    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            markdown_text = f.read()
        text_chunks = [chunk.strip() for chunk in markdown_text.split("\n\n") if len(chunk.strip()) > 10]
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(text_chunks), batch_size):
            batch_chunks = text_chunks[i:i+batch_size]
            batch_embeddings = [embed_text(chunk).tolist() for chunk in batch_chunks]
            all_embeddings.extend(batch_embeddings)
            print(f"Processed batch {i//batch_size + 1}")
        insert_data = [text_chunks, all_embeddings]
        collection.insert(insert_data)
        collection.flush()
    
    return {"message": f"Indexed {len(md_files)} Markdown files in {collection_name}"}

def load_retriever_model():
    tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_MODEL_DIR)
    model = AutoModel.from_pretrained(RETRIEVER_MODEL_DIR)
    return tokenizer, model

def embed_query(query, retriever_model, retriever_tokenizer):
    inputs = retriever_tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = retriever_model(**inputs)
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding_np = embedding[0].cpu().numpy()
        
        # Use the actual vector dimension from your model
        expected_dim = VECTOR_DIMENSION  # Make sure VECTOR_DIMENSION is 768
        
        actual_dim = embedding_np.shape[0]
        if actual_dim != expected_dim:
            if actual_dim > expected_dim:
                embedding_np = embedding_np[:expected_dim]
            else:
                padding = np.zeros(expected_dim - actual_dim)
                embedding_np = np.concatenate([embedding_np, padding])
        return embedding_np.tolist()

def retrieve_documents(collection, query_embedding, limit=10):
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
    if not documents:
        return [], []
    pairs = [(query, doc) for doc in documents]
    scores = reranker_model.predict(pairs)
    ranked_results = [(doc, score) for doc, score in zip(documents, scores)]
    ranked_results.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in ranked_results[:top_k]]
    top_scores = [float(score) for _, score in ranked_results[:top_k]]
    return top_docs, top_scores

def extract_source(text):
    match = re.match(r"Source:\s*(.+?)\n", text)
    return match.group(1) if match else "Unknown"

def load_restricted_words():
    if os.path.exists(RESTRICTED_WORDS_FILE):
        with open(RESTRICTED_WORDS_FILE, 'r') as f:
            return [line.strip().lower() for line in f.readlines()]
    return ["hack", "exploit"]

def validate_query(query):
    restricted_words = load_restricted_words()
    if len(query) < MIN_QUERY_LENGTH:
        raise ValueError("Invalid query: Too short.")
    if any(word in query.lower() for word in restricted_words):
        raise ValueError("Invalid query: Contains restricted content.")
    sql_injection_patterns = [
        r"\b(select|insert|update|delete|drop|alter)\b",
        r"\b(union|into|from|where)\b",
        r"--|/\*|\*/"
    ]
    for pattern in sql_injection_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            raise ValueError("Invalid query: Potential SQL injection attempt.")
    return query

def filter_response(response, nlp):
    if any(word in response.lower() for word in PROFANITY_LIST):
        return "[Response blocked due to inappropriate content.]"
    doc = nlp(response)
    redacted_response = response
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
            redacted_response = redacted_response.replace(ent.text, "[REDACTED]")
    return redacted_response

def generate_response(prompt):
    inputs = tokenizer_phi(prompt, return_tensors="pt").to("cuda")
    input_length = inputs["input_ids"].shape[1]
    outputs = model_phi.generate(**inputs, max_new_tokens=200)
    generated_sequence = outputs[0, input_length:]
    return tokenizer_phi.decode(generated_sequence, skip_special_tokens=True)

def process_query(username, project_name, query, top_k=DEFAULT_TOP_K):
    connect_to_milvus()
    collection_name = get_collection_name(username, project_name)
    collection = Collection(collection_name)
    collection.load()

    query = validate_query(query)
    query_embedding = embed_query(query, model_retriever, tokenizer_retriever)
    retrieved_docs, _ = retrieve_documents(collection, query_embedding, limit=top_k * 2)
    reranked_docs, confidence_scores = rerank_documents(reranker_model, query, retrieved_docs, top_k)
    sources = [extract_source(doc) for doc in reranked_docs]
    unique_sources = list(set(sources))
    context = "\n".join(reranked_docs)
    # Truncate context if needed
    if len(tokenizer_phi.encode(context)) > 1500:
        context = tokenizer_phi.decode(tokenizer_phi.encode(context)[:1500])
    prompt = (
        f"Context: {context}\n\n"
        f"Question: {query}\n\n"
        "Only provide the exact answer to the question without additional information.\n"
        "Answer:"
    )
    answer = generate_response(prompt)
    final_response = filter_response(answer, nlp)
    return {
        "query": query,
        "reference_text": context,
        "response": final_response,
        "reference_doc": unique_sources,
        "confidence_score": max(confidence_scores) if confidence_scores else 0.0
    }

# ---- Gradio wrapper functions ----

def convert_and_index(username, project_name, pdf_file):
    """
    This function:
      1. Saves the uploaded PDF.
      2. Converts it to Markdown.
      3. Indexes the Markdown file in Milvus.
    """
    if pdf_file is None:
        return "Please upload a PDF file."
    
    # pdf_file is now raw bytes (because type="binary" returns the binary data)
    pdf_bytes = pdf_file
    
    # Define directories
    user_dir = Path(BASE_DIR) / username
    project_dir = user_dir / project_name
    parsed_dir = user_dir / f"{project_name}_parsed"
    project_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    
    # For file naming, use a default name if not provided by the user
    filename = "uploaded.pdf"
    pdf_path = project_dir / filename
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    
    base_name = os.path.splitext(filename)[0]
    output_md_path = parsed_dir / f"{base_name}.md"
    output_img_dir = parsed_dir / f"{base_name}_images"
    output_img_dir.mkdir(exist_ok=True)
    
    # Convert PDF to Markdown
    md_text = convert_pdf_to_md(pdf_bytes, output_md_path, output_img_dir)
    # Index the Markdown file(s)
    index_result = index_markdown(username, project_name)
    
    return f"Conversion complete!\nPDF saved at: {pdf_path}\nMarkdown saved at: {output_md_path}\nIndexing result: {index_result}"

def query_interface(username, project_name, query, top_k):
    try:
        result = process_query(username, project_name, query, top_k)
        out_str = f"Query: {result['query']}\n\nResponse: {result['response']}\n\nReferences: {result['reference_doc']}\nConfidence: {result['confidence_score']}"
        return out_str
    except Exception as e:
        return f"Error: {str(e)}"

# ---- Gradio UI Setup ----

with gr.Blocks(title="RAG Application Demo") as demo:
    gr.Markdown("## RAG Application Demo")
        
    with gr.Tab("Upload, Convert & Index"):
        gr.Markdown("Upload a PDF file, convert it to Markdown, and index the content.")
        with gr.Row():
            username_input = gr.Textbox(label="Username", placeholder="Enter username")
            project_input = gr.Textbox(label="Project Name", placeholder="Enter project name")
        # Set type="binary" so that pdf_file is raw bytes
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="binary")
        convert_btn = gr.Button("Convert and Index")
        convert_output = gr.Textbox(label="Output", interactive=False)
        convert_btn.click(convert_and_index, inputs=[username_input, project_input, pdf_input], outputs=convert_output)

    
    with gr.Tab("Query"):
        gr.Markdown("Enter a query to generate a response based on the indexed content.")
        with gr.Row():
            q_username = gr.Textbox(label="Username", placeholder="Enter username")
            q_project = gr.Textbox(label="Project Name", placeholder="Enter project name")
        query_input = gr.Textbox(label="Query", placeholder="Enter your question")
        topk_input = gr.Number(label="Top K", value=DEFAULT_TOP_K)
        query_btn = gr.Button("Generate Response")
        query_output = gr.Textbox(label="Response", interactive=False, lines=10)
        query_btn.click(query_interface, inputs=[q_username, q_project, query_input, topk_input], outputs=query_output)

if __name__ == "__main__":
    demo.launch()