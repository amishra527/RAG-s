from fastapi import FastAPI, UploadFile, File, Form
import os
import tempfile
import requests
from pathlib import Path
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from transformers import AutoTokenizer, AutoModel
import torch
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
from io import BytesIO
import atexit
import multiprocessing
import warnings
import numpy as np

# Clean up leaked multiprocessing resources
def cleanup_resources():
    multiprocessing.active_children()  # Ensure child processes are cleaned up
    warnings.filterwarnings("ignore", category=UserWarning)  # Suppress warnings

app = FastAPI()

# Base Directory for File Storage
BASE_DIR = r"D:\WorkSpace_0\CAI\infrence\ConvData\MarkerData"
os.makedirs(BASE_DIR, exist_ok=True)

# Milvus Configuration
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"

# Embedding Model (Local Path) - Use your retriever model
RETRIEVER_MODEL_DIR = r"D:\WorkSpace_0\CAI\infrence\retriever_model"

# Load the retriever model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_MODEL_DIR)
model = AutoModel.from_pretrained(RETRIEVER_MODEL_DIR)

# Calculate vector dimension from the model
with torch.no_grad():
    # Get sample embedding to determine dimension
    sample_inputs = tokenizer("This is a test", return_tensors="pt")
    sample_outputs = model(**sample_inputs)
    # The hidden state of the [CLS] token is often used as sentence embedding
    VECTOR_DIMENSION = sample_outputs.last_hidden_state[:, 0, :].shape[1]
    print(f"Using vector dimension: {VECTOR_DIMENSION}")

# Set Hugging Face Local Model Path
os.environ["HF_HOME"] = r"D:\WorkSpace_0\CAI\infrence\local_models_marker"

def embed_text(text):
    """Create embeddings using the retriever model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling to get the sentence embedding
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return embedding[0].cpu().numpy()

def get_collection_name(username, project_name):
    """Generate a unique collection name using NV_username_projectname."""
    return f"NV_{username}_{project_name}"

def connect_to_milvus():
    """Connect to Milvus."""
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

def create_collection(collection_name):
    """Create Milvus collection if not exists."""
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

def convert_pdf_to_md(pdf_bytes, output_md_path, output_img_dir):
    """Converts PDF to Markdown and saves extracted images."""
    
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
            # Replace image placeholders in Markdown with correct path
            text = text.replace(f"![]({img_name})", f"![]({img_path})")

        # Improve table formatting (Markdown tables)
        text = fix_markdown_tables(text)

        with open(output_md_path, "w", encoding="utf-8") as md_file:
            md_file.write(text)
        print(f"Converted PDF to Markdown: {output_md_path} with images in {output_img_dir}")
        return text # Return Markdown text for API response if needed
    finally:
        # Cleanup temp file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

def fix_markdown_tables(text):
    """Ensure tables are formatted correctly for Markdown."""
    lines = text.split("\n")
    new_lines = []
    inside_table = False

    for line in lines:
        if "|" in line:  # Detect table rows
            if not inside_table:
                inside_table = True
                # Add header formatting if missing
                new_lines.append(line)
                new_lines.append("|---" * (line.count("|") - 1) + "|")
            else:
                new_lines.append(line)
        else:
            inside_table = False
            new_lines.append(line)

    return "\n".join(new_lines)

@app.post("/convert")
async def convert_pdf(
    username: str = Form(...),
    project_name: str = Form(...),
    file: UploadFile = File(...)
):
    """Converts PDF to Markdown and saves metadata."""
    pdf_bytes = await file.read()

    # Define user, project, and parsed directories
    user_dir = Path(BASE_DIR) / username
    project_dir = user_dir / project_name
    parsed_dir = user_dir / f"{project_name}_parsed"

    # Ensure directories exist    
    project_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    pdf_path = project_dir / file.filename # Original PDF location
    base_name = os.path.splitext(file.filename)[0] # Extract file name without extension
    output_md_path = parsed_dir / f"{base_name}.md" # Markdown file
    output_img_dir = parsed_dir / f"{base_name}_images" # Images directory
    output_img_dir.mkdir(exist_ok=True) # Ensure image directory exists

    # Save the uploaded PDF
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # Call the function to convert PDF to Markdown
    markdown_text = convert_pdf_to_md(pdf_bytes, output_md_path, output_img_dir)

    return {
        "message": "Conversion successful!",
        "pdf_saved_at": str(pdf_path),
        "markdown_saved_at": str(output_md_path),
        "images_saved_in": str(output_img_dir),
        "markdown_content": markdown_text
    }

# Register cleanup at exit
atexit.register(cleanup_resources)

@app.post("/index")
async def index_markdown(username: str, project_name: str):
    """Indexes Markdown content in Milvus after PDF conversion."""
    user_dir = Path(BASE_DIR) / username
    parsed_dir = user_dir / f"{project_name}_parsed"

    md_files = list(parsed_dir.glob("*.md"))
    if not md_files:
        return {"error": "No Markdown files found for indexing"}

    connect_to_milvus()
    collection_name = get_collection_name(username, project_name)
    
    # Check if collection already exists with different dimensions
    if utility.has_collection(collection_name):
        old_collection = Collection(collection_name)
        old_schema = old_collection.schema
        old_dim = None
        for field in old_schema.fields:
            if field.name == "embedding":
                old_dim = field.params.get("dim")
                break
                
        # If dimensions don't match, drop and recreate the collection
        if old_dim is not None and old_dim != VECTOR_DIMENSION:
            print(f"Dropping collection {collection_name} due to dimension mismatch (old: {old_dim}, new: {VECTOR_DIMENSION})")
            utility.drop_collection(collection_name)
    
    collection = create_collection(collection_name)

    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        text_chunks = [chunk.strip() for chunk in markdown_text.split("\n\n") if len(chunk.strip()) > 10]
        
        # Process chunks in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(text_chunks), batch_size):
            batch_chunks = text_chunks[i:i+batch_size]
            batch_embeddings = [embed_text(chunk).tolist() for chunk in batch_chunks]
            all_embeddings.extend(batch_embeddings)
            print(f"Processed batch {i//batch_size + 1}/{(len(text_chunks) + batch_size - 1)//batch_size}")

        insert_data = [text_chunks, all_embeddings]
        collection.insert(insert_data)
        collection.flush()

    return {"message": f"Indexed {len(md_files)} Markdown files in {collection_name}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10001)