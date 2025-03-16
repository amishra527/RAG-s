import gradio as gr
import requests
import os
from pathlib import Path
import shutil

# FastAPI endpoints
API_BASE_URL = "http://127.0.0.1:10001"
API_GEN_URL = "http://127.0.0.1:10002"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/convert"
INDEX_ENDPOINT = f"{API_BASE_URL}/index"
GENERATE_ENDPOINT = f"{API_GEN_URL}/generate"

# Directory for uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def upload_and_convert(username, project_name, pdf_file):
    """Uploads a PDF to the FastAPI server and converts it to Markdown."""
    if not pdf_file:
        return "Please upload a PDF file."
    
    file_path = Path(pdf_file.name)  # Use the original file path directly

    files = {"file": open(file_path, "rb")}
    data = {"username": username, "project_name": project_name}
    response = requests.post(UPLOAD_ENDPOINT, files=files, data=data)
    
    if response.status_code == 200:
        return response.json()["markdown_content"], str(file_path)  # Convert Path to string
    else:
        return f"Error: {response.json()}"


def index_markdown(username, project_name):
    """Indexes the converted Markdown content in Milvus."""
    data = {"username": username, "project_name": project_name}
    response = requests.post(INDEX_ENDPOINT, json=data)
    return response.json()["message"] if response.status_code == 200 else "Indexing failed."

def generate_response(username, project_name, query):
    """Queries the indexed data and generates a response."""
    data = {"username": username, "project_name": project_name, "query": query}
    response = requests.post(GENERATE_ENDPOINT, json=data)
    return response.json()["response"] if response.status_code == 200 else "Query failed."

def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## PDF Viewer & Chat with Indexed Content")
        
        with gr.Row():
            with gr.Column():
                username = gr.Textbox(label="Username")
                project_name = gr.Textbox(label="Project Name")
                # pdf_file = gr.File(label="Upload PDF", type="file")
                pdf_file = gr.File(label="Upload PDF", type="filepath")  # Change 'file' to 'filepath'

                upload_btn = gr.Button("Upload & Convert")
                index_btn = gr.Button("Index Document")
                
                pdf_viewer = gr.File(label="View PDF", interactive=False)
                markdown_output = gr.Textbox(label="Extracted Text", interactive=False, lines=10)
            
            with gr.Column():
                query_input = gr.Textbox(label="Ask a Question")
                query_btn = gr.Button("Generate Response")
                response_output = gr.Textbox(label="Response", interactive=False, lines=5)
        
        upload_btn.click(upload_and_convert, inputs=[username, project_name, pdf_file], outputs=[markdown_output, pdf_viewer])
        index_btn.click(index_markdown, inputs=[username, project_name], outputs=gr.Textbox())
        query_btn.click(generate_response, inputs=[username, project_name, query_input], outputs=response_output)
        
    return demo

demo = gradio_interface()

demo.launch(server_name="0.0.0.0", server_port=7860)
