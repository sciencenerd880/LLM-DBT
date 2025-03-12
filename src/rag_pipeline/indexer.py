"""
Reference: https://huggingface.co/learn/cookbook/en/rag_with_hf_and_milvus

Description: indexer.py
- Loads the PDFs stored at ./data/submitted_docs/
- Splits each of the PDF into overlapping text chunks
- Then use HF embedding model to generate the embeddings
- Stores the embedding in Milvus with the metadata as well 
(e.g. document_id, filename, submission_date, page_number, text/content)
"""

import os

import uuid
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from tqdm import tqdm

# Initialize embedding model
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()

# Hyperparameters for chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize Milvus client
milvus_client = MilvusClient(uri="././milvus_data/2025_budget.db")

# Define my collection name (versioned for tracking)
COLLECTION_NAME = "sg_budget"

def emb_text(text):
    """Generate embedding for a given text chunk."""
    return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

def process_pdf(pdf_path):
    """Process a single PDF, split into chunks, and insert into Milvus."""
    
    # Extract filename (for metadata tracking)
    filename = os.path.basename(pdf_path)
    doc_id = str(uuid.uuid4())  # Generate a unique document ID
    submission_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Processing file: {filename}, Document ID: {doc_id}")

    # Load and split the PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(docs)
    
    text_lines = []
    for chunk in chunks:
        text_lines.append({
            "content": chunk.page_content,
            "page": chunk.metadata.get("page", -1)  # Extract page number
        })

    print(f"Done splitting into {len(text_lines)} chunks.")

    # Create collection if it doesn't exist
    if not milvus_client.has_collection(COLLECTION_NAME):
        print(f"Creating collection: {COLLECTION_NAME}")
        milvus_client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=EMBEDDING_DIM,
            metric_type="IP",
            consistency_level="Strong"
        )

    # Generate embeddings and insert into Milvus
    data = []
    for i, chunk in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        embedding = emb_text(chunk["content"])
        data.append({
            "id": i, # Unique ID per chunk
            "vector": embedding,
            "text": chunk["content"],
            "filename": filename,
            "document_id": doc_id,
            "submission_date": submission_date,
            "page_number": chunk["page"]
        })

    # Insert into Milvus
    insert_res = milvus_client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"Inserted {insert_res['insert_count']} chunks for {filename}\n")

if __name__ == "__main__":
    # Path to uploaded PDFs
    pdf_folder = "./data/sg_budget/"
    
    # Process all PDFs in the folder
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            process_pdf(os.path.join(pdf_folder, pdf_file))