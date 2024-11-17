import os
import hashlib
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import FunctionTool

def calculate_file_hash(file_path):
    """Calculate the SHA-256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            hash_sha256.update(byte_block)
    return hash_sha256.hexdigest()

def text_rag(query, uploaded_file=None):
    DATA_DIR = "./data"
    PERSIST_DIR = "./storage"
    hash_file_path = os.path.join(PERSIST_DIR, "pdf_hash.txt")
    
    os.makedirs(DATA_DIR, exist_ok=True)  # Ensure data directory exists
    os.makedirs(PERSIST_DIR, exist_ok=True)  # Ensure persist directory exists

    # Load existing hash if it exists
    current_pdf_hash = None
    if os.path.exists(hash_file_path):
        with open(hash_file_path, "r") as hash_file:
            current_pdf_hash = hash_file.read().strip()

    # Check if a new file is uploaded
    if uploaded_file:
        # Save the uploaded file to the data directory
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Calculate the hash of the uploaded PDF
        new_pdf_hash = calculate_file_hash(file_path)

        # Compare the new hash with the existing hash
        if new_pdf_hash != current_pdf_hash:
            # Clear existing storage to ensure new embeddings
            for filename in os.listdir(PERSIST_DIR):
                os.remove(os.path.join(PERSIST_DIR, filename))

            # Load documents from the newly uploaded PDF
            documents = SimpleDirectoryReader(DATA_DIR).load_data()
            
            # Create a new index and persist it
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=PERSIST_DIR)

            # Update the hash file with the new hash
            with open(hash_file_path, "w") as hash_file:
                hash_file.write(new_pdf_hash)
        else:
            print("No changes detected in the PDF. Using the existing index.")

    # Load documents from the current PDF file in the data directory
    documents = SimpleDirectoryReader(DATA_DIR).load_data()

    # Check if the index already exists
    index_path = os.path.join(PERSIST_DIR, "index.json")
    if os.path.exists(index_path):
        print("Loading existing index...")
        index = VectorStoreIndex.load(index_path)
    else:
        # Create a new index if it does not exist
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)

    # Perform query on the index
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response

text_rag_tool = FunctionTool.from_defaults(
    fn=text_rag,
    name="text_rag",
    description="Retrieval Augmented Generation tool to handle text-based PDF queries. Please make sure that the query is valid and not contain information of the previous pdf "
)