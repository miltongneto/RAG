import PyPDF2
from openai import OpenAI
import chromadb
import uuid
import os

CHUNK_SIZE = 1000
OFFSET = 200

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

chromadb_path = "path/to/save/" # CONFIG YOUR PATH
chroma_client = chromadb.PersistentClient(path=chromadb_path)
collection = chroma_client.create_collection(name="my_collection")

def get_document(document_path):
    """Read a PDF document and return text in string"""
    file = open(document_path, 'rb')
    reader = PyPDF2.PdfReader(file)

    document_text = ""
    for i in range(len(reader.pages)):
        page = reader.pages[i]
        content = page.extract_text()
        document_text += content

    len(document_text)
    return document_text

def split_document(document_text):
    """Split a document in a list of string"""
    documents = []
    for i in range(0, len(document_text), CHUNK_SIZE):
        start = i
        end = i + 1000
        if start != 0:
            start = start - OFFSET
            end =  end - OFFSET
        documents.append(document_text[start:end])
    return documents

def get_embedding(text):
    """Transform a text in a vector using embedding model"""
    embedding = openai_client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    
    return embedding.data[0].embedding

def prepare_documents(documents, document_name):
    """Prepare documents to vector database, generating embedding and metadata"""
    embeddings = []
    metadatas = []
    for i, doc in enumerate(documents):
        embeddings.append(get_embedding(doc))
        metadatas.append({"source": document_name, "partition" : i})

    return embeddings, metadatas

def create_ids(documents):
    """Create a list of IDs for documents"""
    return [str(uuid.uuid4()) for _ in documents]

def insert_data(documents, embeddings, metadatas, ids):
    """Insert data in a ChromaDB collection"""
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Data successfully entered! {len(documents)} Chunks")

def run():
    print("Running prep docs...")
    path = 'data/'

    documents = []
    embeddings = []
    metadatas = []

    documents_names = os.listdir(path)
    documents_names_size = len(documents_names)
    for i, document_name in enumerate(documents_names): 
        print(f"{i+1}/{documents_names_size}: {document_name}")

        document = get_document(os.path.join(path, document_name))
        document_chunks = split_document(document)
        document_embeddings, document_metadatas = prepare_documents(document_chunks, document_name)
        documents.extend(document_chunks)
        embeddings.extend(document_embeddings)
        metadatas.extend(document_metadatas)
    
    ids = create_ids(documents)
    insert_data(documents, embeddings, metadatas, ids)
        
if __name__ == "__main__":
    run()