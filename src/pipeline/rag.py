import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM

# Step 1: Load and preprocess text data
def load_and_split_text(text_files_dir):
    documents = []
    for root, _, files in os.walk(text_files_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                loader = TextLoader(file_path)
                documents.extend(loader.load())
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

# Step 2: Create embeddings and vector store
def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

# Step 3: Set up the LLM and retrieval chain
def setup_qa_chain(vector_store):
    llm = OllamaLLM(model="deepseek-r1:latest")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    )
    return qa_chain

# Step 4: Ask questions
def ask_question(qa_chain, question):
    result = qa_chain.run(question)
    return result

# Example usage
if __name__ == "__main__":
    # Directory containing text files
    text_files_dir = "/home/bowserj/profunc/data/text_output"
    
    # Load and preprocess text data
    texts = load_and_split_text(text_files_dir)
    
    # Create vector store
    vector_store = create_vector_store(texts)
    
    # Set up QA chain
    qa_chain = setup_qa_chain(vector_store)
    
    # Ask a question
    question = "What is the JIG?"
    answer = ask_question(qa_chain, question)
    print(f"Question: {question}\nAnswer: {answer}")
