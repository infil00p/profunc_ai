import os
from langchain.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Directory containing your text files
text_files_dir = "/home/bowserj/profunc/data/text_output"

# File to store non-readable text file paths
non_readable_files_path = "non_readable_files.txt"

# Initialize Ollama with a local LLM
llm = OllamaLLM(model="deepseek-r1:latest")

template = """Evaluate the readability of the following text. Is it clear, coherent, and free of major errors? If not, explain why. Text: {text}"""

prompt  = ChatPromptTemplate.from_template(template)

# Create an LLM chain
readability_chain = prompt | llm

# Function to evaluate readability of a text file
def evaluate_readability(file_path):
    # Load the text file
    loader = TextLoader(file_path)
    document = loader.load()
    text = document[0].page_content  # Extract the text content

    # Evaluate readability using the LLM
    response = readability_chain.invoke(text)
    return response

# Open the file to store non-readable file paths
with open(non_readable_files_path, "w") as non_readable_file:
    # Process all text files in the directory
    for root, _, files in os.walk(text_files_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                print(f"Evaluating {file_path}...")
                
                try:
                    readability_result = evaluate_readability(file_path)
                    
                    # Check if the text is non-readable
                    if "not readable" in readability_result.lower() or "unclear" in readability_result.lower():
                        print(f"Non-readable file: {file_path}")
                        non_readable_file.write(f"{file_path}\n")
                    else:
                        print(f"Readable file: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    non_readable_file.write(f"{file_path} (Error: {e})\n")
