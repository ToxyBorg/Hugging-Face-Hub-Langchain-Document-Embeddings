# Hugging-Face-Hub-Langchain-Document-Embeddings
Using Hugging Face Hub Embeddings with Langchain document loaders to do some query answering

# # STEP 0:
  
  ## RENAMING THE .example.env TO JUST .env AND MODIFYING WHAT'S NECESSARY:
    HUGGINGFACEHUB_API_TOKEN="your hugging face hub api token"
    HUGGINGFACE_REPO_ID="google/flan-ul2"

    DIRECTORY_DOCUMENTS_TO_LOAD="./data/documents"
    DIRECTORY_FOR_DOCUMENTS_JSON_CHUNKS="./data/chunked_data"

    SAVING_EMBEDDINGS_FILE_NAME="default HUGGINGFACEHUB Embeddings"
    SAVING_EMBEDDINGS_DIRECTORY="./data/embeddings_data"

    SAVING_VECTORSTORE_FILE_NAME="default HUGGINGFACEHUB VectorStore"
    SAVING_VECTORSTORE_DIRECTORY="./data/vectorstore_data"

    SAVING_SIMILARITY_SEARCH_DOCS_DIRECTORY="./data/similarity_search_docs"
    SAVING_SIMILARITY_SEARCH_DOCS_FILE_NAME="default similarity search docs"
    
  ## INSTALL REQUIRED PACKAGES:
    pip install -r requirements.txt

# # STEP 1 LOADING AND CHUNKING THE DOCUMENTS:

  ## The function load_documents:
    This code is a Python function that loads documents from a directory and returns a list of dictionaries containing the name of each document and its chunks. 
    The function uses the langchain package to load documents from different file types such as pdf or unstructured files. 
    It then splits each document into smaller chunks using the CharacterTextSplitter class from the same package. 
    The chunks are then saved in a dictionary format with keys such as “chunk_1”, “chunk_2”, etc. 
    
    Finally, the function returns a list of dictionaries containing the name of each document and its chunks.
    
  ## The function save_documents:
    This code defines a function called save_documents that saves a list of objects to JSON files. 
    
    Each object in the list should have two properties: 
      the name of the document that was chunked, and the chunked data itself. 
    
    The JSON file should be named after the document name, with "Chunks" appended to the end of the name. 
    The content of the JSON file should be the chunked data. 
    The function uses the os and json modules to create the directory for the chunked data if it doesn't exist, 
    and to save the documents to JSON files with dynamic names. 
    
    The resulting JSON files are saved in the directory specified by the save_json_chunks_directory argument.

# # STEP 2 CREATING AND SAVING THE EMBEDDINGS:

  ## The function create_embeddings:
    This code creates embeddings for a list of documents stored in JSON format. 
    The create_embeddings function takes:
        - a directory path as an argument, which contains JSON files with documents to be processed. 
    It uses the HuggingFaceHubEmbeddings object to create embeddings for each document and appends them to a list. 
    
    The function then returns the list of embeddings.

  ## The function save_embeddings:
    This function takes in three parameters: 
      "embeddings" which is an instance of the "Embeddings" class, 
      "saving_embeddings_file_name" which is a string representing the name of the file to be saved, and 
      "saving_embeddings_directory" which is a string representing the path to the directory where the file will be saved.
      
    The function first creates a directory at the specified path if it does not already exist. 
    It then creates a file path by joining the directory path and file name with a ".pkl" extension. 
    
    Finally, it saves the embeddings object to the binary file using the "pickle" module.

# # STEP 3 CREATING AND SAVING VECTORSTORES:

  ## The function create_vectorstore_from_json:
    This code creates a vector store from JSON files. The function takes two arguments: 
        - json_files_directory and huggingfacehub_api_token. 
        
    The json_files_directory argument is the directory where the JSON files are stored. 
    The huggingfacehub_api_token argument is the API token for Hugging Face Hub.
    The function loads the HuggingFaceHubEmbeddings object and the embeddings from disk. 
    It then creates an empty list to store the texts. 
    The function loops through each file in the directory and adds the text to the list of texts. 
    It then combines the texts and embeddings into a list of tuples.

    Finally, it creates a FAISS object from the embeddings and text embeddings and returns it.
    
  ## The function load_embeddings:
    This code defines a function called load_embeddings that loads embeddings from a file using the pickle module. 
 
    The function takes one argument: 
        file_path which is the path to the file containing the embeddings. 
    
    The function: 
        opens the file in binary mode, 
        loads the embeddings using pickle.load(), and 
        returns the embeddings.
    
  ## The function save_vectorstore:
    This code defines a function called save_vectorstore that saves a FAISS index as a file at the specified directory path and file name. 
    It imports the os module and the FAISS class from the langchain.vectorstores.faiss module. 
    
    The function takes three arguments: 
        vectorstore which is the FAISS index to be saved, 
        directory_path which is the path to the directory where the file will be saved, and 
        file_name which is the name of the file to be saved. 
    
    The function:
        creates the directory if it doesn't exist, 
        creates the file path, and 
        saves the FAISS index to the file.

# # STEP 4 USING THE CREATED VECTOR STORE FROM EMBEDDINGS TO QUERY THE DOCS

  ## create_similarity_search_docs: 
    This code defines a function named create_similarity_search_docs that takes in three arguments: 
        - query, 
        - huggingfacehub_api_token, and 
        - path_to_vectorstore. 
    
    The function returns a list of documents that are most similar to the query.
    The function first loads the environment variables from the .env file using the load_dotenv() method. 
    It then sets the values of two variables named saving_vectorstore_file_name and saving_vectorstore_directory to the values of the environment variables with the same names. 
    It then creates a variable named vectorstore_path which is set to the path of the vectorstore file.
    The function then loads an object of type HuggingFaceHubEmbeddings using the HuggingFaceHubEmbeddings() method. 
    It then loads a FAISS vectorstore using the FAISS.load_local() method. 
    
    The function then finds the most similar documents to the query using the faiss.similarity_search()

  ## Q_and_A_implementation: 
    This function is used to implement a question answering system. 
    The function takes in a list of Document objects, a query string, and two optional parameters for the Hugging Face Hub API token and repository ID.
    The function uses the HuggingFaceHub class from the llms module to load a pre-trained language model from the Hugging Face Hub. 
    It then uses the load_qa_chain() function from the question_answering module to load a question answering chain. 
    
    Finally, it uses the chain to find the answer to the query.
