"""
    This code defines a function called save_documents that saves a list of objects to JSON files. 
    
    Each object in the list should have two properties: 
    the name of the document that was chunked, and the chunked data itself. 
    
    The JSON file should be named after the document name, with "Chunks" appended to the end of the name. 
    
    The content of the JSON file should be the chunked data. 
    
    The function uses the os and json modules to create the directory for the chunked data if it doesn't exist, 
    and to save the documents to JSON files with dynamic names. 
    
    The resulting JSON files are saved in the directory specified by the save_json_chunks_directory argument.
"""

import os
import json
import sys
from dotenv import load_dotenv

from typing import List, Dict, Union

load_dotenv()  # Load environment variables from .env file


def save_documents(
    documents: List[Dict[str, Union[str, List[Dict[str, str]]]]],
    save_json_chunks_directory: str = os.getenv("DIRECTORY_FOR_DOCUMENTS_JSON_CHUNKS"),
) -> None:
    """
    Saves a list of objects to JSON files. Each object in the list should have two properties:
        - the name of the document that was chunked,
        - and the chunked data itself.

    The JSON file should be named after the document name, with "Chunks"
    appended to the end of the name.
    The content of the JSON file should be the chunked data.

    Args:
        - documents (List[Dict[str, Union[str, List[Dict[str, str]]]]]):
            - A list of objects, where each object has two properties:
                - the name of the document that was chunked,
                - and the chunked data itself.
        - save_json_chunks_directory (str): The path to the directory where the JSON files will be saved.

    Returns:
        - None
    """

    # Create directory for chunked data if it doesn't exist
    if not os.path.exists(save_json_chunks_directory):
        os.makedirs(save_json_chunks_directory)

    # Save documents to JSON file with dynamic name
    for doc in documents:
        json_file_path = os.path.join(
            save_json_chunks_directory, f"{doc['name']} Chunks.json"
        )
        with open(json_file_path, "w") as f:
            json.dump(doc["chunks"], f)
