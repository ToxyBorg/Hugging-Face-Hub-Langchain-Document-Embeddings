"""
    The code is used to create a similarity search document and use it to answer a query.

    The create_similarity_search_docs() function creates a similarity search document using the Hugging Face API. 
    The function takes a query and an API token as input and returns a similarity search document.

    The Q_and_A_implementation() function takes the query, API token, and similarity search document as input and returns an answer to the query.

    Finally, the answer is printed using the print() function.
"""

import os
import sys
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from HELPERS.step_4A_create_similarity_search_docs import (
    create_similarity_search_docs,
)
from HELPERS.step_4B_using_similarity_search_docs_for_QA import Q_and_A_implementation


"""################# CALLING THE FUNCTION #################"""


huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

query = "What is is document about?"

print(
    "\n####################### CREATING SIMILARITY SEARCH DOCS ########################\n"
)

similarity_search_docs = create_similarity_search_docs(
    query=query,
    huggingfacehub_api_token=huggingfacehub_api_token,
)

print(
    "\n####################### USING SIMILARITY SEARCH DOCS AND QUERY ########################\n"
)

Q_and_A_answer = Q_and_A_implementation(
    query=query,
    huggingfacehub_api_token=huggingfacehub_api_token,
    similarity_search_docs=similarity_search_docs,
)


print("\n####################### ANSWERING THE QUERY ########################\n")

print(Q_and_A_answer)
