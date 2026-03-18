from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI

# Import necessary modules from LangChain for document loading/splitting
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from a .env file (if present)
load_dotenv()

app = FastAPI()

# Initialize OpenAI client using the API key from environment variables
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ----------- Document Loading, Splitting, Pre-embedding Preparation -----------

# Path to `/docs` folder (can be overridden with DOCS_DIRECTORY env var)
DOCS_DIRECTORY = os.environ.get(
    "DOCS_DIRECTORY",
    os.path.join(os.path.dirname(__file__), "docs"),
)


def load_all_documents(directory_path: str):
    """
    Load all .txt and .pdf files from a directory into a list of Document objects.
    """
    if not os.path.isdir(directory_path):
        # No docs folder yet – return an empty list so the app still starts.
        return []

    # Separate loaders for txt and pdf, then concatenate results
    txt_loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
    )
    pdf_loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )

    documents = txt_loader.load() + pdf_loader.load()
    return documents


def split_documents(documents):
    """
    Split documents into chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = splitter.split_documents(documents)
    return split_docs


# Load and split the documents at startup so they're ready for embeddings
documents = load_all_documents(DOCS_DIRECTORY)
split_docs = split_documents(documents)

# Texts prepared for embedding (e.g., to pass to an embeddings API)
texts_for_embedding = [doc.page_content for doc in split_docs]

# ------------------------------------------------------------------------------

class UserRequest(BaseModel):
    response: str

# Define a POST endpoint at '/ask_openai'.
# This endpoint expects a request body matching UserRequest.
# It sends the user's input to OpenAI's GPT-4o-mini model and returns the response.
@app.post("/ask_openai")
async def ask_openai(user_request: UserRequest):
    try:
        completion = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL"),
            messages=[{"role": "user", "content": user_request.response}],
        )
        # Extract the model's reply from the API response.
        answer = completion.choices[0].message.content
        # Return the reply in a JSON object with the key 'openai_response'.
        return {"openai_response": answer}
    except Exception as e:
        # If there's an error (e.g., network, API), raise an HTTP 500 error with details.
        raise HTTPException(status_code=500, detail=str(e))
