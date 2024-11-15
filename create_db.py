# https://github.com/pixegami/rag-tutorial-v2
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from IPython.display import Markdown
import textwrap
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.bedrock import BedrockEmbeddings

CHROMA_PATH = "db"
DATA_PATH = "path/to/papers"
aws_profile_name = "aws-profile-name"
region_name = "region-name"
model_name="amazon.titan-embed-text-v2:0"

def get_embedding_function():
    embeddings = BedrockEmbeddings(
        credentials_profile_name=aws_profile_name,
        region_name=region_name,
        model_id=model_name,
    )
    return embeddings


def load_documents(DATA_PATH):
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def to_markdown(text):
    text = text.replace("•", "  *")
    return Markdown(textwrap.indent(text, " ", predicate=lambda _: True))


documents = load_documents(DATA_PATH)
chunks = split_documents(documents)
add_to_chroma(chunks)
