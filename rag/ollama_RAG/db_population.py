#%% Import Relevant Libraries
#============================
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader   # Load in and prep PDf files
from langchain_community.document_loaders.text import TextLoader            # Load in and prep txt files 
from langchain_text_splitters import RecursiveCharacterTextSplitter         # splits up text into chunks
from langchain.schema.document import Document                              # ??
from langchain_community.embeddings.ollama import OllamaEmbeddings          # Ability to using local LLMs for embedding
import argparse
from langchain_chroma import Chroma
from tqdm import tqdm                                                       # progress bars for iterations
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate

import time     as tim                                                      # Time operations
import os       as os                                                       # Operation system actvities
import shutil   as shu                                                      # Shell operations
import argparse
from ollama_rag import get_embedding

#%% Initial declared variables
#----------------------------
pdf_path        = './pdfdir'           # Location of stored PDF documents
txt_path        = './txt_dir'          # Location of stored text documents
pdf_db          = 'pdf_db'             # Location of created vector Database
model_select    = 'gemma:2b'           # LLM model from Ollama to use.

#%% 0. Function to remove Vector Database
def remove_database():
    if os.path.exists(pdf_db):
        shu.rmtree(pdf_db)

#%% 1. Function to Load in PDF Documents 
#=======================================
def pdf_load():
    pdf_loader = PyPDFDirectoryLoader(pdf_path)
    return pdf_loader.load()

# #%% 1. Function to Load in text Documents 
# #=======================================
# def txt_load():
#     txt_loader = TextLoader(file_path=txt_path,encoding='UTF-8')
#     return txt_loader.load()

#%% 2. Function to split the documents into chunks ready for embeddings 
#===================================================================
def split_docs(docs:list[Document]):
    txt_splitter = RecursiveCharacterTextSplitter(
                                                    chunk_size          = 800
                                                    ,chunk_overlap      = 80
                                                    ,length_function    = len
                                                    ,is_separator_regex = False
                                                 )
    return txt_splitter.split_documents(docs)
#%% 3. Function to create a chunk id 
#====================================
def get_chunk_ids(chunks):

    for chunk in tqdm(chunks, desc="getting Chunk IDS"):
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        # Unique ID using source, page, and chunk index within the page
        chunk_id = f"{source}:{page}:{chunks.index(chunk)}"
        chunk.metadata['id'] = chunk_id

    return chunks

#%% 4. Function to create vector database
#=====================================
def add_chroma_db(chunks: list[Document]):
    # try:
    print('4.1 Creating the Database')
    tim.sleep(0.2)
    db = Chroma(
                persist_directory   = pdf_db
                ,embedding_function = get_embedding()  
                )

    print('4.2 Creating ids for chunks')
    tim.sleep(0.2)
    chunks_with_ids = get_chunk_ids(chunks)

    print('4.3 Upserting chunks')
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in tqdm(chunks_with_ids,desc= "Checking if ids already in database"):
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"⚠️ Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(documents= new_chunks, ids=new_chunk_ids)
        print(f"➕{len(new_chunks)} Document converted into vectors and added to Database")
        tim.sleep(0.2)
    else:
        print("✅ No new documents to add")
    
    

    # except Exception as e:
    #     print(f'There is a odd error: {e}')
#////////////////////////////////////
#%% main query 
#-----------------------------------
def main():

    print('0. Cleaning up last session')
    tim.sleep(1)
    remove_database()

    print('1. Processing the Documents')
    tim.sleep(1)
    docs    = pdf_load()        # process pdfs

    print('2. Splitting PDF documents into chunks')
    tim.sleep(1)
    chunks  = split_docs(docs)  # split into chunks
    
    print("3. Producing ID's for chunks made" )
    tim.sleep(1)
    chunks = get_chunk_ids(chunks)

    print("4. Preparing the Vector Database")
    tim.sleep(1)
    add_chroma_db(chunks = chunks)

    print('check whats in the database')

main()
    
#%% Used to run script from terminal    
if __name__ == "__main__":
    main()
