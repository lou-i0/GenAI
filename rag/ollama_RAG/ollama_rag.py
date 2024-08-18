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

#%% Initial declared variables
#----------------------------
pdf_path        = './pdfdir'           # Location of stored PDF documents
txt_path        = './txt_dir'          # Location of stored text documents
pdf_db          = 'pdf_db'             # Location of created vector Database
model_select    = 'gemma:2b'           # LLM model from Ollama to use.


#%% ? Function to create vector embeddings from chunks
#===================================================
def get_embedding():
    embeddings = OllamaEmbeddings(model = model_select)
    return  embeddings


#%% 5. Function to Query LLM with RAG vector embedded RAG 
#=====================================================
def llm_rag_qry(query_text: str, model = model_select):
    embedding_func = get_embedding()  
    db = Chroma(persist_directory= pdf_db, embedding_function=embedding_func)

    results = db.similarity_search_with_score(query = query_text, k = 10)
    print(results[0])

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_temp= f" Answer the question, based only on the following context:{context_text}. Answer the question based on the above context {query_text}."
    prompt_template = ChatPromptTemplate.from_template(prompt_temp)
   

    model = Ollama(model = model)
    response_text = model.invoke(prompt_temp)
    print(response_text)

#///////////////////////////////////////////////////
#%% Main Query
#====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    llm_rag_qry(query_text)

#%% to run code in terminal
#==================================================
if __name__ == "__main__":
    main()

