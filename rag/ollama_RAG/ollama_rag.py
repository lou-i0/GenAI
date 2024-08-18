#%% Import Relevant Libraries
#============================
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader   # Load in and prep PDf files
from langchain_text_splitters import RecursiveCharacterTextSplitter         # splits up text into chunks
from langchain.schema.document import Document                              # ??
from langchain_community.embeddings.ollama import OllamaEmbeddings          # Ability to using local LLMs for embedding
from langchain.vectorstores.chroma import Chroma                            # for the Chroma vector Database
from tqdm import tqdm                                                       # progress bars for iterations
from langchain_community.llms.ollama import Ollama

from langchain.prompts import ChatPromptTemplate
#%% Function to Load in PDF Documents 
#========================
def pdf_load():
    pdf_loader = PyPDFDirectoryLoader('./pdfdir')
    return pdf_loader.load()

# #%% Test pdf_load Function
# #-------------------------
# docs = pdf_load()
# print(docs[0])

#%% Function to split the documents into chunks ready for embeddings 
#===================================================================
def split_pdfs(docs:list[Document]):
    txt_splitter = RecursiveCharacterTextSplitter(
                                                    chunk_size          = 800
                                                    ,chunk_overlap      = 80
                                                    ,length_function    = len
                                                    ,is_separator_regex = False
                                                 )
    return txt_splitter.split_documents(docs)

#%% Function to create vector embeddings from chunks
#===================================================
def get_embedding():
    embeddings = OllamaEmbeddings(model='solar')
    return  embeddings

#%% Function to create vector database
#=====================================
def chroma_db(chunks: list[Document]):
    db = Chroma(
                persist_directory   = 'pdfdb'
                ,embedding_function = get_embedding()  
                )
    db.add_documents(new_chunks, ids = new_chunk_ids)
    db.persist()

#%% Function to Query LLM with RAG vector embedded RAG 
#=====================================================
def llm_rag_qry(query_text: str):
    embedding_func = get_embedding()  


    db = Chroma(persist_directory= 'pdfdb', embedding_function=embedding_func)

    results = db.similarity_search_with_score(query = query_text, k = 2)
    print(results[0])

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_temp= f" Answer the question, based only on the following context:{context_text}. Answer the question based on the above context {query_text}."
    prompt_template = ChatPromptTemplate.from_template(prompt_temp)
   

    model = Ollama(model='solar')
    response_text = model.invoke(prompt_template)
    print(response_text)


#%% main query 
#-----------------------------------
def main():
    # process pdfs
    docs    = pdf_load()
    #split into chunks
    chunks  = split_pdfs(docs)
    

    last_page_id = None
    curr_chunk_idx = 0

    for chunk in tqdm(chunks):
        source = chunk.metadata.get('source')[0:18]
        page = chunk.metadata.get('page')
        current_page_id = f"{source}:{page}"
        chunk.metadata['id'] = current_page_id

    chunk_with_ids = chunks

    # Add or update documents into db
    # existing_items  = db.get(include = [])
    # existing_ids     = set(existing_items["ids"])


    # print(f"Number of existing documents in Database: {len(existing_ids)}")

    # new_chunks = []
    # for chunk in tqdm(chunk_with_ids):
    #     if chunk.metadata['id'] not in existing_ids:
    #         new_chunks.append(chunk)
    # new_chunk_ids = [chunk.metadata['id'] for chunk in new_chunks]
    # print("...checking to add documents...")
    # chroma_db(chunks= new_chunks)
    # print('... Database updated if required...')

    llm_rag_qry("Provide a summary of what this information refers to? ")

main()

# %%
