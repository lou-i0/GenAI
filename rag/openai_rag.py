#%% Import in the relevant libraries
#===============================================
import os                                       # for operating system activities / system folders
import chromadb as cdb                          # for vector databases

from openai import OpenAI                       # allows use of open ai api 
from chromadb.utils import embedding_functions  # to embed the documents/data that will be connected with the LLM

#%% Get Open API key, hook in open AI embedding functions
openai_api_key = os.getenv("OPEN_API_KEY")
openai_embfunc = embedding_functions.OpenAIEmbeddingFunction(
                                                                api_key     = openai_api_key
                                                                ,model_name = "text-embedding-3-small"  
                                                            )
#%% Create chroma database client with persistance?
chroma_client       = cdb.PersistentClient(path = "cdb_pers_store")
doc_collection_name = "openai_doc_collection"
collection          = chroma_client.get_or_create_collection(
                                                            name = doc_collection_name
                                                            ,embedding_function=openai_embfunc
                                                            ) 
#%% Create the client (OpenAI)
client = OpenAI(api_key = openai_api_key)

#%% Creating a thread for conversation with the model? Get a response. Not the end goal , but to test connection to open AI.
# response = client.chat.completions.create(                              # processes a query and gets a response
#                                             model = 'gpt-3.5-turbo'     # Choose Open AI Model to interact with
#                                             ,messages = [               # the context of the model , and the question to answer
#                                                 {"role":"system","content":"You are a sassy and sarcastic, but helpful assistant"}
#                                                 ,{"role":"user","content":"Can you summmarise what is like at Wrexham University?"}
#                                             ]
#                                          )
# print(response.choices[0].message.content)                              # Get the response from the above query


#%% Create a function to load documents from a directory
def get_docs_from_dir(dir_path):
    print("....... Getting the documents for you......")
    docs = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".txt"):
            with open(os.path.join(dir_path, filename)) as file:
                docs.append({"id":filename, "text": file.read()})
    return docs

#%% Create function to split text into "chunks"
#==============================================
def split_text(text, chunk_size = 100, chunk_overlap = 2):
    chunks = []
    start = 0
    while start < len(text):
        end  = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

#%% Load documents from directory and view the results of read in 
#================================
dir_path    = "./documentdata"
docs        = get_docs_from_dir(dir_path=dir_path)

print(docs)
print(f"processed {len(docs)} documents")

#%% Split processed documents into chunks
#========================================
chunk_docs = []
for doc in docs:
    chunks = split_text(doc["text"])
    print(".... splitting text from documents processed into chunks....")
    for i , chunk in enumerate(chunks):
        chunk_docs.append({"id": f"{doc['id']}_chunk{i+1}","text": chunk})

print(chunk_docs)
print(f"Split of {len(docs)} into {len(chunk_docs)}")

#%% Create function to process chunks into embeddings 
#====================================================
def get_embedding(text):
    response = client.embeddings.create(input=text, model = "text-embedding-3-small")
    embedding = response.data[0].embedding
    print("... lets generate those embeddings....")
    return embedding

#%% Create the embeddings from the processed document chunks
#====================================================
for doc in chunk_docs:
    print('.... making embeddings....')
    doc["embedding"] = get_embedding(doc["text"])
    #print(doc['text'])

# print(doc["embedding"])

#%% Upsert document chunk embeddings into chroma
#===============================================
for ddoc in chunk_docs:
    print('... adding embedding to Database ...')
    collection.upsert(ids = doc['id'], documents=doc['text'],embeddings=doc['embedding'])

# Create function to query documents from Database
#=================================================
def qry_docs(question, n_results =2):
    results = collection.query(query_texts = question,n_results = n_results)
    # extract them useful chunks
    useful_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print('... getting the right chunks ...')
    return useful_chunks

# Generate response from OpenAI
#################################
def create_response(question, useful_chunks):
    context = "\n\n".join(useful_chunks)
    prompt = (
                "You are a sassy and sarcastic, but helpful butler called Dave"
                "Use the following context to answer the question. If you don't know the answer, just say so."
                "Try and keep your answer to be concise, but informative with some humour."
                "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
              )
    
    response = client.chat.completions.create(                              # processes a query and gets a response
                                                model = 'gpt-3.5-turbo'     # Choose Open AI Model to interact with
                                                ,messages = [               # the context of the model , and the question to answer
                                                    {"role":"system","content":prompt}
                                                    ,{"role":"user","content":question}
                                                ]
                                            )
    answer = response.choices[0].message

    return answer

# Create a query to test the response with RAG embeddings 
#========================================================
question = "what can you tell me about wrexham university overall?"
useful_chunks = qry_docs(question = question)
answer = create_response(question=question, useful_chunks=useful_chunks)

print(answer)