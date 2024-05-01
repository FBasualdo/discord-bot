
import faiss
import pinecone
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from template import get_template
from langchain.memory import VectorStoreRetrieverMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone
from langchain.docstore import InMemoryDocstore
from os import getenv
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
     
import pinecone
from os import getenv
from docx import Document
import cohere
import os
from langchain.document_loaders import PyPDFLoader

load_dotenv()



class First_Bot():
    def __init__(self) -> None:
        self.template = """You are an AI whose ONLY role is to take User input and extract the key words from it.
    This is the user input: 
    {input}
    REMEMBER: JUST EXTRACT THE KEY WORDS 
    ANSWER IN SPANISH, RESPONDE EN ESPAÑOL"""
        self.set_prompt()
        self.llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=.2)
        self.create_chain()

    def set_prompt(self):
        
        self.prompt = PromptTemplate(
            input_variables=["input"],
            template= self.template
        )

    def create_chain(self):
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, verbose= True)
    
    def run_chain(self, message):
        response = self.chain.run({#    'chat_history': message_history.messages,
                                   'input': message,})
        return response

class Second_Bot():
    def __init__(self,) -> None:
        self.connect_to_pinecone()
        self.template = get_template()
        self.embeddings = OpenAIEmbeddings()
        self.set_database()
        self.set_memory()
        self.connect_to_cohere()
        self.set_prompt()
        self.llm = ChatOpenAI(model='gpt-4-1106-preview', temperature=.2)
        self.create_chain()


    def get_text(self):
        document=[]
        for file in os.listdir("docs"):
            if file.endswith(".pdf"):
                pdf_path="./docs/"+file
                loader=PyPDFLoader(pdf_path)
                document.extend(loader.load())
            elif file.endswith('.docx') or file.endswith('.doc'):
                doc_path="./docs/"+file
                loader=Docx2txtLoader(doc_path)
                document.extend(loader.load())
            elif file.endswith('.txt'):
                text_path="./docs/"+file
                loader=TextLoader(text_path)
                document.extend(loader.load())
    
        document_splitter=CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        document_chunks = document_splitter.split_documents(document)
        print(document_chunks)
        return document_chunks
    
            
    def formulate_answer(self, database):
        answer_to_chain = ""
        counter = 0
        for answer in database:
            answer_modified = answer.document['text'].replace("\n", " ")
            answer_to_chain += f"""
    MATCH NUMBER {counter}:
    {answer_modified}
    -------------------------------------------------
    """
            counter += 1
        print(answer_to_chain)

        return answer_to_chain

    def connect_to_pinecone(self):
        pinecone.init(
            api_key= getenv("PINECONE_API_KEY"),
            environment= getenv("PINECONE_ENV")
        )
        self.index_name = getenv("PINECONE_INDEX")
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
            name=self.index_name,
            metric='cosine',
            dimension=1536  
        )           
            
    def add_indexes_to_pinecone(self, text):
        self.docsearch = Pinecone.from_texts(
            texts= text,
            embedding= self.embeddings,
            index_name= self.index_name
        )
        
        
    def pinecone_from_existing(self):
        self.docsearch = Pinecone.from_existing_index(
            embedding= self.embeddings,
            index_name= self.index_name
        )

    def run_pinecone(self):  
        self.docsearch = Pinecone.from_existing_index(
            embedding= self.embeddings,
            index_name= self.index_name
        )    
    
    
    def connect_to_cohere(self):
        self.co = cohere.Client(api_key=getenv("COHERE_API_KEY"))
        print("Cohere connected")

    def cohere_rerank(self, query, docs):
        results = self.co.rerank(query= query, documents= docs, top_n= 3, model= 'rerank-multilingual-v2.0')
        # print("This is the result: ", results)
        return results
    
    def set_database(self):
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        embedding_fn = OpenAIEmbeddings().embed_query
        self.vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
    
    
    def set_memory(self):
        retriever = self.vectorstore.as_retriever(search_kwargs=dict(k=3))
        self.memory = VectorStoreRetrieverMemory(retriever= retriever, memory_key="history_messages")
        #explain
        self.memory.save_context({"input": "Not input yet"}, {"output": ""})
        self.memory.save_context({"input": "Not input yet"}, {"output": ""})
    
        
    def set_prompt(self):
        self.prompt = PromptTemplate(
            input_variables=['kb_data','conversation', 'user_input'],
            template= self.template
        )
        
    
    def create_chain(self):
        self.chain = LLMChain(
            prompt= self.prompt,
            verbose= True,
            llm= ChatOpenAI(model='gpt-3.5-turbo', temperature=.6)
        )
        
    def run_chain(self,doc_data, user_input, conversation):
        response = self.chain.predict(
            kb_data= doc_data,
            user_input= user_input,
            conversation= conversation
        )
        return response