# FAST University Chatbot
# A basic chatbot that provides information about FAST National University

import requests
from bs4 import BeautifulSoup
import re
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
# from api import app
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

class FASTChatbot:
    def __init__(self):
        self.base_url = "https://www.fast.nu.edu.pk/"  # Corrected to FAST University URL
        load_dotenv()
        self.llm = ChatGroq(temperature=0.7, model="llama-3.1-8b-instant", groq_api_key=os.getenv('GROQ_API_KEY'))
        # self.vector_store = None
        supabase_url = os.getenv('SUPABASE_URL')
        if supabase_url is None:
            raise ValueError("Expected SUPABASE_URL")
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        if supabase_key is None:
            raise ValueError("Expected SUPABASE_SERVICE_ROLE_KEY")
        # Create a Supabase client to interact with the database
        # This client is used to store and retrieve documents from the vector store
        self.supabase: Client = create_client(supabase_url, supabase_key)
        # Initialize a Supabase vector store to store and retrieve documents
        # This vector store is used to store and retrieve documents from the database
        # The embeddings are used to generate vector representations of the documents
        # The vector store is created with the embeddings and the Supabase client
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") #embed model 
        self.vector_store = SupabaseVectorStore(embedding=self.embeddings, 
                                                client=self.supabase,table_name="documents", 
                                                query_name="match_documents")

        
        self.responses = {
            "hi": "Hello! I'm the FAST University chatbot. How can I help you today?",
            "hello": "Hello! I'm the FAST University chatbot. How can I help you today?",
            "admissions": "For admissions information, please visit our admissions page.",
            "courses": "FAST offers programs in Computer Science, Electrical Engineering, and more.",
            "location": "FAST has campuses in Lahore, Islamabad, Karachi, Peshawar, and Faisalabad.",
            "contact": "You can contact us at info@fast.nu.edu.pk or visit our website.",
            "bye": "Goodbye! Have a great day."
        }
        # LangChain prompt for generating responses
        self.prompt = PromptTemplate(
            input_variables=["user_input", "fetched_info"],
            template="""
            You are a helpful assistant for FAST National University.
            Provide accurate and concise answers based on the pdf provided.
            If no context is provided then mention "Sorry, I couldn't find any relevant information" and, answer using your general knowledge about FAST University.
            Keep responses friendly and concise.

            User question: {user_input}
            Context: {fetched_info}

            Response:
            """
        )
        self.chain = self.prompt | self.llm

    def process_pdf(self, file_path):
        try:
            # 1. Load the PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # 2. Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(pages)
            for doc in docs:
                doc.page_content = doc.page_content.replace('\x00', '')
                doc.page_content = doc.page_content.replace('\u0000', '')
            # 3. Create or update the vector store
            self.vector_store.add_documents(docs)
            # if self.vector_store is None:
            #     self.vector_store = FAISS.from_documents(docs, self.embeddings)
            # else:
            #     self.vector_store.add_documents(docs)
                
            return "PDF successfully learned! You can now ask questions about it."
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    def fetch_webpage_content(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract main text content
            text = soup.get_text()
            # Clean up the text
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:1000]  # Limit to first 1000 chars for brevity
        except Exception as e:
            return f"Sorry, I couldn't fetch the information. Error: {str(e)}"

    # def get_response(self, user_input):
    #     user_input_lower = user_input.lower().strip()

    #     fetched_info = ""

    #     # 1. Query the vector store first if a PDF has been uploaded
    #     if self.vector_store is not None:
    #         docs = self.vector_store.similarity_search(user_input, k=3)
    #         print(f"FOUND {len(docs)} RESULTS!") # Debug print
    #         if docs:
    #             fetched_info = "\n\n".join([doc.page_content for doc in docs])

    #     # 2. If no PDF context found, fall back to web scraping for known topics
    #     if not fetched_info:
    #         if "admissions" in user_input_lower:
    #             fetched_info = self.fetch_webpage_content(self.base_url + "#/")
    #         elif "courses" in user_input_lower or "programs" in user_input_lower:
    #             fetched_info = self.fetch_webpage_content(self.base_url + "Degree-Programs/")
    #         elif any(k in user_input_lower for k in ["location", "campuses", "where"]):
    #             fetched_info = "FAST has campuses in Lahore, Islamabad, Karachi, Peshawar, and Faisalabad."
    #         elif any(k in user_input_lower for k in ["contact", "email", "phone"]):
    #             fetched_info = "Contact: info@fast.nu.edu.pk"

    #     # 3. Use LangChain to generate a response
    #     response = self.chain.invoke({"user_input": user_input, "fetched_info": fetched_info}).content
    #     return response
    def get_response(self, user_input):
        print("--- DEBUG: 1. STARTING GET_RESPONSE ---")
        user_input_lower = user_input.lower().strip()
        fetched_info = ""

        # 1. Query the vector store
        try:
            print("--- DEBUG: 2. CHECKING SUPABASE VECTOR STORE ---")
            if self.vector_store is not None:
                docs = self.vector_store.similarity_search(user_input, k=3)
                print(f"--- DEBUG: 3. FOUND {len(docs)} RESULTS FROM PDF! ---")
                if docs:
                    fetched_info = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"--- DEBUG ERROR: SUPABASE FAILED: {str(e)} ---")

        # 2. Fallback to Web info
        if not fetched_info:
            print("--- DEBUG: 4. NO PDF INFO, CHECKING WEB ROUTING ---")
            if "admissions" in user_input_lower:
                fetched_info = self.fetch_webpage_content(self.base_url + "#/")
            elif "courses" in user_input_lower or "programs" in user_input_lower:
                fetched_info = self.fetch_webpage_content(self.base_url + "Degree-Programs/")
            elif any(k in user_input_lower for k in ["location", "campuses", "where"]):
                fetched_info = "FAST has campuses in Lahore, Islamabad, Karachi, Peshawar, and Faisalabad."
            elif any(k in user_input_lower for k in ["contact", "email", "phone"]):
                fetched_info = "Contact: info@fast.nu.edu.pk"

        # 3. Langchain LLM Call
        try:
            print("--- DEBUG: 5. SENDING TO GROQ LLM ---")
            response = self.chain.invoke({"user_input": user_input, "fetched_info": fetched_info}).content
            print("--- DEBUG: 6. SUCCESS! RETURNING RESPONSE ---")
            return response
        except Exception as e:
            print(f"--- DEBUG ERROR: GROQ LLM FAILED: {str(e)} ---")
            return "Sorry, my AI brain encountered an error."
def main():
    chatbot = FASTChatbot()
    print("FAST University Chatbot: Hello! Ask me about FAST University.")
    print("Type 'bye' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "bye":
            print("Chatbot:", chatbot.get_response("bye"))
            break
        response = chatbot.get_response(user_input)
        print("Chatbot:", response)

# if __name__ == "__main__":
#     main()
