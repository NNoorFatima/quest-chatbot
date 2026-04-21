import logging
import requests
from bs4 import BeautifulSoup
import re
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
logger = logging.getLogger(__name__)


class FASTChatbot:
    def __init__(self):
        self.base_url = "https://www.fast.nu.edu.pk/"
        self.llm = ChatGroq(
            temperature=0.7,
            model="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

        supabase_url = os.getenv("SUPABASE_URL")
        if supabase_url is None:
            raise ValueError("Expected SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        if supabase_key is None:
            raise ValueError("Expected SUPABASE_SERVICE_KEY")

        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = SupabaseVectorStore(
            embedding=self.embeddings,
            client=self.supabase,
            table_name="documents",
            query_name="match_documents",
        )

        self.prompt = PromptTemplate(
            input_variables=["user_input", "pdf_context", "web_context"],
            template="""You are a helpful assistant for FAST National University.
Provide accurate and concise answers based on the context provided below.
If no relevant context is available, say "Sorry, I couldn't find any relevant information in the documents." and answer from your general knowledge about FAST University.
Keep responses friendly and concise.

User question: {user_input}

PDF Document Context:
{pdf_context}

Website Context:
{web_context}

Response:""",
        )

        self.rewrite_prompt = PromptTemplate(
            input_variables=["user_input"],
            template="""Rewrite the following user question into a clear, concise search query optimized for document retrieval about FAST National University. Return only the rewritten query, nothing else.

User question: {user_input}
Rewritten query:""",
        )

        self.chain = self.prompt | self.llm
        self.rewrite_chain = self.rewrite_prompt | self.llm

    def _rewrite_query(self, user_input: str) -> str:
        try:
            result = self.rewrite_chain.invoke({"user_input": user_input})
            rewritten = result.content.strip()
            logger.info("Rewrote query: %r -> %r", user_input, rewritten)
            return rewritten
        except Exception as e:
            logger.warning("Query rewrite failed, using original: %s", e)
            return user_input

    def _get_pdf_context(self, query: str) -> str:
        try:
            docs = self.vector_store.similarity_search(query, k=3)
            if docs:
                logger.info("Retrieved %d chunks from vector store", len(docs))
                return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            logger.error("Vector store search failed: %s", e)
        return ""

    def _get_web_context(self, user_input_lower: str) -> str:
        if "admissions" in user_input_lower:
            return self.fetch_webpage_content(self.base_url + "#/")
        elif "courses" in user_input_lower or "programs" in user_input_lower:
            return self.fetch_webpage_content(self.base_url + "Degree-Programs/")
        elif any(k in user_input_lower for k in ["location", "campuses", "where"]):
            return "FAST has campuses in Lahore, Islamabad, Karachi, Peshawar, and Faisalabad."
        elif any(k in user_input_lower for k in ["contact", "email", "phone"]):
            return "Contact: info@fast.nu.edu.pk"
        return ""

    def process_pdf(self, file_path):
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            # Larger overlap + explicit separators so chunks respect paragraph boundaries
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            docs = text_splitter.split_documents(pages)
            for doc in docs:
                doc.page_content = doc.page_content.replace("\x00", "").replace("\u0000", "")

            self.vector_store.add_documents(docs)
            logger.info("Processed PDF %s into %d chunks", file_path, len(docs))
            return "PDF successfully learned! You can now ask questions about it."
        except Exception as e:
            logger.error("Error processing PDF: %s", e)
            return f"Error processing PDF: {str(e)}"

    def fetch_webpage_content(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            text = re.sub(r"\s+", " ", soup.get_text()).strip()
            return text[:1000]
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            return ""

    def stream_response(self, user_input: str):
        """Yield LLM response tokens one chunk at a time."""
        retrieval_query = self._rewrite_query(user_input)
        pdf_context = self._get_pdf_context(retrieval_query)
        web_context = self._get_web_context(user_input.lower().strip())

        try:
            for chunk in self.chain.stream({
                "user_input": user_input,
                "pdf_context": pdf_context or "No PDF context available.",
                "web_context": web_context or "No website context available.",
            }):
                yield chunk.content
        except Exception as e:
            logger.error("LLM streaming failed: %s", e)
            yield "Sorry, my AI brain encountered an error."

    def get_response(self, user_input: str) -> str:
        return "".join(self.stream_response(user_input))
