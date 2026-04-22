import logging
from langchain_core.documents import Document
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
import pytesseract
import pdfplumber
from pdf2image import convert_from_path
from langchain_core.documents import Document
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
            template="""Rewrite the following user question into a clear, concise search query optimized for document retrieval. Return only the rewritten query, nothing else.

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



    def _is_scanned(self, file_path: str) -> bool:
        """Returns True if the PDF appears to be a scanned document."""
        total_chars = 0
        total_pages = 0

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text() or ""
                total_chars += len(text.strip())

        avg_chars = total_chars / total_pages if total_pages > 0 else 0
        logger.info("Avg chars/page: %.1f (pages: %d)", avg_chars, total_pages)
        return avg_chars < 100  # threshold — tweak if needed


    def _ocr_pdf(self, file_path: str) -> list[Document]:
        """Convert each page to image, run Tesseract, return Document list."""
        images = convert_from_path(file_path, dpi=300)
        docs = []

        for i, image in enumerate(images):
            raw_text = pytesseract.image_to_string(image, lang="eng", config="--psm 6")
            raw_text = raw_text.strip()

            if not raw_text:
                logger.warning("Page %d returned empty OCR text, skipping", i + 1)
                continue

            # Pass raw OCR text to LLM to fix errors
            cleaned_text = self._clean_ocr_text(raw_text, page_num=i + 1)
            logger.info("cleaned text is " + cleaned_text + "\n")
            docs.append(Document(
                page_content=cleaned_text,
                metadata={
                    "source": file_path,
                    "page": i + 1,
                    "source_type": "scanned_ocr",
                    "ocr_engine": "pytesseract",
                }
            ))
            logger.info("OCR'd page %d — raw: %d chars, cleaned: %d chars",
                        i + 1, len(raw_text), len(cleaned_text))

        return docs


    def _clean_ocr_text(self, raw_text: str, page_num: int) -> str:
        """
        Pass raw OCR output to the LLM to fix common OCR errors:
        broken words, garbled characters, spacing issues, etc.
        """
        clean_prompt = f"""You are a document cleanup assistant.
    The following text was extracted from a scanned PDF using OCR and may contain errors such as:
    - Broken or merged words (e.g. "admis sion" or "admissionfees")
    - Garbled characters (e.g. "0" instead of "O", "1" instead of "l")
    - Missing spaces or extra line breaks
    - Repeated headers/footers from the scanner

    Fix ONLY the OCR errors. Do NOT add new information, summarize, or change the meaning.
    Return only the corrected text, nothing else.

    Raw OCR text (page {page_num}):
    {raw_text}

    Corrected text:"""

        try:
            result = self.llm.invoke(clean_prompt)
            cleaned = result.content.strip()
            logger.info("LLM cleaned OCR text for page %d", page_num)
            return cleaned
        except Exception as e:
            logger.warning("LLM OCR cleanup failed for page %d, using raw text: %s", page_num, e)
            return raw_text  # fallback to raw if LLM fails


    def process_pdf(self, file_path: str) -> str:
        try:
            is_scanned = self._is_scanned(file_path)
            logger.info("Document type: %s", "scanned" if is_scanned else "typed")

            if is_scanned:
                docs = self._ocr_pdf(file_path)
                if not docs:
                    return "Could not extract any text from the scanned PDF."
            else:
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                docs = pages
                for doc in docs:
                    doc.page_content = doc.page_content.replace("\x00", "").replace("\u0000", "")
                    doc.metadata["source_type"] = "typed_pdf"

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            split_docs = text_splitter.split_documents(docs)
            self.vector_store.add_documents(split_docs)

            logger.info("Processed %s PDF '%s' into %d chunks",
                        "scanned" if is_scanned else "typed", file_path, len(split_docs))
            return f"PDF successfully learned! ({'scanned' if is_scanned else 'typed'}, {len(split_docs)} chunks)"

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
