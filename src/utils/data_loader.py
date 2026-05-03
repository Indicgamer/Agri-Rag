"""
Data Loader Module for Agri-RAG
Handles ingestion of agricultural texts (PDF, TXT) with chunking strategies.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

from configs.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document chunk"""
    content: str
    source: str
    page: Optional[int] = None
    chunk_id: Optional[int] = None
    metadata: Optional[Dict] = None


class DataLoader:
    """
    Loads and preprocesses agricultural data sources.
    Supports PDF and TXT formats with configurable chunking.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_length: int = 100,
        chunk_strategy: str = "sentence"
    ):
        """
        Initialize DataLoader
        
        Args:
            chunk_size: Max tokens per chunk
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_length: Minimum chunk length to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.chunk_strategy = chunk_strategy
        self.documents: List[Document] = []

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load and extract text from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Document objects with text chunks
        """
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required to load PDFs. Install: pip install PyPDF2")

        logger.info(f"Loading PDF: {pdf_path}")
        documents = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    # Clean text
                    text = self._clean_text(text)
                    
                    if text:
                        # Chunk the text
                        chunks = self._chunk_text(text)
                        
                        for chunk_id, chunk in enumerate(chunks):
                            doc = Document(
                                content=chunk,
                                source=Path(pdf_path).name,
                                page=page_num + 1,  # 1-indexed
                                chunk_id=chunk_id,
                                metadata={
                                    "source_type": "PDF",
                                    "total_pages": num_pages,
                                    "page_num": page_num + 1
                                }
                            )
                            documents.append(doc)
                            
            logger.info(f"Extracted {len(documents)} chunks from {num_pages} pages")
            self.documents.extend(documents)
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {str(e)}")
            raise

    def load_txt(self, txt_path: str) -> List[Document]:
        """
        Load and extract text from TXT file
        
        Args:
            txt_path: Path to TXT file
            
        Returns:
            List of Document objects with text chunks
        """
        logger.info(f"Loading TXT: {txt_path}")
        documents = []
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Clean text
            text = self._clean_text(text)
            
            # Chunk the text
            chunks = self._chunk_text(text)
            
            for chunk_id, chunk in enumerate(chunks):
                doc = Document(
                    content=chunk,
                    source=Path(txt_path).name,
                    chunk_id=chunk_id,
                    metadata={
                        "source_type": "TXT",
                    }
                )
                documents.append(doc)
                
            logger.info(f"Extracted {len(documents)} chunks from TXT file")
            self.documents.extend(documents)
            return documents
            
        except Exception as e:
            logger.error(f"Error loading TXT {txt_path}: {str(e)}")
            raise

    def load_directory(self, directory: str, file_types: List[str] = None) -> List[Document]:
        """
        Load all documents from a directory
        
        Args:
            directory: Directory path
            file_types: List of file extensions to load (e.g., ['.pdf', '.txt'])
            
        Returns:
            List of all loaded documents
        """
        if file_types is None:
            file_types = ['.pdf', '.txt']
            
        logger.info(f"Loading documents from directory: {directory}")
        all_documents = []
        
        dir_path = Path(directory)
        for file_type in file_types:
            for file_path in dir_path.glob(f"*{file_type}"):
                logger.info(f"Processing {file_path.name}")
                
                if file_type.lower() == '.pdf':
                    docs = self.load_pdf(str(file_path))
                elif file_type.lower() == '.txt':
                    docs = self.load_txt(str(file_path))
                else:
                    logger.warning(f"Unsupported file type: {file_type}")
                    continue
                    
                all_documents.extend(docs)
        
        logger.info(f"Loaded {len(all_documents)} total documents from directory")
        return all_documents

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into overlapping segments
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of text chunks
        """
        if self.chunk_strategy == "sentence":
            return self._chunk_by_sentence_windows(text)

        if self.chunk_strategy == "paragraph":
            return self._chunk_by_paragraph(text)

        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Save current chunk if it meets minimum length
                if current_length >= self.min_chunk_length:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_size = int(self.chunk_overlap / len(current_chunk)) if current_chunk else 0
                overlap_sentences = current_chunk[-overlap_size:] if overlap_size > 0 else []
                
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
        
        # Add final chunk. Keep short files as a single chunk so small demo notes
        # and concise advisories are still ingestible.
        if current_chunk and (current_length >= self.min_chunk_length or not chunks):
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks

    def _chunk_by_sentence_windows(self, text: str) -> List[str]:
        """
        Chunk text into small sentence windows.

        This is better for KG extraction because each chunk usually contains one
        or two relations instead of an entire mixed paragraph.
        """
        sentences = [
            sentence.strip()
            for sentence in re.split(r'(?<=[.!?])\s+', text)
            if sentence.strip()
        ]
        if not sentences:
            return [text] if text.strip() else []

        chunks = []
        current_chunk = []
        current_length = 0
        target_size = max(20, min(self.chunk_size, 1024))

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_chunk and current_length + sentence_length > target_size:
                chunks.append(" ".join(current_chunk))
                overlap_size = int(self.chunk_overlap / 2) if self.chunk_overlap > 0 else 0
                overlap_sentences = current_chunk[-overlap_size:] if overlap_size > 0 else []
                current_chunk = overlap_sentences
                current_length = sum(len(s.split()) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _chunk_by_paragraph(self, text: str) -> List[str]:
        """
        Chunk text by paragraph blocks.

        This mode is more efficient for large PDFs because it keeps more coherent
        text together and reduces the total number of LLM calls.
        """
        paragraphs = [
            p.strip() for p in re.split(r'\n{2,}', text) if p.strip()
        ]
        if not paragraphs:
            return [text] if text.strip() else []

        chunks = []
        current_chunk = []
        current_length = 0

        for paragraph in paragraphs:
            paragraph_length = len(paragraph.split())

            if current_length + paragraph_length <= self.chunk_size or not current_chunk:
                current_chunk.append(paragraph)
                current_length += paragraph_length
                continue

            if current_length >= self.min_chunk_length:
                chunks.append("\n\n".join(current_chunk))

            overlap_paragraphs = current_chunk[-1:] if self.chunk_overlap > 0 else []
            current_chunk = overlap_paragraphs + [paragraph]
            current_length = sum(len(p.split()) for p in current_chunk)

        if current_chunk and (current_length >= self.min_chunk_length or not chunks):
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Preserve paragraph separators while normalizing internal whitespace
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove special characters but keep alphanumeric, punctuation, and newlines
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\n]', '', text)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove page numbers and headers/footers
        text = re.sub(r'Page \d+|^\d+$', '', text, flags=re.MULTILINE)

        # Strip leading/trailing whitespace from each paragraph
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        text = '\n\n'.join(paragraphs)
        return text

    def create_agricultural_triplets(self, documents: List[Document]) -> List[Dict]:
        """
        Create structured triplets from documents for KG construction
        This is a utility method - actual extraction will be done by Llama-3
        
        Args:
            documents: List of documents
            
        Returns:
            List of triplet dictionaries
        """
        triplets = []
        
        for doc in documents:
            # This is a placeholder
            # Actual triplet extraction will be done by the TripletExtractor model
            triplet = {
                "document_id": doc.chunk_id,
                "source": doc.source,
                "content": doc.content,
                "page": doc.page,
                "metadata": doc.metadata,
                "extracted_triplets": []  # Will be populated by Llama-3
            }
            triplets.append(triplet)
        
        return triplets

    def get_documents_by_source(self, source: str) -> List[Document]:
        """
        Get all documents from a specific source
        
        Args:
            source: Source filename
            
        Returns:
            List of documents from that source
        """
        return [doc for doc in self.documents if doc.source == source]

    def get_documents_by_page(self, source: str, page: int) -> List[Document]:
        """
        Get all documents from a specific page
        
        Args:
            source: Source filename
            page: Page number
            
        Returns:
            List of documents from that page
        """
        return [
            doc for doc in self.documents 
            if doc.source == source and doc.page == page
        ]

    def save_documents(self, output_path: str):
        """
        Save documents to a file for inspection
        
        Args:
            output_path: Path to save documents
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in self.documents:
                f.write(f"Source: {doc.source}\n")
                f.write(f"Page: {doc.page}\n")
                f.write(f"Chunk: {doc.chunk_id}\n")
                f.write(f"Metadata: {doc.metadata}\n")
                f.write(f"Content:\n{doc.content}\n")
                f.write("\n" + "="*80 + "\n\n")
        
        logger.info(f"Documents saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize loader
    loader = DataLoader(chunk_size=512, chunk_overlap=50)
    
    # Example: Load from directory
    # docs = loader.load_directory("data/")
    
    # Example: Load specific PDF
    # docs = loader.load_pdf("data/TNAU_CPG_2020.pdf")
    
    print("Data Loader initialized successfully")
