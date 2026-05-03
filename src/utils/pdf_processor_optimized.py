"""
Optimized PDF Processor for Large Files
Streams PDF chunks without loading entire file into memory
Implements batching for efficient API usage
"""

import logging
from pathlib import Path
from typing import List, Generator, Dict, Optional
from dataclasses import dataclass
import json

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

logger = logging.getLogger(__name__)


@dataclass
class ProcessedChunk:
    """Chunk with metadata for efficient processing"""
    content: str
    source: str
    page: int
    chunk_id: int
    file_size_mb: float
    
    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'source': self.source,
            'page': self.page,
            'chunk_id': self.chunk_id
        }


class OptimizedPDFProcessor:
    """
    Processes large PDFs efficiently with streaming and batching
    Designed to minimize memory footprint and API costs
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        batch_size: int = 10,
        max_pages: Optional[int] = None
    ):
        """
        Initialize PDF processor
        
        Args:
            chunk_size: Max tokens per chunk
            chunk_overlap: Overlap between chunks
            batch_size: Number of chunks to batch before processing
            max_pages: Limit processing to first N pages (for testing)
        """
        if PyPDF2 is None:
            raise ImportError("PyPDF2 required. Install: pip install PyPDF2")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.max_pages = max_pages
        self.processed_chunks: List[ProcessedChunk] = []
        
        logger.info(f"OptimizedPDFProcessor initialized: "
                   f"chunk_size={chunk_size}, batch_size={batch_size}")
    
    def process_pdf_streaming(
        self,
        pdf_path: str
    ) -> Generator[List[ProcessedChunk], None, None]:
        """
        Stream PDF processing in batches
        
        Yields batches of chunks without loading entire PDF into memory
        
        Args:
            pdf_path: Path to PDF file
            
        Yields:
            Batches of ProcessedChunk objects
        """
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required")
        
        pdf_path = Path(pdf_path)
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"Starting streaming processing: {pdf_path.name} ({file_size_mb:.1f} MB)")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                pages_to_process = min(total_pages, self.max_pages) if self.max_pages else total_pages
                
                logger.info(f"Total pages: {total_pages}, Processing: {pages_to_process}")
                
                batch: List[ProcessedChunk] = []
                global_chunk_id = 0
                
                for page_num in range(pages_to_process):
                    try:
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        
                        if not text or len(text.strip()) < 20:
                            logger.warning(f"Page {page_num + 1}: Empty or unreadable")
                            continue
                        
                        # Clean and chunk text
                        text = self._clean_text(text)
                        chunks = self._chunk_text(text)
                        
                        # Create ProcessedChunk objects
                        for chunk_idx, chunk in enumerate(chunks):
                            processed = ProcessedChunk(
                                content=chunk,
                                source=pdf_path.name,
                                page=page_num + 1,
                                chunk_id=global_chunk_id,
                                file_size_mb=file_size_mb
                            )
                            batch.append(processed)
                            global_chunk_id += 1
                            
                            # Yield batch when full
                            if len(batch) >= self.batch_size:
                                logger.info(f"Yielding batch: {len(batch)} chunks "
                                          f"(page {page_num + 1}/{pages_to_process})")
                                yield batch
                                batch = []
                        
                    except Exception as e:
                        logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                        continue
                
                # Yield remaining chunks
                if batch:
                    logger.info(f"Yielding final batch: {len(batch)} chunks")
                    yield batch
                
                logger.info(f"PDF processing complete. Total chunks: {global_chunk_id}")
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if len(chunk.strip()) > 20:  # Minimum chunk size
                chunks.append(chunk)
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted PDF text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove common PDF artifacts
        text = text.replace('\x00', '')
        return text
    
    def get_processing_stats(self) -> Dict:
        """Get statistics about processed PDF"""
        if not self.processed_chunks:
            return {"status": "No chunks processed yet"}
        
        total_content_length = sum(len(c.content) for c in self.processed_chunks)
        unique_pages = len(set(c.page for c in self.processed_chunks))
        
        return {
            "total_chunks": len(self.processed_chunks),
            "total_pages": unique_pages,
            "avg_chunk_length": total_content_length // len(self.processed_chunks),
            "total_content_chars": total_content_length,
            "batch_size": self.batch_size
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    processor = OptimizedPDFProcessor(
        chunk_size=512,
        chunk_overlap=50,
        batch_size=10,
        max_pages=5  # Test with first 5 pages only
    )
    
    pdf_path = "data/Agriculture-CPG-2020.pdf"
    batch_count = 0
    
    for batch in processor.process_pdf_streaming(pdf_path):
        batch_count += 1
        print(f"\nBatch {batch_count}:")
        print(f"  Chunks in batch: {len(batch)}")
        print(f"  First chunk (first 100 chars): {batch[0].content[:100]}...")
        print(f"  Last chunk page: {batch[-1].page}")
        
        # HERE: Send batch to TripletExtractor or other processing
        # This prevents entire PDF from being in memory
