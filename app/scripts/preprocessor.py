"""Data preprocessing utilities for Harry Potter text into chunks, summaries, and quotes."""

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

import pymupdf
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Config


class DataType(Enum):
    """Enum for different types of processed data."""
    CHUNKS = "chunks"
    QUOTES = "quotes"


@dataclass
class ProcessedData:
    """Data structure for processed content."""
    data_type: DataType
    documents: List[Document]
    metadata: Dict[str, Any]


class PreProcessor:
    """Class for preprocessing Harry Potter text into different data formats."""
    
    def __init__(self):
        self.pdf_path = Config.DATA_PATH
        self.full_text = None
        self.chapters = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
    
    def _load_and_clean_text(self) -> str:
        """Load PDF and clean the text while preserving double quotes."""
        doc = pymupdf.open(self.pdf_path)
        full_text = " ".join([page.get_text() for page in doc])
        
        # Remove book title occurrences
        full_text = re.sub(r'HP\s*1\s*-\s*Harry\s*Potter\s*and\s*the\s*Sorcerer\'s\s*Stone', '', full_text)
        
        # Remove single quotes but preserve double quotes
        full_text = full_text.replace("’", "")
        full_text = full_text.replace("“", "\"")
        full_text = full_text.replace("”", "\"")
        
        # Remove unnecessary whitespaces and newlines
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        # Remove non-English characters but preserve double quotes and common punctuation
        full_text = re.sub(r'[^\x00-\x7F".,!?;:()-]+', '', full_text)
        
        return full_text
    

    
    def _extract_chapters(self) -> List[Dict[str, Any]]:
        """Extract chapters from the text."""
        if self.full_text is None:
            self.full_text = self._load_and_clean_text()
        
        # Split into chapters
        chapter_texts = re.split(r'CHAPTER\s[A-Z]+', self.full_text)
        
        chapter_texts.pop(0)
        
        chapters = []
        for i, chapter_text in enumerate(chapter_texts):
            cleaned_text = chapter_text.strip()
            if cleaned_text:  # Only include non-empty chapters
                chapters.append({
                    'number': i + 1,
                    'title': f"Chapter {i + 1}",
                    'content': cleaned_text,
                    'word_count': len(cleaned_text.split()),
                    'char_count': len(cleaned_text)
                })
        
        return chapters
    
    def get_chunks(self) -> ProcessedData:
        """Get text chunks using RecursiveCharacterTextSplitter."""
        if self.chapters is None:
            self.chapters = self._extract_chapters()
        
        documents = []
        
        for chapter in self.chapters:
            # Create a document for each chapter
            chapter_doc = Document(
                page_content=chapter['content'],
                metadata={
                    "chapter_number": chapter['number'],
                    "chapter_title": chapter['title'],
                    "source": "harry_potter_book_1",
                    "data_type": "chapter"
                }
            )
            
            # Split chapter into chunks
            chunks = self.text_splitter.split_documents([chapter_doc])
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "chunk_id": f"ch{chapter['number']}_chunk{i}",
                    "data_type": "chunk"
                })
                documents.append(chunk)
        
        return ProcessedData(
            data_type=DataType.CHUNKS,
            documents=documents,
            metadata={
                "total_chunks": len(documents),
                "total_chapters": len(self.chapters),
                "chunk_size": self.text_splitter._chunk_size,
                "chunk_overlap": self.text_splitter._chunk_overlap
            }
        )
    
    def get_quotes(self) -> ProcessedData:
        """Extract quotes from the text using improved pattern matching."""
        if self.chapters is None:
            self.chapters = self._extract_chapters()
        
        documents = []
        
        
        # Extract dialogue using the improved pattern
        for chapter in self.chapters:
            chapter_text = chapter['content']
            
            # Find all quotes matching the minimum length requirement
            quotes = re.findall('"([^"]*)"', chapter_text)
            # filter out quotes that are less than min_length
            quotes = [quote for quote in quotes if len(quote) >= Config.MIN_QUOTE_LENGTH]
        
            
            for i, quote_text in enumerate(quotes):
                # Clean up the quote text
                cleaned_quote = quote_text.strip()

                
                if cleaned_quote:  # Only include non-empty quotes
                    quote_doc = Document(
                        page_content=cleaned_quote,
                        metadata={
                            "chapter_number": chapter['number'],
                            "chapter_title": chapter['title'],
                            "quote_index": i,
                            "quote_length": len(cleaned_quote),
                        }
                    )
                    documents.append(quote_doc)
        
        return ProcessedData(
            data_type=DataType.QUOTES,
            documents=documents,
            metadata={
                "total_quotes": len(documents),
                "min_length_filter": Config.MIN_QUOTE_LENGTH,
                "extraction_method": "regex_pattern"
            }
        )
    
    
    
    def get_all_processed_data(self) -> Dict[DataType, ProcessedData]:
        """Get all three types of processed data."""
        return {
            DataType.CHUNKS: self.get_chunks(),
            DataType.QUOTES: self.get_quotes()
        }
    
    def save_processed_data(self):
        """Save processed data to files for inspection."""
        os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
        
        all_data = self.get_all_processed_data()
        
        for data_type, processed_data in all_data.items():
            type_dir = os.path.join(Config.OUTPUT_PATH, data_type.value)
            os.makedirs(type_dir, exist_ok=True)
            
            # Save metadata
            with open(os.path.join(type_dir, "metadata.txt"), 'w') as f:
                f.write(f"Data Type: {data_type.value}\n")
                f.write(f"Total Documents: {len(processed_data.documents)}\n")
                for key, value in processed_data.metadata.items():
                    f.write(f"{key}: {value}\n")
            
            # Save sample documents
            for i, doc in enumerate(processed_data.documents[:5]):  # Save first 5 as samples
                filename = os.path.join(type_dir, f"sample_{i+1}.txt")
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"=== Document {i+1} ===\n")
                    f.write(f"Metadata: {doc.metadata}\n\n")
                    f.write(f"Content:\n{doc.page_content}\n")
        
        print(f"Processed data saved to {Config.OUTPUT_PATH}")


if __name__ == "__main__":
    # Example usage
    preprocessor = PreProcessor()
    
    # Save all processed data for inspection
    preprocessor.save_processed_data()
    
    # Get specific data types
    chunks_data = preprocessor.get_chunks()
    quotes_data = preprocessor.get_quotes()
    
    print(f"Chunks: {len(chunks_data.documents)} documents")
    print(f"Quotes: {len(quotes_data.documents)} documents")