"""
Document processing module for RAG system
Handles PDF loading, text extraction, and chunking
"""

import fitz  # PyMuPDF
import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit as st


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            metadata = {
                "filename": os.path.basename(pdf_path),
                "page_count": len(doc),
                "file_path": pdf_path
            }
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
            
            doc.close()
            
            return {
                "text": text.strip(),
                "metadata": metadata
            }
            
        except Exception as e:
            st.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return None
    
    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Split document text into chunks
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            List of Document objects
        """
        chunks = self.text_splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            chunk_metadata["chunk_size"] = len(chunk)
            
            documents.append(Document(
                page_content=chunk,
                metadata=chunk_metadata
            ))
        
        return documents
    
    def process_pdf_file(self, pdf_path: str) -> List[Document]:
        """
        Process a single PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Document chunks
        """
        # Extract text
        extracted_data = self.extract_text_from_pdf(pdf_path)
        if not extracted_data:
            return []
        
        # Chunk the document
        documents = self.chunk_document(
            extracted_data["text"], 
            extracted_data["metadata"]
        )
        
        return documents
    
    def process_pdf_directory(self, directory_path: str) -> List[Document]:
        """
        Process all PDF files in a directory
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of all Document chunks
        """
        if not os.path.exists(directory_path):
            st.error(f"Directory {directory_path} does not exist")
            return []
        
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            st.warning(f"No PDF files found in {directory_path}")
            return []
        
        all_documents = []
        
        # Process each PDF with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, pdf_file in enumerate(pdf_files):
            status_text.text(f"Processing {pdf_file}...")
            pdf_path = os.path.join(directory_path, pdf_file)
            
            documents = self.process_pdf_file(pdf_path)
            all_documents.extend(documents)
            
            # Update progress
            progress_bar.progress((i + 1) / len(pdf_files))
        
        status_text.text(f"Processed {len(pdf_files)} PDF files, created {len(all_documents)} chunks")
        
        return all_documents
    
    def process_uploaded_file(self, uploaded_file) -> List[Document]:
        """
        Process an uploaded PDF file from Streamlit
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            List of Document chunks
        """
        try:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the file
            documents = self.process_pdf_file(temp_path)
            
            # Clean up temporary file
            os.remove(temp_path)
            
            return documents
            
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            return []
    
    def get_document_summary(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get summary statistics for processed documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Summary statistics
        """
        if not documents:
            return {"total_chunks": 0, "total_characters": 0, "unique_files": 0}
        
        total_chunks = len(documents)
        total_characters = sum(len(doc.page_content) for doc in documents)
        unique_files = len(set(doc.metadata.get("filename", "") for doc in documents))
        
        return {
            "total_chunks": total_chunks,
            "total_characters": total_characters,
            "unique_files": unique_files,
            "average_chunk_size": total_characters / total_chunks if total_chunks > 0 else 0
        }