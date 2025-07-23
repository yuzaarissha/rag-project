
import fitz  # PyMuPDF
import os
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit as st


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\nСтатья", "\n\n", ".\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            metadata = {
                "filename": os.path.basename(pdf_path),
                "page_count": len(doc),
                "file_path": pdf_path
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text().strip()
                if page_text:
                    text += f"\n\n{page_text}"
            
            doc.close()
            
            return {
                "text": text.strip(),
                "metadata": metadata
            }
            
        except Exception as e:
            st.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return None
    
    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        chunks = self.text_splitter.split_text(text)
        
        documents = []
        valid_chunk_id = 0
        
        for chunk in chunks:
            chunk_cleaned = chunk.strip()
            if len(chunk_cleaned) >= 50:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = valid_chunk_id
                chunk_metadata["chunk_size"] = len(chunk_cleaned)
                
                documents.append(Document(
                    page_content=chunk_cleaned,
                    metadata=chunk_metadata
                ))
                valid_chunk_id += 1
        
        return documents
    
    def process_pdf_file(self, pdf_path: str) -> List[Document]:
        extracted_data = self.extract_text_from_pdf(pdf_path)
        if not extracted_data:
            return []
        
        documents = self.chunk_document(
            extracted_data["text"], 
            extracted_data["metadata"]
        )
        
        return documents
    
    def process_pdf_directory(self, directory_path: str) -> List[Document]:
        if not os.path.exists(directory_path):
            st.error(f"Directory {directory_path} does not exist")
            return []
        
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            st.warning(f"No PDF files found in {directory_path}")
            return []
        
        all_documents = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, pdf_file in enumerate(pdf_files):
            status_text.text(f"Processing {pdf_file}...")
            pdf_path = os.path.join(directory_path, pdf_file)
            
            documents = self.process_pdf_file(pdf_path)
            all_documents.extend(documents)
            
            progress_bar.progress((i + 1) / len(pdf_files))
        
        status_text.text(f"Processed {len(pdf_files)} PDF files, created {len(all_documents)} chunks")
        
        return all_documents
    
    def _generate_readable_filename(self, pdf_content: bytes, original_name: str) -> str:
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            text = ""
            
            for page_num in range(min(2, len(doc))):
                page = doc[page_num]
                page_text = page.get_text()
                text += page_text
                if len(text) > 500:
                    break
            
            doc.close()
            
            if not text.strip():
                base_name = self._sanitize_filename(original_name)
                return base_name if base_name.endswith('.pdf') else base_name + '.pdf'
            
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            meaningful_lines = []
            
            for line in lines:
                if len(line) < 5 or line.isdigit():
                    continue
                if len(re.sub(r'[^\w\s]', '', line)) < len(line) * 0.6:
                    continue
                if line.lower() in ['page', 'chapter', 'section', 'contents', 'table of contents']:
                    continue
                meaningful_lines.append(line)
                if len(meaningful_lines) >= 2:
                    break
            
            if meaningful_lines:
                title_text = ' '.join(meaningful_lines)[:40]
                clean_title = self._sanitize_filename(title_text)
                if len(clean_title) > 8:
                    return clean_title + '.pdf'
            
            base_name = self._sanitize_filename(original_name)
            return base_name if base_name.endswith('.pdf') else base_name + '.pdf'
            
        except Exception as e:
            st.warning(f"Could not extract title from PDF: {str(e)}")
            base_name = self._sanitize_filename(original_name)
            return base_name if base_name.endswith('.pdf') else base_name + '.pdf'
    
    def _sanitize_filename(self, filename: str) -> str:
        name_without_ext = os.path.splitext(filename)[0]
        
        clean_name = re.sub(r'[^\w\s-]', '', name_without_ext)
        clean_name = re.sub(r'\s+', '_', clean_name)
        clean_name = re.sub(r'-+', '_', clean_name)
        clean_name = re.sub(r'_+', '_', clean_name)
        
        clean_name = clean_name.strip('_')
        
        if len(clean_name) > 40:
            clean_name = clean_name[:40]
        
        if len(clean_name) < 3:
            clean_name = f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return clean_name
    
    def _get_unique_filepath(self, directory: str, filename: str) -> str:
        base_path = os.path.join(directory, filename)
        
        if not os.path.exists(base_path):
            return base_path
        
        name_without_ext = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]
        
        counter = 1
        while True:
            new_filename = f"{name_without_ext}_({counter}){extension}"
            new_path = os.path.join(directory, new_filename)
            if not os.path.exists(new_path):
                return new_path
            counter += 1
    
    def process_uploaded_file(self, uploaded_file) -> List[Document]:
        try:
            docs_dir = os.path.abspath("./data/documents")
            os.makedirs(docs_dir, exist_ok=True)
            
            file_content = uploaded_file.getbuffer()
            
            readable_name = self._generate_readable_filename(file_content, uploaded_file.name)
            
            final_path = self._get_unique_filepath(docs_dir, readable_name)
            final_filename = os.path.basename(final_path)
            
            with open(final_path, "wb") as f:
                f.write(file_content)
            
            st.info(f"Файл сохранен: {final_filename}")
            
            extracted_data = self.extract_text_from_pdf(final_path)
            if not extracted_data:
                st.error("Не удалось извлечь текст из PDF")
                return []
            
            extracted_data["metadata"]["filename"] = final_filename
            extracted_data["metadata"]["file_path"] = final_path
            extracted_data["metadata"]["original_name"] = uploaded_file.name
            
            documents = self.chunk_document(
                extracted_data["text"], 
                extracted_data["metadata"]
            )
            
            if documents:
                st.success(f"Создано {len(documents)} фрагментов из {extracted_data['metadata']['page_count']} страниц")
            
            return documents
            
        except Exception as e:
            st.error(f"Ошибка обработки файла: {str(e)}")
            return []
    
    def get_document_summary(self, documents: List[Document]) -> Dict[str, Any]:
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
    
    def delete_physical_file(self, filename: str) -> bool:
        try:
            file_path = os.path.join("./data/documents", filename)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                st.success(f"Physical file deleted: {filename}")
                return True
            else:
                st.warning(f"Physical file not found: {filename}")
                return False
                
        except Exception as e:
            st.error(f"Error deleting physical file {filename}: {str(e)}")
            return False
    
    def rename_physical_file(self, old_filename: str, new_filename: str) -> bool:
        try:
            docs_dir = "./data/documents"
            old_path = os.path.join(docs_dir, old_filename)
            new_path = os.path.join(docs_dir, new_filename)
            
            if not os.path.exists(old_path):
                st.warning(f"Original file not found: {old_filename}")
                return False
            
            if os.path.exists(new_path):
                st.error(f"File with name {new_filename} already exists")
                return False
            
            os.rename(old_path, new_path)
            st.success(f"File renamed: {old_filename} → {new_filename}")
            return True
            
        except Exception as e:
            st.error(f"Error renaming file: {str(e)}")
            return False
    
    def get_physical_file_info(self, filename: str) -> Dict[str, Any]:
        try:
            file_path = os.path.join("./data/documents", filename)
            
            if not os.path.exists(file_path):
                return {"exists": False, "path": file_path}
            
            stat_info = os.stat(file_path)
            
            return {
                "exists": True,
                "path": file_path,
                "size_bytes": stat_info.st_size,
                "size_mb": round(stat_info.st_size / (1024*1024), 2),
                "modified_time": datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            st.error(f"Error getting file info for {filename}: {str(e)}")
            return {"exists": False, "error": str(e)}