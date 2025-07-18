"""
Vector store management using ChromaDB
Handles embeddings and similarity search
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import ollama
import streamlit as st
from langchain.schema import Document
import os
import hashlib


class VectorStore:
    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = "./data/chroma_db"):
        """
        Initialize ChromaDB vector store
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Ollama nomic-embed-text
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        try:
            embeddings = []
            
            # Show progress for large batches
            if len(texts) > 10:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            for i, text in enumerate(texts):
                try:
                    response = ollama.embeddings(
                        model="nomic-embed-text:latest",
                        prompt=text
                    )
                    
                    if 'embedding' in response and response['embedding']:
                        embeddings.append(response['embedding'])
                    else:
                        st.error(f"Не удалось получить эмбеддинг для текста {i+1}")
                        return []
                    
                except Exception as e:
                    st.error(f"Ошибка генерации эмбеддинга для текста {i+1}: {e}")
                    return []
                
                # Update progress
                if len(texts) > 10:
                    progress_bar.progress((i + 1) / len(texts))
                    status_text.text(f"Генерация эмбеддингов: {i + 1}/{len(texts)}")
            
            if len(texts) > 10:
                status_text.text(f"Сгенерировано {len(embeddings)} эмбеддингов")
            
            return embeddings
            
        except Exception as e:
            st.error(f"Ошибка генерации эмбеддингов: {str(e)}")
            return []
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to the vector store
        
        Args:
            documents: List of Document objects
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                return False
            
            # Prepare data for ChromaDB
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Generate unique IDs for each document
            ids = []
            for i, doc in enumerate(documents):
                # Create unique ID based on filename and chunk
                filename = doc.metadata.get("filename", "unknown")
                chunk_id = doc.metadata.get("chunk_id", i)
                unique_id = f"{filename}_{chunk_id}_{hashlib.md5(doc.page_content.encode()).hexdigest()[:8]}"
                ids.append(unique_id)
            
            # Generate embeddings
            st.info("Generating embeddings...")
            embeddings = self.generate_embeddings(texts)
            
            if not embeddings:
                return False
            
            # Add to collection
            st.info("Adding documents to vector store...")
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            st.success(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            return False
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            st.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection
        
        Returns:
            Collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "document_count": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            st.error(f"Error getting collection info: {str(e)}")
            return {"document_count": 0}
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection
            self.client.delete_collection(self.collection_name)
            
            # Recreate the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            st.success("Collection cleared successfully")
            return True
            
        except Exception as e:
            st.error(f"Error clearing collection: {str(e)}")
            return False
    
    def delete_documents_by_filename(self, filename: str) -> bool:
        """
        Delete all documents from a specific file
        
        Args:
            filename: Name of the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all documents
            results = self.collection.get(
                include=["metadatas"]
            )
            
            # Find IDs of documents to delete
            ids_to_delete = []
            for i, metadata in enumerate(results['metadatas']):
                if metadata.get('filename') == filename:
                    ids_to_delete.append(results['ids'][i])
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                st.success(f"Deleted {len(ids_to_delete)} documents from {filename}")
                return True
            else:
                st.warning(f"No documents found for {filename}")
                return False
                
        except Exception as e:
            st.error(f"Error deleting documents: {str(e)}")
            return False
    
    def get_document_summary(self) -> Dict[str, Any]:
        """
        Get summary of documents in the collection
        
        Returns:
            Summary statistics
        """
        try:
            results = self.collection.get(
                include=["metadatas"]
            )
            
            if not results['metadatas']:
                return {"total_documents": 0, "unique_files": 0}
            
            filenames = [metadata.get('filename', 'unknown') for metadata in results['metadatas']]
            unique_files = len(set(filenames))
            
            return {
                "total_documents": len(results['metadatas']),
                "unique_files": unique_files,
                "filenames": list(set(filenames))
            }
            
        except Exception as e:
            st.error(f"Error getting document summary: {str(e)}")
            return {"total_documents": 0, "unique_files": 0}