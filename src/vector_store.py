
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Union
import ollama
import streamlit as st
from langchain.schema import Document
import os
import hashlib
import re
class VectorStore:
    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = "./data/chroma_db", embedding_model: str = "nomic-embed-text:latest"):
        self.base_collection_name = collection_name
        self.persist_directory = os.path.abspath(persist_directory)
        self.embedding_model = embedding_model
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        if os.path.exists(self.persist_directory):
            os.chmod(self.persist_directory, 0o755)
        
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        
        self.collection_name = self._get_collection_name_for_model(embedding_model)
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": embedding_model}
        )
    
    def _get_collection_name_for_model(self, model_name: str) -> str:
        model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
        return f"{self.base_collection_name}_{model_hash}"
    
    def update_embedding_model(self, model_name: str) -> bool:
        try:
            test_response = ollama.embeddings(
                model=model_name,
                prompt="test"
            )
            
            if 'embedding' in test_response and test_response['embedding']:
                self.embedding_model = model_name
                
                self.collection_name = self._get_collection_name_for_model(model_name)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine", "embedding_model": model_name}
                )
                
                return True
            else:
                st.error(f"Failed to test embedding model {model_name}")
                return False
                
        except Exception as e:
            st.error(f"Failed to update embedding model to {model_name}: {str(e)}")
            return False
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = []
            
            if len(texts) > 10:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            for i, text in enumerate(texts):
                try:
                    response = ollama.embeddings(
                        model=self.embedding_model,
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
        try:
            if not documents:
                return False
            
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            ids = []
            for i, doc in enumerate(documents):
                filename = doc.metadata.get("filename", "unknown")
                chunk_id = doc.metadata.get("chunk_id", i)
                unique_id = f"{filename}_{chunk_id}_{hashlib.md5(doc.page_content.encode()).hexdigest()[:8]}"
                ids.append(unique_id)
            
            st.info("Generating embeddings...")
            embeddings = self.generate_embeddings(texts)
            
            if not embeddings:
                return False
            
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
    
    def search_similar(self, query: str, k: int = 5, selected_documents: Any = "all", distance_threshold: float = 0.7, use_hybrid: bool = True) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.generate_embeddings([query])[0]
            
            search_params = {
                "query_embeddings": [query_embedding],
                "n_results": min(k * 3, 50),
                "include": ["documents", "metadatas", "distances"]
            }
            
            if selected_documents != "all" and isinstance(selected_documents, list) and selected_documents:
                search_params["where"] = {"filename": {"$in": selected_documents}}
            
            results = self.collection.query(**search_params)
            
            formatted_results = []
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                similarity = 1 - distance
                
                if distance <= distance_threshold:
                    result = {
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": distance,
                        "similarity": similarity
                    }
                    formatted_results.append(result)
            
            return formatted_results[:k]
            
        except Exception as e:
            st.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
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
        try:
            self.client.delete_collection(self.collection_name)
            
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
        try:
            results = self.collection.get(
                include=["metadatas"]
            )
            
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
    
    def delete_documents_by_filenames(self, filenames: List[str]) -> Dict[str, bool]:
        results = {}
        
        for filename in filenames:
            results[filename] = self.delete_documents_by_filename(filename)
        
        return results
    
    def update_filename_in_metadata(self, old_filename: str, new_filename: str) -> bool:
        try:
            results = self.collection.get(
                where={"filename": old_filename},
                include=["metadatas", "documents", "embeddings"]
            )
            
            if not results['ids']:
                st.warning(f"No documents found for {old_filename}")
                return False
            
            updated_metadatas = []
            for metadata in results['metadatas']:
                updated_metadata = metadata.copy()
                updated_metadata['filename'] = new_filename
                updated_metadatas.append(updated_metadata)
            
            self.collection.delete(ids=results['ids'])
            
            self.collection.add(
                ids=results['ids'],
                documents=results['documents'],
                metadatas=updated_metadatas,
                embeddings=results['embeddings']
            )
            
            st.success(f"Updated filename from {old_filename} to {new_filename}")
            return True
            
        except Exception as e:
            st.error(f"Error updating filename: {str(e)}")
            return False
    
    def get_document_summary(self) -> Dict[str, Any]:
        try:
            results = self.collection.get(include=["metadatas"])
            
            if not results or not results.get('metadatas'):
                return {
                    "total_documents": 0, 
                    "unique_files": 0, 
                    "filenames": [], 
                    "file_details": {}
                }
            
            filenames = []
            for metadata in results['metadatas']:
                filename = metadata.get('filename', 'unknown')
                if filename and filename != 'unknown':
                    filenames.append(filename)
            
            unique_filenames = list(set(filenames))
            
            file_details = {}
            for filename in unique_filenames:
                file_chunks = [m for m in results['metadatas'] if m.get('filename') == filename]
                if file_chunks:
                    first_chunk = file_chunks[0]
                    file_details[filename] = {
                        "chunk_count": len(file_chunks),
                        "page_count": first_chunk.get('page_count', 'Unknown'),
                        "file_path": first_chunk.get('file_path', 'Unknown'),
                        "original_name": first_chunk.get('original_name', filename)
                    }
            
            return {
                "total_documents": len(results['metadatas']),
                "unique_files": len(unique_filenames),
                "filenames": unique_filenames,
                "file_details": file_details
            }
            
        except Exception as e:
            st.error(f"Error getting document summary: {str(e)}")
            return {"total_documents": 0, "unique_files": 0, "filenames": [], "file_details": {}}
    
    def get_document_preview(self, filename: str, max_length: int = 300) -> str:
        try:
            results = self.collection.get(
                where={"filename": filename},
                include=["documents"],
                limit=2
            )
            
            if not results or not results.get('documents') or len(results['documents']) == 0:
                return "Предпросмотр недоступен"
            
            preview_text = ""
            for doc_text in results['documents']:
                if doc_text and doc_text.strip():
                    preview_text += doc_text.strip() + " "
                    if len(preview_text) > max_length:
                        break
            
            preview_text = preview_text.strip()
            
            preview_text = re.sub(r'--- Page \d+ ---', '', preview_text)
            
            if len(preview_text) > max_length:
                preview_text = preview_text[:max_length].rsplit(' ', 1)[0] + "..."
            
            return preview_text if preview_text else "Контент не найден"
            
        except Exception as e:
            return f"Ошибка загрузки предпросмотра: {str(e)}"
    
    def get_full_document_content(self, filename: str) -> Dict[str, Any]:
        try:
            results = self.collection.get(
                where={"filename": filename},
                include=["documents", "metadatas"]
            )
            
            if not results or not results.get('documents'):
                return {
                    "success": False,
                    "error": "Документ не найден",
                    "content": "",
                    "chunks": []
                }
            
            chunks_data = []
            for i, doc_text in enumerate(results['documents']):
                metadata = results['metadatas'][i] if i < len(results['metadatas']) else {}
                chunk_id = metadata.get('chunk_id', i)
                
                chunks_data.append({
                    "chunk_id": chunk_id,
                    "text": doc_text,
                    "metadata": metadata
                })
            
            chunks_data.sort(key=lambda x: x['chunk_id'])
            
            full_text = ""
            for chunk_data in chunks_data:
                text = chunk_data['text'].strip()
                if text:
                    full_text += text + "\n\n"
            
            full_text = full_text.strip()
            
            full_text = re.sub(r'--- Page \d+ ---\s*', '\n\n=== Страница ===\n\n', full_text)
            
            full_text = re.sub(r'\n\s*\n\s*\n', '\n\n', full_text)
            
            first_metadata = chunks_data[0]['metadata'] if chunks_data else {}
            
            return {
                "success": True,
                "content": full_text,
                "chunks": chunks_data,
                "total_chunks": len(chunks_data),
                "page_count": first_metadata.get('page_count', 'Unknown'),
                "file_path": first_metadata.get('file_path', 'Unknown'),
                "original_name": first_metadata.get('original_name', filename),
                "total_characters": len(full_text)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Ошибка загрузки документа: {str(e)}",
                "content": "",
                "chunks": []
            }