
from typing import Dict, Any, List, Optional
import streamlit as st
import time
from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .llm_manager import LLMManager
from .router import SmartRouter
from .config import ConfigManager
class RAGPipeline:
    def __init__(self):
        self.config_manager = ConfigManager()
        config = self.config_manager.get_current_config()
        self.document_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        self.vector_store = VectorStore(embedding_model=config.embedding_model)
        self.llm_manager = LLMManager(model_name=config.llm_model)
        self.router = SmartRouter(self.llm_manager, self.vector_store)
        self.stats = {
            "total_queries": 0,
            "successful_answers": 0,
            "failed_answers": 0,
            "average_response_time": 0.0,
            "total_documents": 0
        }
    def update_models(self, llm_model: str = None, embedding_model: str = None) -> bool:
        success = True
        if llm_model:
            if self.config_manager.update_llm_model(llm_model):
                if self.llm_manager.update_model(llm_model):
                    st.success(f"LLM модель обновлена: {llm_model}")
                else:
                    st.error(f"Не удалось обновить LLM модель: {llm_model}")
                    success = False
            else:
                st.error(f"Модель {llm_model} недоступна")
                success = False
        
        if embedding_model:
            if self.config_manager.update_embedding_model(embedding_model):
                if self.vector_store.update_embedding_model(embedding_model):
                    st.success(f"Embedding модель обновлена: {embedding_model}")
                    st.info("Создана новая коллекция для этой модели")
                    st.warning("Необходимо переиндексировать документы для новой модели")
                else:
                    st.error(f"Не удалось обновить embedding модель: {embedding_model}")
                    success = False
            else:
                st.error(f"Embedding модель {embedding_model} недоступна")
                success = False
        
        return success
    
    def initialize_system(self, silent: bool = False) -> bool:
        try:
            if not silent:
                st.info("Проверка системных компонентов...")
            
            try:
                import requests
                response = requests.get("http://localhost:11434/api/version", timeout=5)
                if response.status_code != 200:
                    st.error("Ollama сервис недоступен. Запустите: ollama serve")
                    return False
                else:
                    if not silent:
                        st.success("Ollama сервис запущен")
            except requests.exceptions.ConnectionError:
                st.error("Ollama сервис не запущен. Запустите: ollama serve")
                return False
            except Exception as e:
                st.warning(f"Не удалось проверить статус Ollama: {e}")
            
            if not self.llm_manager.model_name:
                st.error("Не найдено доступных LLM моделей")
                st.info("Установите LLM модель: ollama pull <model_name>")
                return False
                
            if not silent:
                st.info(f"Тестирование модели LLM ({self.llm_manager.model_name})...")
            try:
                import ollama
                test_response = ollama.generate(
                    model=self.llm_manager.model_name,
                    prompt="test",
                    options={"num_predict": 1}
                )
                if test_response and 'response' in test_response:
                    if not silent:
                        st.success(f"Модель LLM ({self.llm_manager.model_name}) работает")
                else:
                    st.error("Модель LLM не отвечает корректно")
                    return False
            except Exception as e:
                st.error(f"Ошибка тестирования LLM модели: {e}")
                st.info(f"Убедитесь, что модель установлена: ollama pull {self.llm_manager.model_name}")
                return False
            
            if not self.vector_store.embedding_model:
                st.error("Не найдено доступных Embedding моделей")
                st.info("Установите Embedding модель: ollama pull <model_name>")
                return False
                
            if not silent:
                st.info(f"Тестирование модели эмбеддингов ({self.vector_store.embedding_model})...")
            try:
                import ollama
                test_response = ollama.embeddings(
                    model=self.vector_store.embedding_model,
                    prompt="test"
                )
                if 'embedding' in test_response and test_response['embedding']:
                    if not silent:
                        st.success(f"Модель эмбеддингов ({self.vector_store.embedding_model}) работает")
                else:
                    st.error("Модель эмбеддингов не возвращает векторы")
                    return False
            except Exception as e:
                st.error(f"Ошибка тестирования модели эмбеддингов: {e}")
                st.info(f"Убедитесь, что модель установлена: ollama pull {self.vector_store.embedding_model}")
                return False
            
            collection_info = self.vector_store.get_collection_info()
            if collection_info.get("document_count", 0) == 0:
                st.warning("Векторная база пуста. Загрузите документы для начала работы.")
            else:
                if not silent:
                    st.info(f"Загружено {collection_info.get('document_count', 0)} фрагментов")
            
            if not silent:
                st.success("Все компоненты системы готовы к работе!")
            return True
            
        except Exception as e:
            st.error(f"Ошибка инициализации системы: {str(e)}")
            return False
    
    def load_documents_from_directory(self, directory_path: str) -> bool:
        try:
            st.info("Загрузка документов...")
            
            documents = self.document_processor.process_pdf_directory(directory_path)
            
            if not documents:
                st.error("Не удалось обработать документы")
                return False
            
            # Add to vector store
            success = self.vector_store.add_documents(documents)
            
            if success:
                self.stats["total_documents"] = len(documents)
                st.success(f"Успешно загружено {len(documents)} фрагментов документов")
                return True
            else:
                st.error("Не удалось добавить документы в векторную базу")
                return False
                
        except Exception as e:
            st.error(f"Ошибка загрузки документов: {str(e)}")
            return False
    
    def reindex_existing_documents(self, directory_path: str) -> bool:
        try:
            st.info(f"Переиндексация документов из {directory_path}...")
            
            documents = self.document_processor.process_pdf_directory(directory_path)
            
            if not documents:
                st.error("Не удалось обработать документы")
                return False
            
            # Add to vector store
            success = self.vector_store.add_documents(documents)
            
            if success:
                self.stats["total_documents"] = len(documents)
                st.success(f"Успешно переиндексировано {len(documents)} фрагментов документов")
                
                status = self.get_system_status()
                st.info(f"Статистика: {status['vector_store']['total_documents']} фрагментов из {status['vector_store']['unique_files']} файлов")
                return True
            else:
                st.error("Не удалось добавить документы в векторную базу")
                return False
                
        except Exception as e:
            st.error(f"Ошибка переиндексации документов: {str(e)}")
            return False

    def load_uploaded_file(self, uploaded_file) -> bool:
        try:
            st.info(f"Обработка файла: {uploaded_file.name}")
            
            documents = self.document_processor.process_uploaded_file(uploaded_file)
            
            if not documents:
                st.error("Не удалось обработать файл")
                return False
            
            # Add to vector store
            success = self.vector_store.add_documents(documents)
            
            if success:
                self.stats["total_documents"] += len(documents)
                st.success(f"Файл обработан: {len(documents)} фрагментов добавлено")
                return True
            else:
                st.error("Не удалось добавить документы в векторную базу")
                return False
                
        except Exception as e:
            st.error(f"Ошибка обработки файла: {str(e)}")
            return False
    
    def process_query(self, query: str, show_debug: bool = False, selected_documents: Any = "all") -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            search_results = self.vector_store.search_similar(
                query, 
                k=15, 
                selected_documents=selected_documents,
                distance_threshold=0.6
            )
            
            routing_result = self.router.route_query(query, search_results)
            
            if routing_result["can_answer"]:
                enhanced_context = self.router.enhance_context(
                    routing_result["context"], 
                    query
                )
                
                answer = self.llm_manager.generate_response(
                    prompt=query,
                    context=enhanced_context,
                    temperature=0.2
                )
                
                response_type = "success"
                self.stats["successful_answers"] += 1
                
            else:
                if routing_result.get("context", "").strip():
                    answer = self.llm_manager.generate_response(
                        prompt=query,
                        context=routing_result["context"],
                        temperature=0.2
                    )
                    response_type = "partial"
                else:
                    answer = self._generate_fallback_response(query, routing_result)
                    response_type = "fallback"
                self.stats["failed_answers"] += 1
            
            response_time = time.time() - start_time
            self.stats["total_queries"] += 1
            self.stats["average_response_time"] = (
                (self.stats["average_response_time"] * (self.stats["total_queries"] - 1) + response_time) 
                / self.stats["total_queries"]
            )
            
            
            complete_response = {
                "answer": answer,
                "response_type": response_type,
                "response_time": response_time,
                "routing_result": routing_result,
                "search_results": search_results,
                "query_analysis": routing_result.get("query_analysis", {}),
                "sources": self._extract_sources(search_results)
            }
            
            return complete_response
            
        except Exception as e:
            st.error(f"Ошибка обработки запроса: {str(e)}")
            return {
                "answer": "Извините, произошла ошибка при обработке вашего запроса.",
                "response_type": "error",
                "error": str(e)
            }
    
    def _generate_fallback_response(self, query: str, routing_result: Dict[str, Any]) -> str:
        language = routing_result.get("query_analysis", {}).get("language", "russian")
        confidence = routing_result.get("confidence", 0.0)
        
        if language == "kazakh":
            return f"""Кешіріңіз, жүктелген құжаттарда сіздің сұрағыңызға толық жауап беру үшін жеткілікті ақпарат жоқ (сенімділік: {confidence:.2f}).

Мүмкін:
1. Сұрақты басқаша тұжырымдау
2. Қосымша құжаттар жүктеу
3. Неғұрлым нақты сұрақ қою

Табылған ақпарат негізінде ішінара жауап беруге тырысайын..."""
        elif language == "russian":
            return f"""Извините, в загруженных документах недостаточно информации для полного ответа на ваш вопрос (уверенность: {confidence:.2f}).

Возможные варианты:
1. Переформулировать вопрос
2. Загрузить дополнительные документы
3. Задать более конкретный вопрос

Попробую дать частичный ответ на основе найденной информации..."""
        else:
            return f"""Sorry, there's insufficient information in the loaded documents to fully answer your question (confidence: {confidence:.2f}).

Possible options:
1. Rephrase the question
2. Upload additional documents
3. Ask a more specific question

I'll try to provide a partial answer based on the available information..."""
    
    def _extract_sources(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        seen_files = set()
        
        for result in search_results:
            metadata = result.get("metadata", {})
            filename = metadata.get("filename", "Unknown")
            
            if filename not in seen_files:
                sources.append({
                    "filename": filename,
                    "page_count": metadata.get("page_count", "Unknown"),
                    "relevance": 1 - result.get("distance", 1.0)
                })
                seen_files.add(filename)
        
        return sources
    
    def get_system_status(self) -> Dict[str, Any]:
        collection_info = self.vector_store.get_collection_info()
        document_summary = self.vector_store.get_document_summary()
        
        llm_info = self.llm_manager.get_model_info()
        
        router_metrics = self.router.get_routing_metrics()
        
        return {
            "vector_store": {
                "total_documents": collection_info.get("document_count", 0),
                "unique_files": document_summary.get("unique_files", 0),
                "filenames": document_summary.get("filenames", [])
            },
            "llm": llm_info,
            "router": router_metrics,
            "pipeline_stats": self.stats.copy()
        }
    
    def clear_all_data(self) -> bool:
        try:
            success = self.vector_store.clear_collection()
            if success:
                self.stats = {
                    "total_queries": 0,
                    "successful_answers": 0,
                    "failed_answers": 0,
                    "average_response_time": 0.0,
                    "total_documents": 0
                }
            return success
        except Exception as e:
            st.error(f"Ошибка очистки данных: {str(e)}")
            return False
    
    def export_conversation(self, conversation_history: List[Dict[str, Any]]) -> str:
        export_text = "# RAG System - История разговоров\n\n"
        
        for i, entry in enumerate(conversation_history, 1):
            export_text += f"## Запрос {i}\n"
            export_text += f"**Вопрос:** {entry.get('question', 'N/A')}\n\n"
            export_text += f"**Ответ:** {entry.get('answer', 'N/A')}\n\n"
            
            if entry.get('sources'):
                export_text += "**Источники:**\n"
                for source in entry['sources']:
                    export_text += f"- {source.get('filename', 'Unknown')}\n"
                export_text += "\n"
            
            export_text += "---\n\n"
        
        return export_text