
import fitz  # PyMuPDF
import os
import re
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit as st
from enum import Enum
from dataclasses import dataclass


class ProcessingStage(Enum):
    INITIALIZING = "initializing"
    READING_PDF = "reading_pdf"
    PROCESSING_TEXT = "processing_text"
    CREATING_CHUNKS = "creating_chunks"
    GENERATING_EMBEDDINGS = "generating_embeddings"
    STORING_DOCUMENTS = "storing_documents"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ProgressState:
    stage: ProcessingStage = ProcessingStage.INITIALIZING
    current_file: str = ""
    current_step: int = 0
    total_steps: int = 0
    current_file_index: int = 0
    total_files: int = 0
    message: str = ""
    error_message: str = ""
    start_time: float = 0
    stage_start_time: float = 0
    
    @property
    def progress_percent(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return min(100.0, (self.current_step / self.total_steps) * 100.0)
    
    @property
    def file_progress_percent(self) -> float:
        if self.total_files == 0:
            return 0.0
        return min(100.0, (self.current_file_index / self.total_files) * 100.0)
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time if self.start_time > 0 else 0
    
    @property
    def stage_elapsed_time(self) -> float:
        return time.time() - self.stage_start_time if self.stage_start_time > 0 else 0


class SimpleProgressTracker:
    """Упрощенный трекер прогресса для v1.5.0, интегрированный со Streamlit"""
    
    def __init__(self):
        self.state = ProgressState()
        self.progress_placeholder = None
        self.status_placeholder = None
        self.detail_placeholder = None
        self._is_active = False
    
    def setup_ui(self, container=None):
        """Настройка UI элементов для отображения прогресса"""
        if container is None:
            container = st.container()
        
        with container:
            self.progress_placeholder = st.empty()
            self.status_placeholder = st.empty()
            self.detail_placeholder = st.empty()
    
    def start_session(self, total_files: int, title: str = "Обработка документов"):
        """Начать сессию обработки"""
        self.state = ProgressState(
            total_files=total_files,
            start_time=time.time(),
            stage_start_time=time.time()
        )
        self._is_active = True
        
        if self.progress_placeholder is None:
            self.setup_ui()
        
        self._update_display(f"{title} ({total_files} файлов)")
    
    def start_file(self, filename: str, total_steps: int = 5):
        """Начать обработку файла"""
        self.state.current_file = filename
        self.state.current_file_index += 1
        self.state.current_step = 0
        self.state.total_steps = total_steps
        self.state.stage = ProcessingStage.READING_PDF
        self.state.stage_start_time = time.time()
        self.state.message = f"Начинаю обработку файла {filename}"
        self.state.error_message = ""
        
        self._update_display()
    
    def update_stage(self, stage: ProcessingStage, message: str = "", step_increment: int = 1):
        """Обновить текущий этап"""
        self.state.stage = stage
        self.state.current_step += step_increment
        self.state.stage_start_time = time.time()
        
        if message:
            self.state.message = message
        else:
            self.state.message = self._get_default_message(stage)
        
        self._update_display()
    
    def update_progress(self, current: int, total: int = None, message: str = ""):
        """Обновить прогресс текущего этапа"""
        self.state.current_step = current
        if total is not None:
            self.state.total_steps = total
        if message:
            self.state.message = message
        
        self._update_display()
    
    def set_error(self, error_message: str):
        """Зарегистрировать ошибку"""
        self.state.stage = ProcessingStage.ERROR
        self.state.error_message = error_message
        self.state.message = f"Ошибка: {error_message}"
        
        self._update_display()
    
    def complete_file(self):
        """Завершить обработку текущего файла"""
        self.state.stage = ProcessingStage.COMPLETED
        self.state.current_step = self.state.total_steps
        self.state.message = f"Файл {self.state.current_file} обработан успешно"
        
        self._update_display()
    
    def finish_session(self, success: bool = True):
        """Завершить сессию"""
        self._is_active = False
        
        if success:
            self._show_final_success()
        else:
            self._show_final_error()
    
    def _update_display(self, title: str = None):
        """Обновить отображение прогресса"""
        if not self._is_active or not self.progress_placeholder:
            return
        
        try:
            # Основной прогресс-бар (файлы)
            with self.progress_placeholder:
                if title:
                    st.subheader(title)
                
                if self.state.total_files > 0:
                    file_progress = self.state.file_progress_percent / 100
                    st.progress(
                        file_progress, 
                        text=f"Файл {self.state.current_file_index}/{self.state.total_files}"
                    )
            
            # Статус текущего файла
            with self.status_placeholder:
                if self.state.current_file:
                    step_progress = self.state.progress_percent / 100
                    
                    st.progress(
                        step_progress,
                        text=f"{self.state.current_file}: {self.state.message} ({self.state.progress_percent:.0f}%)"
                    )
            
            # Детали и ошибки
            with self.detail_placeholder:
                if self.state.error_message:
                    st.error(f"Ошибка: {self.state.error_message}")
                elif self.state.stage_elapsed_time > 0:
                    elapsed = self.state.stage_elapsed_time
                    if elapsed < 60:
                        time_str = f"{elapsed:.1f} сек"
                    else:
                        time_str = f"{elapsed/60:.1f} мин"
                    
                    st.caption(f"Время этапа: {time_str}")
        
        except Exception as e:
            # Игнорируем ошибки UI для стабильности
            pass
    
    def _show_final_success(self):
        """Показать финальное сообщение об успехе"""
        if self.progress_placeholder:
            with self.progress_placeholder:
                st.success(f"Успешно обработано {self.state.total_files} файлов")
        
        if self.status_placeholder:
            with self.status_placeholder:
                total_time = self.state.elapsed_time
                if total_time < 60:
                    time_str = f"{total_time:.1f} секунд"
                else:
                    time_str = f"{total_time/60:.1f} минут"
                st.info(f"Общее время обработки: {time_str}")
        
        if self.detail_placeholder:
            with self.detail_placeholder:
                st.empty()
    
    def _show_final_error(self):
        """Показать финальное сообщение об ошибке"""
        if self.progress_placeholder:
            with self.progress_placeholder:
                st.error("Обработка завершена с ошибками")
    
    def _get_default_message(self, stage: ProcessingStage) -> str:
        """Получить сообщение по умолчанию для этапа"""
        message_map = {
            ProcessingStage.INITIALIZING: "Инициализация",
            ProcessingStage.READING_PDF: "Чтение PDF файла",
            ProcessingStage.PROCESSING_TEXT: "Обработка текста",
            ProcessingStage.CREATING_CHUNKS: "Создание фрагментов",
            ProcessingStage.GENERATING_EMBEDDINGS: "Генерация эмбеддингов",
            ProcessingStage.STORING_DOCUMENTS: "Сохранение в базу данных",
            ProcessingStage.COMPLETED: "Завершено",
            ProcessingStage.ERROR: "Ошибка"
        }
        return message_map.get(stage, "Обработка")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Получить текущее состояние для отладки"""
        return {
            "stage": self.state.stage.value,
            "current_file": self.state.current_file,
            "progress_percent": self.state.progress_percent,
            "file_progress_percent": self.state.file_progress_percent,
            "current_step": self.state.current_step,
            "total_steps": self.state.total_steps,
            "current_file_index": self.state.current_file_index,
            "total_files": self.state.total_files,
            "message": self.state.message,
            "error_message": self.state.error_message,
            "elapsed_time": self.state.elapsed_time,
            "is_active": self._is_active
        }


class ProgressContext:
    """Контекстный менеджер для автоматического управления прогрессом"""
    
    def __init__(self, total_files: int, title: str = "Обработка документов"):
        self.tracker = SimpleProgressTracker()
        self.total_files = total_files
        self.title = title
        self.success = True
    
    def __enter__(self):
        self.tracker.setup_ui()
        self.tracker.start_session(self.total_files, self.title)
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.success = False
            if hasattr(exc_val, '__str__'):
                self.tracker.set_error(str(exc_val))
        
        self.tracker.finish_session(self.success)
        return False  # Не подавлять исключения


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, progress_tracker: Optional[SimpleProgressTracker] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.progress_tracker = progress_tracker
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\nСтатья", "\n\n", ".\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
    
    def update_chunk_settings(self, chunk_size: int, chunk_overlap: int):
        """Обновить настройки разбиения на фрагменты"""
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
            if self.progress_tracker:
                self.progress_tracker.update_stage(ProcessingStage.READING_PDF, "Открытие PDF файла")
            
            doc = fitz.open(pdf_path)
            text = ""
            metadata = {
                "filename": os.path.basename(pdf_path),
                "page_count": len(doc),
                "file_path": pdf_path
            }
            
            if self.progress_tracker:
                self.progress_tracker.update_stage(ProcessingStage.PROCESSING_TEXT, f"Извлечение текста из {len(doc)} страниц")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text().strip()
                if page_text:
                    text += f"\n\n{page_text}"
                
                if self.progress_tracker:
                    progress = int(((page_num + 1) / len(doc)) * 100)
                    self.progress_tracker.update_progress(page_num + 1, len(doc), f"Обработана страница {page_num + 1} из {len(doc)}")
            
            doc.close()
            
            return {
                "text": text.strip(),
                "metadata": metadata
            }
            
        except Exception as e:
            error_msg = f"Error extracting text from {pdf_path}: {str(e)}"
            if self.progress_tracker:
                self.progress_tracker.set_error(error_msg)
            st.error(error_msg)
            return None
    
    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        if self.progress_tracker:
            self.progress_tracker.update_stage(ProcessingStage.CREATING_CHUNKS, "Разбиение текста на фрагменты")
        
        chunks = self.text_splitter.split_text(text)
        valid_chunks = [c for c in chunks if len(c.strip()) >= 50]
        
        if self.progress_tracker:
            self.progress_tracker.update_progress(0, len(valid_chunks), f"Создание {len(valid_chunks)} фрагментов")
        
        documents = []
        valid_chunk_id = 0
        
        for i, chunk in enumerate(chunks):
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
                
                if self.progress_tracker:
                    self.progress_tracker.update_progress(valid_chunk_id, len(valid_chunks), f"Создан фрагмент {valid_chunk_id} из {len(valid_chunks)}")
        
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
        
        # Если есть прогресс-трекер, используем его, иначе fallback к обычным прогресс-барам
        if self.progress_tracker:
            if not self.progress_tracker._is_active:
                self.progress_tracker.start_session(len(pdf_files), f"Обработка {len(pdf_files)} PDF файлов")
            
            for i, pdf_file in enumerate(pdf_files):
                try:
                    self.progress_tracker.start_file(pdf_file, 5)  # 5 основных этапов
                    pdf_path = os.path.join(directory_path, pdf_file)
                    
                    documents = self.process_pdf_file(pdf_path)
                    all_documents.extend(documents)
                    
                    self.progress_tracker.complete_file()
                    
                except Exception as e:
                    self.progress_tracker.set_error(f"Ошибка обработки {pdf_file}: {str(e)}")
                    continue
            
        else:
            # Fallback к стандартным прогресс-барам Streamlit
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
            if self.progress_tracker:
                self.progress_tracker.start_file(uploaded_file.name, 5)
            
            docs_dir = os.path.abspath("./data/documents")
            os.makedirs(docs_dir, exist_ok=True)
            
            file_content = uploaded_file.getbuffer()
            
            readable_name = self._generate_readable_filename(file_content, uploaded_file.name)
            
            final_path = self._get_unique_filepath(docs_dir, readable_name)
            final_filename = os.path.basename(final_path)
            
            with open(final_path, "wb") as f:
                f.write(file_content)
            
            if not self.progress_tracker:
                st.info(f"Файл сохранен: {final_filename}")
            
            extracted_data = self.extract_text_from_pdf(final_path)
            if not extracted_data:
                error_msg = "Не удалось извлечь текст из PDF"
                if self.progress_tracker:
                    self.progress_tracker.set_error(error_msg)
                else:
                    st.error(error_msg)
                return []
            
            extracted_data["metadata"]["filename"] = final_filename
            extracted_data["metadata"]["file_path"] = final_path
            extracted_data["metadata"]["original_name"] = uploaded_file.name
            
            documents = self.chunk_document(
                extracted_data["text"], 
                extracted_data["metadata"]
            )
            
            if self.progress_tracker:
                self.progress_tracker.complete_file()
            elif documents:
                st.success(f"Создано {len(documents)} фрагментов из {extracted_data['metadata']['page_count']} страниц")
            
            return documents
            
        except Exception as e:
            error_msg = f"Ошибка обработки файла: {str(e)}"
            if self.progress_tracker:
                self.progress_tracker.set_error(error_msg)
            else:
                st.error(error_msg)
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