import streamlit as st
import os
import time
import uuid
import re
from datetime import datetime, timedelta
from src .main import RAGPipeline
from src .document_processor import SimpleProgressTracker, ProgressContext
from src .session_manager import SessionManager
import json
import markdown
from bs4 import BeautifulSoup


st .set_page_config(
    page_title="RAG System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


if "session_manager"not in st .session_state:
    st .session_state .session_manager = SessionManager()

    st .session_state .session_manager .initialize_with_autorestore()
if "rag_pipeline"not in st .session_state:
    st .session_state .rag_pipeline = RAGPipeline()
if "conversation_history"not in st .session_state:
    st .session_state .conversation_history = []
if "system_initialized"not in st .session_state:
    st .session_state .system_initialized = False
if "debug_mode"not in st .session_state:
    st .session_state .debug_mode = False
if "selected_documents"not in st .session_state:
    st .session_state .selected_documents = "all"
if "full_content_dialog"not in st .session_state:
    st .session_state .full_content_dialog = {
        "show": False, "filename": "", "content_data": None}
if "chat_sessions"not in st .session_state:
    st .session_state .chat_sessions = {"default": {
        "name": "Новый чат", "messages": [], "created_at": datetime .now()}}
if "current_session"not in st .session_state:
    st .session_state .current_session = "default"
if "show_rename_dialog"not in st .session_state:
    st .session_state .show_rename_dialog = None
if "show_dropdown_menu"not in st .session_state:
    st .session_state .show_dropdown_menu = None
if "documents_to_delete"not in st .session_state:
    st .session_state .documents_to_delete = []
if "rename_dialog"not in st .session_state:
    st .session_state .rename_dialog = {
        "show": False, "filename": "", "new_name": ""}
st .sidebar .title("RAG System")
page = st .sidebar .selectbox(
    "Выберите страницу",
    ["Главная", "Чат", "Документы", "Настройки"]
)
if not st .session_state .system_initialized:
    try:
        with st .spinner("Инициализация системы..."):
            st .session_state .system_initialized = st .session_state .rag_pipeline .initialize_system(
                silent=True)
    except Exception as e:
        with st .sidebar:
            st .subheader("Инициализация системы")
            st .error(f"Автоматическая инициализация не удалась: {str(e)}")
            if st .button("Повторить инициализацию", type="primary"):
                st .session_state .system_initialized = st .session_state .rag_pipeline .initialize_system()
st .session_state .debug_mode = st .sidebar .checkbox(
    "Режим отладки", value=st .session_state .debug_mode)

if page == "Чат":
    st .sidebar .markdown("""<style>div[data-testid="stSidebar"].stButton > button {background: transparent;border: 1px solid rgba(255, 255, 255, 0.2);color: rgba(255, 255, 255, 0.9);padding: 8px 12px;border-radius: 6px;font-size: 14px;font-weight: normal;transition: all 0.2s ease;min-height: 36px;}div[data-testid="stSidebar"].stButton > button:hover {background: rgba(255, 255, 255, 0.1);border-color: rgba(255, 255, 255, 0.3);color: white;}div[data-testid="stSidebar"].stSelectbox > div > div {background: transparent;border: 1px solid rgba(255, 255, 255, 0.2);border-radius: 6px;color: rgba(255, 255, 255, 0.9);min-height: 36px;}div[data-testid="stSidebar"].stSelectbox > div > div > div {color: rgba(255, 255, 255, 0.9);padding: 8px 12px;font-size: 14px;}div[data-testid="stSidebar"].stTextInput > div > div > input {background: transparent;border: 1px solid rgba(255, 255, 255, 0.2);color: rgba(255, 255, 255, 0.9);padding: 8px 12px;border-radius: 6px;font-size: 14px;min-height: 36px;}div[data-testid="stSidebar"].stTextInput > div > div > input:focus {border-color: rgba(255, 255, 255, 0.4);outline: none;background: rgba(255, 255, 255, 0.05);}div[data-testid="stSidebar"].stSelectbox svg {display: none;}div[data-testid="stSidebar"].stSelectbox > div > div::after {content: "⋯";position: absolute;right: 12px;top: 50%;transform: translateY(-50%);color: rgba(255, 255, 255, 0.5);font-size: 16px;pointer-events: none;}</style>""", unsafe_allow_html=True)

    all_chats = []
    chat_options = {}

    for session_id, session_data in st .session_state .chat_sessions .items():
        display_name = session_data["name"]
        if session_data["messages"] and display_name == "Новый чат":
            first_user_msg = next(
                (msg for msg in session_data["messages"]if msg["role"] == "user"), None)
            if first_user_msg:
                display_name = first_user_msg["content"][:50]
                if len(first_user_msg["content"]) > 50:
                    display_name += "..."

        all_chats .append(display_name)
        chat_options[display_name] = session_id

    current_chat_name = None
    for name, sid in chat_options .items():
        if sid == st .session_state .current_session:
            current_chat_name = name
            break

    current_index = all_chats .index(
        current_chat_name)if current_chat_name in all_chats else 0

    if st .session_state .show_rename_dialog:
        st .sidebar .markdown('<div class="rename-mode">',
                              unsafe_allow_html=True)
        current_name = st .session_state .chat_sessions[st .session_state .show_rename_dialog]["name"]
        new_name = st .sidebar .text_input(
            "Переименовать чат:",
            value=current_name,
            key="rename_input",
            label_visibility="collapsed",
            placeholder="Введите новое название..."
        )
        st .sidebar .markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st .sidebar .columns(2)
        with col1:
            if st .button("Сохранить", key="save_rename", use_container_width=True):
                if new_name .strip():
                    st .session_state .chat_sessions[st .session_state .show_rename_dialog]["name"] = new_name .strip(
                    )
                st .session_state .show_rename_dialog = None
                st .rerun()
        with col2:
            if st .button("Отмена", key="cancel_rename", use_container_width=True):
                st .session_state .show_rename_dialog = None
                st .rerun()
    else:
        selected_chat = st .sidebar .selectbox(
            "Выберите чат",
            all_chats,
            index=current_index,
            key="chat_selector"
        )

        if selected_chat and chat_options[selected_chat] != st .session_state .current_session:
            st .session_state .current_session = chat_options[selected_chat]
            st .rerun()

        if st .sidebar .button("Переименовать", key="rename_current", use_container_width=True):
            st .session_state .show_rename_dialog = st .session_state .current_session
            st .rerun()

        col1, col2 = st .sidebar .columns(2)
        with col1:
            if st .button("Новый", key="new_chat_btn", use_container_width=True):
                new_session_id = str(uuid .uuid4())[:8]
                st .session_state .chat_sessions[new_session_id] = {
                    "name": "Новый чат",
                    "messages": [],
                    "created_at": datetime .now()
                }
                st .session_state .current_session = new_session_id
                st .rerun()
        with col2:
            if st .button("Удалить", key="delete_current", use_container_width=True):
                if len(st .session_state .chat_sessions) > 1:
                    del st .session_state .chat_sessions[st .session_state .current_session]
                    st .session_state .current_session = list(
                        st .session_state .chat_sessions .keys())[0]
                    st .rerun()


if st .session_state .system_initialized:
    status = st .session_state .rag_pipeline .get_system_status()
    st .sidebar .subheader("Статус системы")
    st .sidebar .metric(
        "Фрагментов", status["vector_store"]["total_documents"])
    st .sidebar .metric("Загруженных файлов",
                        status["vector_store"]["unique_files"])
    st .sidebar .metric("Запросов", status["pipeline_stats"]["total_queries"])
    if st .session_state .selected_documents == "all":
        st .sidebar .write("**Активных:** Все документы")
    else:
        active_count = len(st .session_state .selected_documents)if isinstance(
            st .session_state .selected_documents, list)else 0
        total_count = status["vector_store"]["unique_files"]
        st .sidebar .write(f"**Активных:** {active_count} из {total_count}")

if page == "Главная":
    st .title("Главная страница")

    if not st .session_state .system_initialized:
        st .error(
            "Система не инициализирована. Проверьте систему в боковой панели.")
        st .stop()

    status = st .session_state .rag_pipeline .get_system_status()

    col1, col2, col3 = st .columns(3)
    with col1:
        st .metric("Документов", status["vector_store"]["unique_files"])
    with col2:
        st .metric("Фрагментов", status["vector_store"]["total_documents"])
    with col3:
        st .metric("Запросов", status["pipeline_stats"]["total_queries"])

    col1, col2 = st .columns(2)

    with col1:
        st .subheader("Основные возможности")
        st .markdown("""
 - **Hybrid Search** - комбинированный поиск по документам
 - **Точные ответы** - каждый ответ основан на загруженных документах
 - **Контекстные диалоги** - система помнит предыдущие вопросы
 - **Умная маршрутизация** - определение релевантности запросов
 - **Многоязычность** - поддержка русского и казахского языков
 - **Локальная обработка** - все данные остаются на вашем компьютере
 """)

    with col2:
        st .subheader("Технологический стек")
        st .markdown("""
 - **LLM:** Ollama
 - **Embeddings:** SentenceTransformers
 - **Векторная БД:** ChromaDB
 - **Поиск:** BM25 + Vector Hybrid
 - **Обработка PDF:** PyMuPDF
 - **Интерфейс:** Streamlit
 """)

    col1, col2 = st .columns(2)

    with col1:
        st .subheader("Используемые модели")
        config = st .session_state .rag_pipeline .config_manager .get_current_config()
        st .info(f"""
        **LLM модель:** {config .llm_model}

        **Embedding модель:** {config .embedding_model}
        """)

    with col2:
        st .subheader("Как начать")
        st .markdown("""
 1. Перейдите в **"Документы"** и загрузите PDF файлы
 2. Откройте **"Чат"** и задайте вопросы
 3. В **"Настройки"** можно изменить модели
 """)

    if status["vector_store"]["total_documents"] == 0:
        st .warning("Для начала работы загрузите документы")
    else:
        st .success(
            f"Система готова! Загружено {status['vector_store']['unique_files']} документов")
elif page == "Документы":
    st .title("Управление документами")
    if not st .session_state .system_initialized:
        st .error("Система не инициализирована. Сначала проверьте систему.")
    else:
        tab1, tab2, tab3 = st .tabs(["Загрузка", "Просмотр", "Управление"])
        with tab1:
            st .subheader("Загрузка документов")
            upload_method = st .radio(
                "Способ загрузки:",
                ["Загрузить файлы", "Загрузить из папки",
                 "Переиндексировать существующие"]
            )
            if upload_method == "Загрузить файлы":
                uploaded_files = st .file_uploader(
                    "Выберите документы",
                    type=["pdf", "docx", "txt"],
                    accept_multiple_files=True,
                    help="Поддерживаемые форматы: PDF, DOCX, TXT"
                )
                if uploaded_files:
                    if st .button("Обработать файлы", type="primary"):
                        progress_tracker = SimpleProgressTracker()
                        progress_tracker .setup_ui()

                        rag_with_progress = RAGPipeline(
                            progress_tracker=progress_tracker)

                        success_count = 0

                        progress_tracker .start_session(
                            len(uploaded_files), "Обработка загруженных файлов")

                        for uploaded_file in uploaded_files:
                            try:
                                if rag_with_progress .load_uploaded_file(uploaded_file):
                                    success_count += 1
                            except Exception as e:
                                progress_tracker .set_error(
                                    f"Ошибка обработки {uploaded_file .name}: {str(e)}")
                                continue

                        progress_tracker .finish_session(success_count > 0)

                        if success_count > 0:
                            st .session_state .rag_pipeline = rag_with_progress

                        if success_count == len(uploaded_files):
                            st .success(
                                f"Успешно обработано {success_count} из {len(uploaded_files)} файлов")
                        elif success_count > 0:
                            st .warning(
                                f"Обработано {success_count} из {len(uploaded_files)} файлов (с ошибками)")
                        else:
                            st .error(f"Не удалось обработать ни одного файла")

            elif upload_method == "Загрузить из папки":
                directory_path = st .text_input(
                    "Путь к папке с документами:",
                    placeholder="./data/documents",
                    help="Папка должна содержать файлы в форматах PDF, DOCX или TXT"
                )
                if directory_path and st .button("Загрузить из папки", type="primary"):
                    try:
                        import os
                        if not os .path .exists(directory_path):
                            st .error(f"Папка не найдена: {directory_path}")
                        elif not os .path .isdir(directory_path):
                            st .error(
                                f"Указанный путь не является папкой: {directory_path}")
                        else:
                            supported_extensions = ['.pdf', '.docx', '.txt']
                            supported_files = [
                                f for f in os .listdir(directory_path)
                                if any(f .lower().endswith(ext)for ext in supported_extensions)
                            ]
                            if not supported_files:
                                st .warning(
                                    f"В папке {directory_path} не найдено поддерживаемых файлов (PDF, DOCX, TXT)")
                            else:
                                progress_tracker = SimpleProgressTracker()
                                progress_tracker .setup_ui()

                                rag_with_progress = RAGPipeline(
                                    progress_tracker=progress_tracker)

                                success = rag_with_progress .load_documents_from_directory(
                                    directory_path)

                                if success:
                                    st .session_state .rag_pipeline = rag_with_progress

                    except Exception as e:
                        st .error(f"Ошибка: {str(e)}")

            else:
                st .info(
                    "Эта опция переиндексирует файлы, уже находящиеся в папке./data/documents")
                docs_dir = "./data/documents"
                if os .path .exists(docs_dir):
                    supported_extensions = ['.pdf', '.docx', '.txt']
                    supported_files = [
                        f for f in os .listdir(docs_dir)
                        if any(f .lower().endswith(ext)for ext in supported_extensions)
                    ]
                    if supported_files:
                        st .write(
                            f"**Найдено {len(supported_files)} файлов:**")
                        for file_name in supported_files:
                            file_path = os .path .join(docs_dir, file_name)
                            file_size = os .path .getsize(
                                file_path)/1024 / 1024
                            file_ext = os .path .splitext(file_name)[1].upper()
                            st .write(
                                f"• {file_name} ({file_size:.1f} MB, {file_ext})")
                        col1, col2 = st .columns(2)
                        with col1:
                            clear_first = st .checkbox(
                                "Очистить векторную базу перед переиндексацией",
                                value=True,
                                help="Рекомендуется для применения новых параметров (overlap=200, улучшенный поиск)"
                            )
                        with col2:
                            if st .button("Переиндексировать", type="primary"):
                                if clear_first:
                                    with st .spinner("Очистка векторной базы..."):
                                        if st .session_state .rag_pipeline .clear_all_data():
                                            st .success(
                                                "Векторная база очищена")
                                        else:
                                            st .error("Ошибка очистки")
                                            st .stop()

                                progress_tracker = SimpleProgressTracker()
                                progress_tracker .setup_ui()

                                rag_with_progress = RAGPipeline(
                                    progress_tracker=progress_tracker)

                                success = rag_with_progress .reindex_existing_documents(
                                    docs_dir)

                                if success:
                                    st .session_state .rag_pipeline = rag_with_progress
                                else:
                                    st .error("Ошибка переиндексации")
                    else:
                        st .warning(
                            "В папке./data/documents нет поддерживаемых файлов (PDF, DOCX, TXT)")
                else:
                    st .error("Папка./data/documents не существует")
        with tab2:
            st .subheader("Загруженные документы")
            document_summary = st .session_state .rag_pipeline .vector_store .get_document_summary()
            if document_summary["total_documents"] > 0:
                col1, col2, col3 = st .columns(3)
                with col1:
                    st .metric("Всего фрагментов",
                               document_summary["total_documents"])
                with col2:
                    st .metric("Загруженных файлов",
                               document_summary["unique_files"])
                with col3:
                    docs_dir = "./data/documents"
                    if os .path .exists(docs_dir):
                        supported_extensions = ['.pdf', '.docx', '.txt']
                        physical_files = len([
                            f for f in os .listdir(docs_dir)
                            if any(f .lower().endswith(ext)for ext in supported_extensions)
                        ])
                        st .metric("Физических файлов", physical_files)
                    else:
                        st .metric("Физических файлов", 0)
                if document_summary["filenames"]:
                    st .write("**Список документов:**")
                    for idx, filename in enumerate(document_summary["filenames"]):
                        file_details = document_summary["file_details"].get(
                            filename, {})
                        chunk_count = file_details .get('chunk_count', 0)
                        page_count = file_details .get('page_count', 'Unknown')
                        expander_title = filename
                        with st .expander(expander_title, expanded=False):
                            col_info, col_preview = st .columns([1, 2])
                            with col_info:
                                st .markdown("**Информация о файле:**")
                                file_ext = os .path .splitext(filename)[
                                    1].lower()

                                st .write(
                                    f"• Фрагментов в базе: {chunk_count}")

                                if file_ext == '.pdf' and page_count != 'N/A':
                                    st .write(f"• Страницы: {page_count}")
                                elif file_ext == '.docx':
                                    st .write(
                                        "• Страницы: определяются при отображении")
                                elif file_ext == '.txt':
                                    st .write(
                                        "• Страницы: непрерывный текст")

                                file_info = st .session_state .rag_pipeline .document_processor .get_physical_file_info(
                                    filename)
                                if file_info .get("exists"):
                                    st .write(
                                        f"• Размер: {file_info .get('size_mb', 0)} MB")

                                upload_date = file_details .get(
                                    'upload_date', 'Unknown')
                                if upload_date == 'Unknown' and file_info .get("exists"):
                                    upload_date = file_info .get(
                                        'modified_time', 'Unknown')
                                st .write(f"• Дата загрузки: {upload_date}")

                                if file_ext == '.docx':
                                    paragraphs_count = file_details .get(
                                        'paragraphs_count', 'Unknown')
                                    tables_count = file_details .get(
                                        'tables_count', 'Unknown')
                                    if paragraphs_count != 'Unknown':
                                        st .write(
                                            f"• Параграфов: {paragraphs_count}")
                                    if tables_count != 'Unknown':
                                        st .write(f"• Таблиц: {tables_count}")
                                elif file_ext == '.txt':
                                    lines_count = file_details .get(
                                        'lines_count', 'Unknown')
                                    characters_count = file_details .get(
                                        'characters_count', 'Unknown')
                                    encoding = file_details .get(
                                        'encoding', 'Unknown')
                                    if lines_count != 'Unknown':
                                        st .write(f"• Строк: {lines_count:,}")
                                    if characters_count != 'Unknown':
                                        st .write(
                                            f"• Символов: {characters_count:,}")
                                    if encoding != 'Unknown':
                                        st .write(f"• Кодировка: {encoding}")
                            with col_preview:
                                st .markdown("**Предпросмотр содержимого:**")
                                with st .spinner("Загрузка предпросмотра..."):
                                    preview = st .session_state .rag_pipeline .vector_store .get_document_preview(
                                        filename, max_length=300)
                                if preview and preview != "Предпросмотр недоступен":
                                    st .text_area(
                                        label="preview",
                                        value=preview,
                                        height=120,
                                        disabled=True,
                                        key=f"preview_{filename}_{idx}",
                                        label_visibility="collapsed"
                                    )
                                    if st .button("Показать весь документ", key=f"full_content_btn_{filename}_{idx}", type="secondary"):
                                        st .session_state .full_content_dialog = {
                                            "show": True,
                                            "filename": filename,
                                            "content_data": None
                                        }
                                        st .rerun()
                                else:
                                    st .warning("Предпросмотр недоступен")
            else:
                st .info(
                    "Документы не загружены. Перейдите на вкладку 'Загрузка' для добавления файлов.")
            if st .session_state .full_content_dialog["show"]:
                filename = st .session_state .full_content_dialog["filename"]
                st .subheader(f"Полное содержимое: {filename}")
                if st .session_state .full_content_dialog["content_data"] is None:
                    with st .spinner("Загрузка полного содержимого документа..."):
                        content_data = st .session_state .rag_pipeline .vector_store .get_full_document_content(
                            filename)
                        st .session_state .full_content_dialog["content_data"] = content_data
                content_data = st .session_state .full_content_dialog["content_data"]
                if content_data and content_data .get("success"):
                    col1, col2, col3, col4 = st .columns(4)
                    with col1:
                        st .metric("Фрагментов", content_data .get(
                            "total_chunks", 0))
                    with col2:
                        page_count = content_data .get("page_count", "Unknown")
                        file_ext = os .path .splitext(filename)[1].lower()
                        if page_count != "N/A":
                            st .metric("Страниц", page_count)
                        elif file_ext == '.docx':
                            st .metric(
                                "Страниц", "без страниц", help="DOCX файлы имеют структуру документа, но не фиксированные страницы")
                        elif file_ext == '.txt':
                            st .metric("Страниц", "непрерывный текст",
                                       help="TXT файлы представляют собой непрерывный поток текста без разделения на страницы")
                        else:
                            st .metric("Страниц", "—")
                    with col3:
                        chars = content_data .get("total_characters", 0)
                        st .metric("Символов", f"{chars:,}")
                    with col4:
                        words = len(content_data .get("content", "").split())
                        st .metric("Слов", f"{words:,}")
                    st .markdown("**Содержимое документа:**")
                    col1, col2, col3 = st .columns([3, 1, 1])
                    with col2:
                        if st .button("Копировать текст", type="secondary", use_container_width=True):
                            st .code(content_data["content"], language=None)
                    with col3:
                        if st .button("Закрыть", use_container_width=True):
                            st .session_state .full_content_dialog = {
                                "show": False, "filename": "", "content_data": None}
                            st .rerun()
                    st .text_area(
                        label="Полное содержимое",
                        value=content_data["content"],
                        height=600,
                        disabled=True,
                        key=f"full_content_{filename}",
                        label_visibility="collapsed"
                    )
                    with st .expander("Детальная информация о фрагментах", expanded=False):
                        st .write(
                            f"**Файл разбит на {content_data['total_chunks']} фрагментов:**")
                        for i, chunk_data in enumerate(content_data .get("chunks", [])[:10]):
                            chunk_text = chunk_data["text"][:200]+"..."if len(
                                chunk_data["text"]) > 200 else chunk_data["text"]
                            st .write(f"**Фрагмент {i + 1}:** {chunk_text}")
                        if len(content_data .get("chunks", [])) > 10:
                            st .write(
                                f"... и еще {len(content_data['chunks'])-10} фрагментов")
                else:
                    st .error(
                        f"Ошибка загрузки содержимого: {content_data .get('error', 'Неизвестная ошибка')}")
                    if st .button("Закрыть", use_container_width=True):
                        st .session_state .full_content_dialog = {
                            "show": False, "filename": "", "content_data": None}
                        st .rerun()
        with tab3:
            st .subheader("Управление документами")
            document_summary = st .session_state .rag_pipeline .vector_store .get_document_summary()
            available_files = document_summary .get("filenames", [])
            if available_files:
                doc_selection = st .radio(
                    "Поиск в документах:",
                    ["Все документы", "Выбранные документы"],
                    key="document_filter_mode_mgmt"
                )
                if doc_selection == "Выбранные документы":
                    selected_files = st .multiselect(
                        "Выберите документы:",
                        options=available_files,
                        default=available_files if st .session_state .selected_documents == "all"else st .session_state .selected_documents,
                        key="selected_files_mgmt"
                    )
                    st .session_state .selected_documents = selected_files if selected_files else "all"
                else:
                    st .session_state .selected_documents = "all"
                if st .session_state .selected_documents == "all":
                    st .info("Поиск во всех документах")
                else:
                    selected_count = len(st .session_state .selected_documents)
                    total_count = len(available_files)
                    st .info(
                        f"Поиск в {selected_count} из {total_count} документов")
                st .write("**Выберите документы для действий:**")
                selected_for_deletion = []
                for filename in available_files:
                    file_details = document_summary["file_details"].get(
                        filename, {})
                    chunk_count = file_details .get('chunk_count', 0)
                    col1, col2, col3 = st .columns([0.1, 0.7, 0.2])
                    with col1:
                        if st .checkbox("", key=f"delete_checkbox_{filename}"):
                            selected_for_deletion .append(filename)
                    with col2:
                        file_ext = os .path .splitext(filename)[1].upper()
                        st .write(f"**{filename}** ({file_ext})")
                        st .caption(f"{chunk_count} фрагментов")
                    with col3:
                        if st .button("Переименовать", help=f"Переименовать {filename}", key=f"rename_btn_{filename}", type="secondary", use_container_width=True):
                            st .session_state .rename_dialog = {
                                "show": True,
                                "filename": filename,
                                "new_name": filename
                            }
                            st .rerun()
                if st .session_state .rename_dialog["show"]:
                    st .subheader(
                        f"Переименование: {st .session_state .rename_dialog['filename']}")
                    col1, col2, col3 = st .columns([2, 1, 1])
                    with col1:
                        new_name = st .text_input(
                            "Новое имя файла:",
                            value=st .session_state .rename_dialog["new_name"],
                            key="rename_input_field"
                        )
                    with col2:
                        if st .button("Сохранить", type="primary", use_container_width=True):
                            if new_name and new_name != st .session_state .rename_dialog["filename"]:
                                old_filename = st .session_state .rename_dialog["filename"]
                                old_extension = os .path .splitext(old_filename)[
                                    1].lower()
                                if not any(new_name .lower().endswith(ext)for ext in ['.pdf', '.docx', '.txt']):
                                    new_name += old_extension
                                success = True
                                file_renamed = st .session_state .rag_pipeline .document_processor .rename_physical_file(
                                    old_filename, new_name)
                                if file_renamed:
                                    metadata_updated = st .session_state .rag_pipeline .vector_store .update_filename_in_metadata(
                                        old_filename, new_name)
                                    success = file_renamed and metadata_updated
                                else:
                                    success = False
                                if success:
                                    st .write(
                                        f"Документ переименован: {old_filename} → {new_name}")
                                    st .session_state .rename_dialog = {
                                        "show": False, "filename": "", "new_name": ""}
                                    st .rerun()
                                else:
                                    st .error("Ошибка переименования")
                            else:
                                st .warning("Введите корректное имя")
                    with col3:
                        if st .button("Отмена", use_container_width=True):
                            st .session_state .rename_dialog = {
                                "show": False, "filename": "", "new_name": ""}
                            st .rerun()
                if selected_for_deletion:
                    st .subheader(
                        f"Удаление документов ({len(selected_for_deletion)} выбрано)")
                    col1, col2 = st .columns(2)
                    with col1:
                        delete_from_vector = st .checkbox(
                            "Удалить из векторной базы",
                            value=True,
                            help="Удалить все фрагменты и метаданные из поиска"
                        )
                    with col2:
                        delete_physical = st .checkbox(
                            "Удалить физические файлы",
                            value=False,
                            help="Удалить файлы с диска безвозвратно"
                        )
                    st .write("**Файлы для удаления:**")
                    for filename in selected_for_deletion:
                        st .write(f"• {filename}")
                    col1, col2 = st .columns(2)
                    with col1:
                        if st .button(f"Удалить ({len(selected_for_deletion)})", type="secondary", use_container_width=True):
                            deleted_count = 0
                            errors = []
                            for filename in selected_for_deletion:
                                try:
                                    success = True
                                    if delete_from_vector:
                                        vector_success = st .session_state .rag_pipeline .vector_store .delete_documents_by_filename(
                                            filename)
                                        success = success and vector_success
                                    if delete_physical:
                                        file_success = st .session_state .rag_pipeline .document_processor .delete_physical_file(
                                            filename)
                                        success = success and file_success
                                    if success:
                                        deleted_count += 1
                                    else:
                                        errors .append(filename)
                                except Exception as e:
                                    errors .append(
                                        f"{filename} (ошибка: {str(e)})")
                            if deleted_count > 0:
                                st .write(
                                    f"Удалено {deleted_count} документов")
                            if errors:
                                st .error(
                                    f"Ошибки при удалении: {','.join(errors)}")
                            if deleted_count > 0:
                                st .rerun()
                    with col2:
                        if st .button("Отменить выбор", use_container_width=True):
                            st .rerun()
            if available_files:
                st .subheader("Управление базой данных")
                col1, col2, col3 = st .columns(3)
                with col1:
                    st .metric("В векторной базе", len(available_files))
                with col2:
                    docs_dir = "./data/documents"
                    physical_count = 0
                    if os .path .exists(docs_dir):
                        supported_extensions = ['.pdf', '.docx', '.txt']
                        physical_count = len([
                            f for f in os .listdir(docs_dir)
                            if any(f .lower().endswith(ext)for ext in supported_extensions)
                        ])
                    st .metric("На диске", physical_count)
                with col3:
                    total_chunks = document_summary .get("total_documents", 0)
                    st .metric("Всего фрагментов", total_chunks)
                col1, col2 = st .columns(2)
                with col1:
                    st .info("Физические файлы остаются в /data/documents/")
                with col2:
                    if st .button("Очистить всю векторную базу", type="secondary", use_container_width=True):
                        if st .session_state .rag_pipeline .clear_all_data():
                            st .write("Векторная база очищена")
                            st .rerun()
                        else:
                            st .error("Ошибка очистки базы")
            else:
                st .info("Нет загруженных документов для управления")
elif page == "Чат":
    st .title("Интерактивный чат")

    status = st .session_state .rag_pipeline .get_system_status()
    total_fragments = status["vector_store"]["total_documents"]
    total_files = status["vector_store"]["unique_files"]

    st .caption(f"{total_files} документов • {total_fragments} фрагментов")

    if not st .session_state .system_initialized:
        st .error("Система не инициализирована. Сначала проверьте систему.")
    elif status["vector_store"]["total_documents"] == 0:
        st .warning(
            "Документы не загружены. Перейдите в раздел 'Документы' для загрузки.")
    else:
        st .markdown(
            """<style>.stChatMessage > div:first-child {display: none!important;}.stJson {max-height: 250px;overflow-y: auto;}</style>""", unsafe_allow_html=True)

        current_messages = st .session_state .chat_sessions[
            st .session_state .current_session]["messages"]

        for message in current_messages:
            with st .chat_message(message["role"]):
                if message["role"] == "user":
                    st .markdown(message["content"])
                else:
                    st .markdown(message["content"])

                    with st .expander("Копировать текст", expanded=False):
                        clean_text = st .session_state .rag_pipeline .markdown_to_text(
                            message["content"])
                        st .text(clean_text)

                    if "metadata" in message and st .session_state .debug_mode:
                        metadata = message["metadata"]
                        debug_info = metadata .get("debug_info", {})
                        with st .expander("Источники", expanded=False):
                            if metadata .get("sources"):
                                for source in metadata["sources"]:
                                    st .caption(
                                        f"• {source['filename']} (релевантность: {source['relevance']:.2f})")
                            else:
                                st .caption(
                                    "Источники недоступны для этого сообщения")

                        with st .expander("Анализ запроса", expanded=False):
                            if debug_info and debug_info .get("query_analysis"):
                                query_analysis = debug_info["query_analysis"]
                                lang = query_analysis .get(
                                    "language", "неизвестен")
                                query_type = query_analysis .get(
                                    "query_type", "неизвестен")
                                keywords = query_analysis .get("keywords", [])

                                st .caption(f"**Язык:** {lang}")
                                st .caption(f"**Тип запроса:** {query_type}")
                                if keywords:
                                    st .caption(
                                        f"**Ключевые слова:** {','.join(keywords)}")
                            else:
                                st .caption(
                                    "Анализ запроса недоступен для этого сообщения")

                        with st .expander("Производительность", expanded=False):
                            time_resp = metadata .get("response_time", 0)
                            type_resp = metadata .get(
                                "response_type", "unknown")
                            st .caption(
                                f"**Время и тип ответа:** {time_resp:.2f}с, тип: {type_resp}")

                            if debug_info and debug_info .get("search_results"):
                                search_results = debug_info["search_results"]
                                sources_count = len(
                                    metadata .get("sources", []))
                                st .caption(
                                    f"**Результаты поиска:** найдено: {len(search_results)}, использовано: {sources_count}")
                            else:
                                st .caption(
                                    "Детали поиска недоступны для этого сообщения")

                        with st .expander("Отладка", expanded=False):
                            if debug_info:
                                st .json(debug_info)
                            else:
                                st .caption(
                                    "Отладочная информация недоступна для этого сообщения")

        prompt = st .chat_input("Введите ваш вопрос...")

        if prompt:
            user_message = {"role": "user", "content": prompt}
            st .session_state .chat_sessions[st .session_state .current_session]["messages"].append(
                user_message)

            st .session_state .session_manager .auto_save_session()

            with st .chat_message("user"):
                st .markdown(prompt)

            with st .chat_message("assistant"):
                with st .spinner("Думаю..."):
                    response = st .session_state .rag_pipeline .process_query(
                        prompt,
                        show_debug=st .session_state .debug_mode,
                        selected_documents=st .session_state .selected_documents,
                        search_k=st .session_state .session_manager .get_setting(
                            'search_k', 10),
                        search_method=st .session_state .session_manager .get_setting(
                            'search_method', 'mmr'),
                        distance_threshold=st .session_state .session_manager .get_setting(
                            'distance_threshold', 0.25),
                        confidence_threshold=st .session_state .session_manager .get_setting(
                            'confidence_threshold', 0.5),
                        temperature=st .session_state .session_manager .get_setting(
                            'temperature', 0.2),
                        max_tokens=st .session_state .session_manager .get_setting(
                            'max_tokens', 2000),
                        system_prompt_style=st .session_state .session_manager .get_setting(
                            'system_prompt_style', 'Профессиональный')
                    )

                full_response = response["answer"]

                def stream_response():
                    for char in full_response:
                        yield char
                        time .sleep(0.02)

                st .write_stream(stream_response())

                with st .expander("Копировать текст", expanded=False):
                    clean_text = st .session_state .rag_pipeline .markdown_to_text(
                        full_response)
                    st .text(clean_text)

                metadata = {
                    "response_time": response .get("response_time", 0),
                    "response_type": response .get("response_type", "unknown"),
                    "sources": response .get("sources", []),
                }

                if st .session_state .debug_mode:
                    metadata["debug_info"] = response

                    with st .expander("Обработка запроса", expanded=False):

                        query_analysis = response .get("query_analysis", {})
                        if query_analysis:
                            st .caption("**Предобработка запроса:**")
                            st .caption(
                                f"• Исходный запрос: {query_analysis .get('original_query', prompt)}")
                            if query_analysis .get('corrected_query') != query_analysis .get('original_query'):
                                st .caption(
                                    f"• Исправленный запрос: {query_analysis .get('corrected_query', 'нет изменений')}")
                            if query_analysis .get('expanded_queries'):
                                st .caption(
                                    f"• Расширенные запросы: {len(query_analysis .get('expanded_queries', []))} вариантов")
                            st .caption(
                                f"• Язык: {query_analysis .get('language', 'неизвестен')}")
                            st .caption(
                                f"• Тип запроса: {query_analysis .get('query_type', 'неизвестен')}")
                            if query_analysis .get('keywords'):
                                st .caption(
                                    f"• Ключевые слова: {','.join(query_analysis .get('keywords', []))}")

                        routing_result = response .get("routing_result", {})
                        if routing_result:
                            st .caption("")
                            st .caption("**Решение маршрутизатора:**")
                            can_answer = routing_result .get(
                                "can_answer", False)
                            confidence = routing_result .get("confidence", 0)
                            num_sources = routing_result .get("num_sources", 0)

                            st .caption(
                                f"• Может ответить: {'Да'if can_answer else 'Нет'}")
                            st .caption(f"• Уверенность: {confidence:.3f}")
                            st .caption(
                                f"• Порог уверенности: {st .session_state .session_manager .get_setting('confidence_threshold', 0.5)}")
                            st .caption(
                                f"• Количество источников: {num_sources}")
                            st .caption(
                                f"• Обоснование: {routing_result .get('reasoning', 'не указано')}")

                    with st .expander("Результаты поиска", expanded=False):

                        if response .get("sources"):
                            st .caption("**Использованные источники:**")
                            for source in response["sources"]:
                                st .caption(
                                    f"• {source['filename']} (релевантность: {source['relevance']:.3f})")

                        if response .get("search_results"):
                            pages_info = {}
                            fragment_counts = {}
                            distances = []

                            for result in response["search_results"]:
                                filename = result .get("metadata", {}).get(
                                    "filename", "Unknown")
                                page = result .get("metadata", {}).get(
                                    "source_page", "Unknown")
                                distance = result .get("distance", 1.0)
                                distances .append(distance)

                                if filename not in pages_info:
                                    pages_info[filename] = set()
                                    fragment_counts[filename] = 0

                                if page != "Unknown":
                                    pages_info[filename].add(str(page))
                                fragment_counts[filename] += 1

                            if pages_info:
                                st .caption("")
                                st .caption("**Детали по документам:**")
                                for filename, pages in pages_info .items():
                                    pages_str = ",".join(
                                        sorted(pages))if pages else "не указаны"
                                    fragments = fragment_counts .get(
                                        filename, 0)
                                    st .caption(
                                        f"• {filename}: стр. {pages_str} ({fragments} фрагментов)")

                            if distances:
                                st .caption(
                                    f"• Расстояния - мин: {min(distances):.3f}, средн: {sum(distances)/len(distances):.3f}")

                        context_relevance = response .get("context_relevance")
                        confidence_assessment = response .get(
                            "confidence_assessment")
                        if context_relevance or confidence_assessment:
                            st .caption("")
                            st .caption("**Оценки качества (2024):**")

                        if context_relevance:
                            st .caption(
                                f"• Релевантность контекста: {context_relevance .get('overall_score', 'н/д')}")
                            st .caption(
                                f"• Полезных/всего предложений: {context_relevance .get('useful_sentences', 'н/д')}/{context_relevance .get('total_sentences', 'н/д')}")

                        if confidence_assessment:
                            st .caption(
                                f"• Уверенность в ответе: {confidence_assessment .get('confidence_score', 'н/д')}")
                            st .caption(
                                f"• Обоснование: {confidence_assessment .get('reasoning', 'н/д')}")
                            uncertainties = confidence_assessment .get(
                                'uncertainty_indicators', [])
                            if uncertainties:
                                st .caption(
                                    f"• Индикаторы неуверенности: {','.join(uncertainties)}")

                    with st .expander("Системная информация", expanded=False):

                        time_resp = response .get("response_time", 0)
                        type_resp = response .get("response_type", "unknown")
                        search_results = response .get("search_results", [])
                        sources_count = len(response .get("sources", []))

                        st .caption("**Производительность:**")
                        st .caption(f"• Время ответа: {time_resp:.3f}с")
                        st .caption(f"• Тип ответа: {type_resp}")
                        st .caption(
                            f"• Поиск: найдено {len(search_results)}, использовано {sources_count}")

                        search_settings = {
                            'search_method': st .session_state .session_manager .get_setting('search_method', 'mmr'),
                            'search_k': st .session_state .session_manager .get_setting('search_k', 10),
                            'distance_threshold': st .session_state .session_manager .get_setting('distance_threshold', 0.5),
                            'temperature': st .session_state .session_manager .get_setting('temperature', 0.5),
                            'max_tokens': st .session_state .session_manager .get_setting('max_tokens', 2000)
                        }
                        st .caption("")
                        st .caption("**Настройки системы:**")
                        for key, value in search_settings .items():
                            st .caption(f"• {key}: {value}")

                        st .caption("")
                        st .caption("**Полные технические данные:**")
                        st .json(response)

            assistant_message = {
                "role": "assistant",
                "content": full_response,
                "metadata": metadata
            }
            st .session_state .chat_sessions[st .session_state .current_session]["messages"].append(
                assistant_message)

            st .session_state .session_manager .auto_save_session()

            conversation_entry = {
                "timestamp": datetime .now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": prompt,
                "answer": full_response,
                "response_type": response .get("response_type", "unknown"),
                "response_time": response .get("response_time", 0),
                "sources": response .get("sources", [])
            }
            st .session_state .conversation_history .insert(
                0, conversation_entry)
elif page == "Настройки":
    st .title("Настройки системы")
    if not st .session_state .system_initialized:
        st .error("Система не инициализирована.")
    else:
        tab1, tab2 = st .tabs(["Система", "Параметры"])
        with tab1:
            status = st .session_state .rag_pipeline .get_system_status()
            current_config = st .session_state .rag_pipeline .config_manager .get_current_config()
            available_models = st .session_state .rag_pipeline .config_manager .get_available_models()

            st .markdown("### Выбор моделей")
            available_models = st .session_state .rag_pipeline .config_manager .get_available_models()
            current_config = st .session_state .rag_pipeline .config_manager .get_current_config()
            col1, col2 = st .columns(2)
            with col1:
                st .markdown("#### Модель LLM")
                if available_models['llm']:
                    current_llm_index = 0
                    if current_config .llm_model in available_models['llm']:
                        current_llm_index = available_models['llm'].index(
                            current_config .llm_model)
                    selected_llm = st .selectbox(
                        "Выберите LLM модель",
                        available_models['llm'],
                        index=current_llm_index,
                        key="llm_model_select"
                    )
                    if selected_llm != current_config .llm_model:
                        if st .button("Применить LLM модель", key="apply_llm"):
                            if st .session_state .rag_pipeline .update_models(llm_model=selected_llm):
                                st .rerun()
                else:
                    st .warning("Не найдено доступных LLM моделей")
                    st .info("Установите LLM модель: ollama pull <model_name>")
            with col2:
                st .markdown("#### Модель Embedding")
                if available_models['embedding']:
                    current_embed_index = 0
                    if current_config .embedding_model in available_models['embedding']:
                        current_embed_index = available_models['embedding'].index(
                            current_config .embedding_model)
                    selected_embedding = st .selectbox(
                        "Выберите Embedding модель",
                        available_models['embedding'],
                        index=current_embed_index,
                        key="embedding_model_select"
                    )
                    if selected_embedding != current_config .embedding_model:
                        if st .button("Применить Embedding модель", key="apply_embedding"):
                            if st .session_state .rag_pipeline .update_models(embedding_model=selected_embedding):
                                st .rerun()
                else:
                    st .warning("Не найдено доступных Embedding моделей")
                    st .info(
                        "Установите Embedding модель: ollama pull <model_name>")
            st .markdown("#### Текущие модели")
            col1, col2 = st .columns(2)
            with col1:
                st .info(f"**LLM:** {current_config .llm_model}")
            with col2:
                st .info(f"**Embedding:** {current_config .embedding_model}")
            if st .button("Обновить список моделей", key="refresh_models"):
                st .success("Список моделей обновлен")
                time .sleep(1)
                st .rerun()

            st .markdown("### Системные промпты")
            system_prompt_styles = st .session_state .session_manager .get_system_prompts()
            current_style = st .session_state .session_manager .get_setting(
                'system_prompt_style', 'Профессиональный')

            style_options = {
                "Профессиональный": "Профессиональный",
                "Академический": "Академический",
                "Дружелюбный": "Дружелюбный",
                "Краткий": "Краткий",
                "Подробный": "Подробный",
                "Формальный": "Формальный",
                "Объясняющий": "Объясняющий",
                "Технический": "Технический",
                "Креативный": "Креативный"
            }

            reverse_mapping = {v: k for k, v in style_options .items()}
            current_display = reverse_mapping .get(
                current_style, "Профессиональный")
            current_index = list(style_options .keys()).index(current_display)

            selected_display = st .selectbox(
                "Выберите стиль общения",
                options=list(style_options .keys()),
                index=current_index,
                help="Стиль влияет на манеру ответов ассистента"
            )

            selected_style = style_options[selected_display]
            if selected_style != current_style:
                st .session_state .session_manager .set_setting(
                    'system_prompt_style', selected_style)
                st .success(f"Стиль обновлен: {selected_style}")

            st .info(
                f"**{selected_style}**: {system_prompt_styles[selected_style]}")

            st .markdown("### Добавить новые модели")
            st .markdown("Для добавления новых моделей используйте:")
            st .code("ollama pull <model_name>", language="bash")

            llm = [
                "**llama3.1** — языковая модель от Meta (8B, 70B, 405B)",
                "**deepseek-r1** — рассуждающая модель от DeepSeek (1.5B, 7B, 8B, 14B, 32B, 70B, 671B)",
                "**mistral** — компактная языковая модель от Mistral AI (7B)",
                "**qwen2.5** — серия языковых моделей от Alibaba (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B)",
                "**gemma3** — модель Gemma от Google DeepMind (1B, 4B, 12B, 27B)",
                "**llava** — мультимодальная модель Vicuna + CLIP (7B, 13B, 34B)",
                "**phi3** — лёгкая языковая модель от Microsoft (3.8B, 14B)",
                "**llama2** — предыдущая линейка от Meta (7B, 13B, 70B)",
                "**minicpm-v** — визуально-языковая модель (8B)",
                "**codellama** — генерация и обсуждение кода (7B, 13B, 34B, 70B)",
                "**tinyllama** — сверхкомпактная языковая модель (1.1B)",
                "**dolphin3** — инструкционная модель на базе Llama 3 (8B)",
                "**olmo2** — открытая языковая модель от AllenAI (7B, 13B)",
                "**qwq** — рассуждающая модель серии Qwen (32B)",
                "**smollm2** — компактная серия SmolLM (135M, 360M, 1.7B)",
                "**mixtral** — экспертная mixture-of-experts модель от Mistral AI (8×7B, 8×22B)",
                "**starcoder2** — кодовая модель BigCode (3B, 7B, 15B)",
                "**openthinker** — рассуждающие модели Distilled DeepSeek (7B, 32B)",
                "**llama4** — мультимодальная линейка от Meta (16×17B, 128×17B)",
            ]

            emb = [
                "**nomic-embed-text** — открытая embedding-модель от Nomic (567M)",
                "**mxbai-embed-large** — высокоточная embedding-модель от Mixedbread.ai (335M)",
                "**bge-m3** — многофункциональная embedding-модель BGE-M3 (567M)",
                "**all-minilm** — семейство MiniLM sentence embeddings (22M, 33M)",
                "**snowflake-arctic-embed** — модель Arctic Embed от Snowflake (22M, 33M, 110M, 137M, 335M)",
                "**bge-large** — крупная embedding-модель BGE-Large (335M)",
                "**snowflake-arctic-embed2** — модель Arctic Embed 2.0 от Snowflake (568M)",
                "**paraphrase-multilingual** — мультиязычная paraphrase-модель (278M)",
                "**granite-embedding** — модель Granite Embedding от IBM (30M, 278M)",
            ]

            st .markdown("### Популярные модели")
            tab_llm, tab_emb = st .tabs(["LLM модели", "Embedding модели"])

            with tab_llm:
                c1, c2 = st .columns(2)
                for i, m in enumerate(llm):
                    (c1 if i % 2 == 0 else c2).markdown(f"- {m}")

            with tab_emb:
                for m in emb:
                    st .markdown(f"- {m}")

            st .markdown("### Экспорт данных")
            if st .button("Экспортировать историю", help="Скачать все разговоры в текстовом формате"):
                if st .session_state .conversation_history:
                    export_text = st .session_state .rag_pipeline .export_conversation(
                        st .session_state .conversation_history
                    )
                    st .download_button(
                        label="Скачать историю",
                        data=export_text,
                        file_name=f"rag_history_{datetime .now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st .info("История разговоров пуста")

            st .markdown("### Системные действия")
            col1, col2 = st .columns(2)
            with col1:
                if st .button("Переинициализировать систему"):
                    st .session_state .system_initialized = False
                    st .session_state .rag_pipeline = RAGPipeline()
                    st .success("Система переинициализирована")
                    time .sleep(1)
                    st .rerun()
            with col2:
                if st .button("Сбросить все настройки"):
                    st .session_state .clear()
                    st .success("Настройки сброшены")
                    time .sleep(1)
                    st .rerun()

        with tab2:
            st .markdown("### Обработка документов")
            col1, col2 = st .columns(2)

            with col1:
                chunk_size = st .number_input(
                    "Размер фрагмента текста",
                    min_value=256,
                    max_value=1024,
                    value=st .session_state .session_manager .get_setting(
                        'chunk_size', 512),
                    step=50,
                    help="Размер каждого фрагмента документа в символах"
                )
                if chunk_size != st .session_state .session_manager .get_setting('chunk_size', 512):
                    st .session_state .session_manager .set_setting(
                        'chunk_size', chunk_size)
                    st .success(f"Размер фрагмента обновлен: {chunk_size}")

                st .caption(
                    "Маленькие фрагменты: более точный поиск, но меньше контекста")
                st .caption(
                    "Большие фрагменты: больше контекста, но менее точный поиск")

            with col2:
                chunk_overlap = st .number_input(
                    "Перекрытие фрагментов (%)",
                    min_value=5,
                    max_value=50,
                    value=st .session_state .session_manager .get_setting(
                        'chunk_overlap', 25),
                    step=5,
                    help="Процент перекрытия между фрагментами"
                )
                if chunk_overlap != st .session_state .session_manager .get_setting('chunk_overlap', 25):
                    st .session_state .session_manager .set_setting(
                        'chunk_overlap', chunk_overlap)
                    st .success(
                        f"Перекрытие фрагментов обновлено: {chunk_overlap}%")

                st .caption(
                    "Малое перекрытие: экономит память, но может терять связи")
                st .caption(
                    "Большое перекрытие: лучше связывает информацию между частями")

            st .markdown("### Поиск информации")
            col1, col2 = st .columns(2)

            with col1:
                search_method = st .selectbox(
                    "Метод поиска",
                    options=["similarity", "mmr", "threshold"],
                    index=["similarity", "mmr", "threshold"].index(
                        st .session_state .session_manager .get_setting('search_method', 'mmr')),
                    help="Алгоритм для выбора релевантных документов"
                )
                if search_method != st .session_state .session_manager .get_setting('search_method', 'mmr'):
                    st .session_state .session_manager .set_setting(
                        'search_method', search_method)
                    st .success(f"Метод поиска обновлен: {search_method}")

                st .caption(
                    "**similarity**: простое сходство по релевантности")
                st .caption(
                    "**mmr**: диверсификация результатов (рекомендуется)")
                st .caption("**threshold**: строгая фильтрация по порогу")

            with col2:
                search_k = st .number_input(
                    "Количество результатов поиска",
                    min_value=5,
                    max_value=15,
                    value=st .session_state .session_manager .get_setting(
                        'search_k', 10),
                    help="Количество наиболее релевантных фрагментов для поиска"
                )
                if search_k != st .session_state .session_manager .get_setting('search_k', 10):
                    st .session_state .session_manager .set_setting(
                        'search_k', search_k)
                    st .success(
                        f"Количество результатов поиска обновлено: {search_k}")

                st .caption(
                    "Мало результатов: быстрая работа, но может пропустить важное")
                st .caption(
                    "Много результатов: полнота поиска, но медленнее и больше шума")

            st .markdown("### Контроль релевантности")
            col1, col2 = st .columns(2)

            with col1:
                distance_threshold = st .slider(
                    "Порог расстояния для поиска",
                    min_value=0.1,
                    max_value=0.9,
                    value=st .session_state .session_manager .get_setting(
                        'distance_threshold', 0.5),
                    step=0.05,
                    help="Максимальное расстояние для включения результата в поиск"
                )
                if abs(distance_threshold - st .session_state .session_manager .get_setting('distance_threshold', 0.5)) > 0.01:
                    st .session_state .session_manager .set_setting(
                        'distance_threshold', distance_threshold)
                    st .success(
                        f"Порог расстояния обновлен: {distance_threshold}")

                st .caption("Низкий порог: только очень похожие результаты")
                st .caption(
                    "Высокий порог: больше результатов, но менее точных")

            with col2:
                confidence_threshold = st .slider(
                    "Порог уверенности для ответов",
                    min_value=0.1,
                    max_value=0.9,
                    value=st .session_state .session_manager .get_setting(
                        'confidence_threshold', 0.5),
                    step=0.05,
                    help="Минимальная уверенность для полного ответа"
                )
                if abs(confidence_threshold - st .session_state .session_manager .get_setting('confidence_threshold', 0.5)) > 0.01:
                    st .session_state .session_manager .set_setting(
                        'confidence_threshold', confidence_threshold)
                    st .success(
                        f"Порог уверенности обновлен: {confidence_threshold}")

                st .caption(
                    "Низкий порог: система отвечает чаще, но может ошибаться")
                st .caption("Высокий порог: система отвечает реже, но точнее")

            st .markdown("### Настройки генерации")
            col1, col2 = st .columns(2)

            with col1:
                temperature = st .slider(
                    "Температура генерации",
                    min_value=0.0,
                    max_value=1.0,
                    value=st .session_state .session_manager .get_setting(
                        'temperature', 0.5),
                    step=0.1,
                    help="Более высокая температура = более креативные ответы"
                )
                if abs(temperature - st .session_state .session_manager .get_setting('temperature', 0.5)) > 0.01:
                    st .session_state .session_manager .set_setting(
                        'temperature', temperature)
                    st .success(
                        f"Температура генерации обновлена: {temperature}")

                st .caption("Низкая температура: точные и одинаковые ответы")
                st .caption(
                    "Высокая температура: креативные, но непредсказуемые ответы")

            with col2:
                max_tokens = st .number_input(
                    "Максимальное количество токенов",
                    min_value=200,
                    max_value=4000,
                    value=st .session_state .session_manager .get_setting(
                        'max_tokens', 2000),
                    step=100,
                    help="Максимальная длина генерируемого ответа"
                )
                if max_tokens != st .session_state .session_manager .get_setting('max_tokens', 2000):
                    st .session_state .session_manager .set_setting(
                        'max_tokens', max_tokens)
                    st .success(f"Максимальные токены обновлены: {max_tokens}")

                st .caption("Малое значение: короткие ответы")
                st .caption("Большое значение: более развернутые ответы")
