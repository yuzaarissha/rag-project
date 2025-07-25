import streamlit as st
import os
import time
import uuid
from datetime import datetime, timedelta
from src.main import RAGPipeline
import json
st.set_page_config(
    page_title="RAG System",
    page_icon="🧠",   
    layout="wide",
    initial_sidebar_state="expanded"
)
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = False
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "selected_documents" not in st.session_state:
    st.session_state.selected_documents = "all"
if "full_content_dialog" not in st.session_state:
    st.session_state.full_content_dialog = {"show": False, "filename": "", "content_data": None}
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"default": {"name": "Новый чат", "messages": [], "created_at": datetime.now()}}
if "current_session" not in st.session_state:
    st.session_state.current_session = "default"
if "show_rename_dialog" not in st.session_state:
    st.session_state.show_rename_dialog = None
if "show_dropdown_menu" not in st.session_state:
    st.session_state.show_dropdown_menu = None
if "documents_to_delete" not in st.session_state:
    st.session_state.documents_to_delete = []
if "rename_dialog" not in st.session_state:
    st.session_state.rename_dialog = {"show": False, "filename": "", "new_name": ""}
st.sidebar.title("RAG System")
page = st.sidebar.selectbox(
    "Выберите страницу",
    ["Главная", "Чат", "Документы", "Настройки"]
)
if not st.session_state.system_initialized:
    try:
        with st.spinner("Инициализация системы..."):
            st.session_state.system_initialized = st.session_state.rag_pipeline.initialize_system(silent=True)
    except Exception as e:
        with st.sidebar:
            st.subheader("Инициализация системы")
            st.error(f"Автоматическая инициализация не удалась: {str(e)}")
            if st.button("Повторить инициализацию", type="primary"):
                st.session_state.system_initialized = st.session_state.rag_pipeline.initialize_system()
if st.session_state.system_initialized:
    with st.sidebar:
        if st.button("Переинициализировать", help="Повторная проверка системы"):
            st.session_state.system_initialized = st.session_state.rag_pipeline.initialize_system()
st.session_state.debug_mode = st.sidebar.checkbox("Режим отладки", value=st.session_state.debug_mode)
if page == "Чат":
    st.sidebar.markdown("""
    <style>
    div[data-testid="stSidebar"] .stButton > button {
        background: transparent;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: rgba(255, 255, 255, 0.9);
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: normal;
        transition: all 0.2s ease;
        min-height: 36px;
    }
    
    div[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.3);
        color: white;
    }
    
    div[data-testid="stSidebar"] .stSelectbox > div > div {
        background: transparent;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 6px;
        color: rgba(255, 255, 255, 0.9);
        min-height: 36px;
    }
    
    div[data-testid="stSidebar"] .stSelectbox > div > div > div {
        color: rgba(255, 255, 255, 0.9);
        padding: 8px 12px;
        font-size: 14px;
    }
    
    div[data-testid="stSidebar"] .stTextInput > div > div > input {
        background: transparent;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: rgba(255, 255, 255, 0.9);
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 14px;
        min-height: 36px;
    }
    
    div[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
        border-color: rgba(255, 255, 255, 0.4);
        outline: none;
        background: rgba(255, 255, 255, 0.05);
    }
    
    div[data-testid="stSidebar"] .stSelectbox svg {
        display: none;
    }
    
    div[data-testid="stSidebar"] .stSelectbox > div > div::after {
        content: "⋯";
        position: absolute;
        right: 12px;
        top: 50%;
        transform: translateY(-50%);
        color: rgba(255, 255, 255, 0.5);
        font-size: 16px;
        pointer-events: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    
    all_chats = []
    chat_options = {}
    
    for session_id, session_data in st.session_state.chat_sessions.items():
        display_name = session_data["name"]
        if session_data["messages"] and display_name == "Новый чат":
            first_user_msg = next((msg for msg in session_data["messages"] if msg["role"] == "user"), None)
            if first_user_msg:
                display_name = first_user_msg["content"][:50]
                if len(first_user_msg["content"]) > 50:
                    display_name += "..."
        
        all_chats.append(display_name)
        chat_options[display_name] = session_id
    
    current_chat_name = None
    for name, sid in chat_options.items():
        if sid == st.session_state.current_session:
            current_chat_name = name
            break
    
    current_index = all_chats.index(current_chat_name) if current_chat_name in all_chats else 0
    
    if st.session_state.show_rename_dialog:
        st.sidebar.markdown('<div class="rename-mode">', unsafe_allow_html=True)
        current_name = st.session_state.chat_sessions[st.session_state.show_rename_dialog]["name"]
        new_name = st.sidebar.text_input(
            "Переименовать чат:",
            value=current_name,
            key="rename_input",
            label_visibility="collapsed",
            placeholder="Введите новое название..."
        )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Сохранить", key="save_rename", use_container_width=True):
                if new_name.strip():
                    st.session_state.chat_sessions[st.session_state.show_rename_dialog]["name"] = new_name.strip()
                st.session_state.show_rename_dialog = None
                st.rerun()
        with col2:
            if st.button("Отмена", key="cancel_rename", use_container_width=True):
                st.session_state.show_rename_dialog = None
                st.rerun()
    else:
        selected_chat = st.sidebar.selectbox(
            "Выберите чат",
            all_chats,
            index=current_index,
            key="chat_selector"
        )
        
        if selected_chat and chat_options[selected_chat] != st.session_state.current_session:
            st.session_state.current_session = chat_options[selected_chat]
            st.rerun()
        
        if st.sidebar.button("Переименовать", key="rename_current", use_container_width=True):
            st.session_state.show_rename_dialog = st.session_state.current_session
            st.rerun()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Новый", key="new_chat_btn", use_container_width=True):
                new_session_id = str(uuid.uuid4())[:8]
                st.session_state.chat_sessions[new_session_id] = {
                    "name": "Новый чат",
                    "messages": [],
                    "created_at": datetime.now()
                }
                st.session_state.current_session = new_session_id
                st.rerun()
        with col2:
            if st.button("Удалить", key="delete_current", use_container_width=True):
                if len(st.session_state.chat_sessions) > 1:
                    del st.session_state.chat_sessions[st.session_state.current_session]
                    st.session_state.current_session = list(st.session_state.chat_sessions.keys())[0]
                    st.rerun()
    
    
if st.session_state.system_initialized:
    status = st.session_state.rag_pipeline.get_system_status()
    st.sidebar.subheader("Статус системы")
    st.sidebar.metric("Фрагментов", status["vector_store"]["total_documents"])
    st.sidebar.metric("Загруженных файлов", status["vector_store"]["unique_files"])
    st.sidebar.metric("Запросов", status["pipeline_stats"]["total_queries"])
    if st.session_state.selected_documents == "all":
        st.sidebar.write("**Активных:** Все документы")
    else:
        active_count = len(st.session_state.selected_documents) if isinstance(st.session_state.selected_documents, list) else 0
        total_count = status["vector_store"]["unique_files"]
        st.sidebar.write(f"**Активных:** {active_count} из {total_count}")
if page == "Главная":
    st.title("Главная страница")
    st.markdown("""
    **Возможности системы:**
    - **Загрузка PDF документов** - поддержка больших файлов
    - **Умная маршрутизация** - определение релевантности запросов
    - **Многоязычность** - поддержка русского и казахского языков
    - **Локальная обработка** - работа без интернета через Ollama
    
    **Технологический стек:**
    - **LLM**: Автоматический выбор из доступных моделей Ollama
    - **Embeddings**: Автоматический выбор из доступных моделей Ollama
    - **Vector DB**: ChromaDB
    - **Interface**: Streamlit
    """)
    if not st.session_state.system_initialized:
        st.warning("Система не инициализирована. Проверьте систему в боковой панели.")
    else:
        config = st.session_state.rag_pipeline.config_manager.get_current_config()
        st.info(f"**Используемые модели:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**LLM:** {config.llm_model}")
        with col2:
            st.write(f"**Embedding:** {config.embedding_model}")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Быстрый старт")
            st.markdown("""
            1. Перейдите в **Управление документами**
            2. Загрузите PDF файлы
            3. Перейдите в **Чат** для вопросов
            """)
        with col2:
            st.subheader("Советы")
            st.markdown("""
            - Задавайте конкретные вопросы
            - Используйте ключевые слова из документов
            - Включите режим отладки для детальной информации
            """)
elif page == "Документы":
    st.title("Управление документами")
    if not st.session_state.system_initialized:
        st.error("Система не инициализирована. Сначала проверьте систему.")
    else:
        tab1, tab2, tab3 = st.tabs(["Загрузка", "Просмотр", "Управление"])
        with tab1:
            st.subheader("Загрузка новых документов")
            upload_method = st.radio(
                "Способ загрузки:",
                ["Загрузить файлы", "Загрузить из папки", "Переиндексировать существующие"]
            )
            if upload_method == "Загрузить файлы":
                uploaded_files = st.file_uploader(
                    "Выберите PDF файлы",
                    type="pdf",
                    accept_multiple_files=True
                )
                if uploaded_files:
                    if st.button("Обработать файлы", type="primary"):
                        success_count = 0
                        for uploaded_file in uploaded_files:
                            if st.session_state.rag_pipeline.load_uploaded_file(uploaded_file):
                                success_count += 1
                        st.write(f"Обработано {success_count} из {len(uploaded_files)} файлов")
            elif upload_method == "Загрузить из папки":
                directory_path = st.text_input(
                    "Путь к папке с PDF файлами:",
                    placeholder="./data/documents"
                )
                if directory_path and st.button("Загрузить из папки", type="primary"):
                    st.session_state.rag_pipeline.load_documents_from_directory(directory_path)
            else:
                st.info("Эта опция переиндексирует файлы, уже находящиеся в папке ./data/documents БЕЗ создания дубликатов")
                docs_dir = "./data/documents"
                if os.path.exists(docs_dir):
                    pdf_files = [f for f in os.listdir(docs_dir) if f.endswith('.pdf')]
                    if pdf_files:
                        st.write(f"**Найдено {len(pdf_files)} PDF файлов:**")
                        for pdf_file in pdf_files:
                            file_path = os.path.join(docs_dir, pdf_file)
                            file_size = os.path.getsize(file_path) / 1024 / 1024
                            st.write(f"• {pdf_file} ({file_size:.1f} MB)")
                        col1, col2 = st.columns(2)
                        with col1:
                            clear_first = st.checkbox(
                                "Очистить векторную базу перед переиндексацией", 
                                value=True,
                                help="Рекомендуется для применения новых параметров (overlap=200, улучшенный поиск)"
                            )
                        with col2:
                            if st.button("Переиндексировать", type="primary"):
                                if clear_first:
                                    st.info("Очистка векторной базы...")
                                    if st.session_state.rag_pipeline.clear_all_data():
                                        st.write("Векторная база очищена")
                                    else:
                                        st.error("Ошибка очистки")
                                        st.stop()
                                st.info(f"Переиндексация {len(pdf_files)} файлов...")
                                success = st.session_state.rag_pipeline.reindex_existing_documents(docs_dir)
                                if success:
                                    st.write("Переиндексация завершена")
                                else:
                                    st.error("Ошибка переиндексации")
                    else:
                        st.warning("В папке ./data/documents нет PDF файлов")
                else:
                    st.error("Папка ./data/documents не существует")
        with tab2:
            st.subheader("Загруженные документы")
            document_summary = st.session_state.rag_pipeline.vector_store.get_document_summary()
            if document_summary["total_documents"] > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Всего фрагментов", document_summary["total_documents"])
                with col2:
                    st.metric("Загруженных файлов", document_summary["unique_files"])
                with col3:
                    docs_dir = "./data/documents"
                    if os.path.exists(docs_dir):
                        physical_files = len([f for f in os.listdir(docs_dir) if f.endswith('.pdf')])
                        st.metric("Физических файлов", physical_files)
                    else:
                        st.metric("Физических файлов", 0)
                st.markdown("---")
                if document_summary["filenames"]:
                    st.write("**Список документов:**")
                    for idx, filename in enumerate(document_summary["filenames"]):
                        file_details = document_summary["file_details"].get(filename, {})
                        chunk_count = file_details.get('chunk_count', 0)
                        page_count = file_details.get('page_count', 'Unknown')
                        with st.expander(f"{filename} ({chunk_count} фрагментов, {page_count} страниц)", expanded=False):
                            col_info, col_preview = st.columns([1, 2])
                            with col_info:
                                st.markdown("**Информация о файле:**")
                                st.write(f"• Фрагментов в базе: {chunk_count}")
                                st.write(f"• Страниц в PDF: {page_count}")
                                st.write(f"• Оригинальное имя: {file_details.get('original_name', filename)}")
                                file_info = st.session_state.rag_pipeline.document_processor.get_physical_file_info(filename)
                                if file_info.get("exists"):
                                    st.write(f"• Размер файла: {file_info.get('size_mb', 0)} MB")
                                    st.write(f"• Дата изменения: {file_info.get('modified_time', 'Unknown')}")
                                    st.write("Физический файл найден")
                                else:
                                    st.error("Физический файл не найден на диске")
                            with col_preview:
                                st.markdown("**Предпросмотр содержимого:**")
                                with st.spinner("Загрузка предпросмотра..."):
                                    preview = st.session_state.rag_pipeline.vector_store.get_document_preview(filename, max_length=300)
                                if preview and preview != "Предпросмотр недоступен":
                                    st.text_area(
                                        label="preview", 
                                        value=preview,
                                        height=120,
                                        disabled=True,
                                        key=f"preview_{filename}_{idx}",
                                        label_visibility="collapsed"
                                    )
                                    if st.button("Показать весь документ", key=f"full_content_btn_{filename}_{idx}", type="secondary"):
                                        st.session_state.full_content_dialog = {
                                            "show": True,
                                            "filename": filename,
                                            "content_data": None
                                        }
                                        st.rerun()
                                else:
                                    st.warning("Предпросмотр недоступен")
            else:
                st.info("Документы не загружены. Перейдите на вкладку 'Загрузка' для добавления файлов.")
            if st.session_state.full_content_dialog["show"]:
                filename = st.session_state.full_content_dialog["filename"]
                st.markdown("---")
                st.subheader(f"Полное содержимое: {filename}")
                if st.session_state.full_content_dialog["content_data"] is None:
                    with st.spinner("Загрузка полного содержимого документа..."):
                        content_data = st.session_state.rag_pipeline.vector_store.get_full_document_content(filename)
                        st.session_state.full_content_dialog["content_data"] = content_data
                content_data = st.session_state.full_content_dialog["content_data"]
                if content_data and content_data.get("success"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Фрагментов", content_data.get("total_chunks", 0))
                    with col2:
                        st.metric("Страниц", content_data.get("page_count", "Unknown"))
                    with col3:
                        chars = content_data.get("total_characters", 0)
                        st.metric("Символов", f"{chars:,}")
                    with col4:
                        words = len(content_data.get("content", "").split())
                        st.metric("Слов", f"{words:,}")
                    st.markdown("---")
                    st.markdown("**Содержимое документа:**")
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col2:
                        if st.button("Копировать текст", type="secondary", use_container_width=True):
                            st.code(content_data["content"], language=None)
                    with col3:
                        if st.button("Закрыть", use_container_width=True):
                            st.session_state.full_content_dialog = {"show": False, "filename": "", "content_data": None}
                            st.rerun()
                    st.text_area(
                        label="Полное содержимое",
                        value=content_data["content"],
                        height=600,
                        disabled=True,
                        key=f"full_content_{filename}",
                        label_visibility="collapsed"
                    )
                    with st.expander("Детальная информация о фрагментах", expanded=False):
                        st.write(f"**Файл разбит на {content_data['total_chunks']} фрагментов:**")
                        for i, chunk_data in enumerate(content_data.get("chunks", [])[:10]):
                            chunk_text = chunk_data["text"][:200] + "..." if len(chunk_data["text"]) > 200 else chunk_data["text"]
                            st.write(f"**Фрагмент {i+1}:** {chunk_text}")
                        if len(content_data.get("chunks", [])) > 10:
                            st.write(f"... и еще {len(content_data['chunks']) - 10} фрагментов")
                else:
                    st.error(f"Ошибка загрузки содержимого: {content_data.get('error', 'Неизвестная ошибка')}")
                    if st.button("Закрыть", use_container_width=True):
                        st.session_state.full_content_dialog = {"show": False, "filename": "", "content_data": None}
                        st.rerun()
        with tab3:
            st.subheader("Управление документами")
            document_summary = st.session_state.rag_pipeline.vector_store.get_document_summary()
            available_files = document_summary.get("filenames", [])
            if available_files:
                st.subheader("Настройки поиска")
                doc_selection = st.radio(
                    "Поиск в документах:",
                    ["Все документы", "Выбранные документы"],
                    key="document_filter_mode_mgmt"
                )
                if doc_selection == "Выбранные документы":
                    selected_files = st.multiselect(
                        "Выберите документы:",
                        options=available_files,
                        default=available_files if st.session_state.selected_documents == "all" else st.session_state.selected_documents,
                        key="selected_files_mgmt"
                    )
                    st.session_state.selected_documents = selected_files if selected_files else "all"
                else:
                    st.session_state.selected_documents = "all"
                if st.session_state.selected_documents == "all":
                    st.info("Поиск во всех документах")
                else:
                    selected_count = len(st.session_state.selected_documents)
                    total_count = len(available_files)
                    st.info(f"Поиск в {selected_count} из {total_count} документов")
                st.markdown("---")
                st.subheader("Управление отдельными документами")
                st.write("**Выберите документы для действий:**")
                selected_for_deletion = []
                for filename in available_files:
                    file_details = document_summary["file_details"].get(filename, {})
                    chunk_count = file_details.get('chunk_count', 0)
                    col1, col2, col3 = st.columns([0.1, 0.7, 0.2])
                    with col1:
                        if st.checkbox("", key=f"delete_checkbox_{filename}"):
                            selected_for_deletion.append(filename)
                    with col2:
                        st.write(f"**{filename}**")
                        st.caption(f"{chunk_count} фрагментов")
                    with col3:
                        if st.button("Переименовать", help=f"Переименовать {filename}", key=f"rename_btn_{filename}", type="secondary", use_container_width=True):
                            st.session_state.rename_dialog = {
                                "show": True,
                                "filename": filename,
                                "new_name": filename
                            }
                            st.rerun()
                if st.session_state.rename_dialog["show"]:
                    st.markdown("---")
                    st.subheader(f"Переименование: {st.session_state.rename_dialog['filename']}")
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        new_name = st.text_input(
                            "Новое имя файла:",
                            value=st.session_state.rename_dialog["new_name"],
                            key="rename_input_field"
                        )
                    with col2:
                        if st.button("Сохранить", type="primary", use_container_width=True):
                            if new_name and new_name != st.session_state.rename_dialog["filename"]:
                                old_filename = st.session_state.rename_dialog["filename"]
                                if not new_name.lower().endswith('.pdf'):
                                    new_name += '.pdf'
                                success = True
                                file_renamed = st.session_state.rag_pipeline.document_processor.rename_physical_file(old_filename, new_name)
                                if file_renamed:
                                    metadata_updated = st.session_state.rag_pipeline.vector_store.update_filename_in_metadata(old_filename, new_name)
                                    success = file_renamed and metadata_updated
                                else:
                                    success = False
                                if success:
                                    st.write(f"Документ переименован: {old_filename} → {new_name}")
                                    st.session_state.rename_dialog = {"show": False, "filename": "", "new_name": ""}
                                    st.rerun()
                                else:
                                    st.error("Ошибка переименования")
                            else:
                                st.warning("Введите корректное имя")
                    with col3:
                        if st.button("Отмена", use_container_width=True):
                            st.session_state.rename_dialog = {"show": False, "filename": "", "new_name": ""}
                            st.rerun()
                if selected_for_deletion:
                    st.markdown("---")
                    st.subheader(f"Удаление документов ({len(selected_for_deletion)} выбрано)")
                    col1, col2 = st.columns(2)
                    with col1:
                        delete_from_vector = st.checkbox(
                            "Удалить из векторной базы", 
                            value=True,
                            help="Удалить все фрагменты и метаданные из поиска"
                        )
                    with col2:
                        delete_physical = st.checkbox(
                            "Удалить физические файлы", 
                            value=False,
                            help="Удалить PDF файлы с диска безвозвратно"
                        )
                    st.write("**Файлы для удаления:**")
                    for filename in selected_for_deletion:
                        st.write(f"• {filename}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Удалить ({len(selected_for_deletion)})", type="secondary", use_container_width=True):
                            deleted_count = 0
                            errors = []
                            for filename in selected_for_deletion:
                                try:
                                    success = True
                                    if delete_from_vector:
                                        vector_success = st.session_state.rag_pipeline.vector_store.delete_documents_by_filename(filename)
                                        success = success and vector_success
                                    if delete_physical:
                                        file_success = st.session_state.rag_pipeline.document_processor.delete_physical_file(filename)
                                        success = success and file_success
                                    if success:
                                        deleted_count += 1
                                    else:
                                        errors.append(filename)
                                except Exception as e:
                                    errors.append(f"{filename} (ошибка: {str(e)})")
                            if deleted_count > 0:
                                st.write(f"Удалено {deleted_count} документов")
                            if errors:
                                st.error(f"Ошибки при удалении: {', '.join(errors)}")
                            if deleted_count > 0:
                                st.rerun()
                    with col2:
                        if st.button("Отменить выбор", use_container_width=True):
                            st.rerun()
                st.markdown("---")
            st.subheader("Общие операции")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Очистить всю векторную базу", type="secondary", use_container_width=True):
                    if st.session_state.rag_pipeline.clear_all_data():
                        st.write("Векторная база очищена")
                        st.rerun()
                    else:
                        st.error("Ошибка очистки базы")
            with col2:
                st.info("Физические файлы остаются в /data/documents/")
            if available_files:
                st.markdown("---")
                st.subheader("Текущий статус")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("В векторной базе", len(available_files))
                with col2:
                    docs_dir = "./data/documents"
                    physical_count = 0
                    if os.path.exists(docs_dir):
                        physical_count = len([f for f in os.listdir(docs_dir) if f.endswith('.pdf')])
                    st.metric("На диске", physical_count)
                with col3:
                    total_chunks = document_summary.get("total_documents", 0)
                    st.metric("Всего фрагментов", total_chunks)
elif page == "Чат":
    st.title("Интерактивный чат")
    
    status = st.session_state.rag_pipeline.get_system_status()
    total_fragments = status["vector_store"]["total_documents"]
    total_files = status["vector_store"]["unique_files"]
    
    st.caption(f"{total_files} документов • {total_fragments} фрагментов")
    
    if not st.session_state.system_initialized:
        st.error("Система не инициализирована. Сначала проверьте систему.")
    elif status["vector_store"]["total_documents"] == 0:
        st.warning("Документы не загружены. Перейдите в раздел 'Управление документами' для загрузки.")
    else:
        st.markdown("""
        <style>
        .stChatMessage > div:first-child {
            display: none !important;
        }
        .stJson {
            max-height: 250px;
            overflow-y: auto;
        }
        </style>
        """, unsafe_allow_html=True)
        
        current_messages = st.session_state.chat_sessions[st.session_state.current_session]["messages"]
        
        for message in current_messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
                    st.markdown(message["content"])
                    
                    if "metadata" in message and st.session_state.debug_mode:
                        metadata = message["metadata"]
                        debug_info = metadata.get("debug_info", {})
                        
                        # 1. Источники (для старых сообщений)
                        with st.expander("Источники", expanded=False):
                            if metadata.get("sources"):
                                for source in metadata["sources"]:
                                    st.caption(f"• {source['filename']} (релевантность: {source['relevance']:.2f})")
                            else:
                                st.caption("Источники недоступны для этого сообщения")
                        
                        # 2. Анализ запроса (для старых сообщений)
                        with st.expander("Анализ запроса", expanded=False):
                            if debug_info and debug_info.get("query_analysis"):
                                query_analysis = debug_info["query_analysis"]
                                lang = query_analysis.get("language", "неизвестен")
                                query_type = query_analysis.get("query_type", "неизвестен") 
                                keywords = query_analysis.get("keywords", [])
                                
                                st.caption(f"**Язык:** {lang}")
                                st.caption(f"**Тип запроса:** {query_type}")
                                if keywords:
                                    st.caption(f"**Ключевые слова:** {', '.join(keywords)}")
                            else:
                                st.caption("Анализ запроса недоступен для этого сообщения")
                        
                        # 3. Производительность (для старых сообщений)
                        with st.expander("Производительность", expanded=False):
                            time_resp = metadata.get("response_time", 0)
                            type_resp = metadata.get("response_type", "unknown")
                            st.caption(f"**Время и тип ответа:** {time_resp:.2f}с, тип: {type_resp}")
                            
                            if debug_info and debug_info.get("search_results"):
                                search_results = debug_info["search_results"]
                                sources_count = len(metadata.get("sources", []))
                                st.caption(f"**Результаты поиска:** найдено: {len(search_results)}, использовано: {sources_count}")
                            else:
                                st.caption("Детали поиска недоступны для этого сообщения")
                        
                        # 4. Отладка (для старых сообщений)
                        with st.expander("Отладка", expanded=False):
                            if debug_info:
                                st.json(debug_info)
                            else:
                                st.caption("Отладочная информация недоступна для этого сообщения")
        
        prompt = st.chat_input("Введите ваш вопрос...")
        
        if prompt:
            user_message = {"role": "user", "content": prompt}
            st.session_state.chat_sessions[st.session_state.current_session]["messages"].append(user_message)
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Думаю..."):
                    response = st.session_state.rag_pipeline.process_query(
                        prompt, 
                        show_debug=st.session_state.debug_mode,
                        selected_documents=st.session_state.selected_documents
                    )
                
                full_response = response["answer"]
                
                # Генератор для эффекта печатания
                def stream_response():
                    for char in full_response:
                        yield char
                        time.sleep(0.02)
                
                st.write_stream(stream_response())
                
                metadata = {
                    "response_time": response.get("response_time", 0),
                    "response_type": response.get("response_type", "unknown"),
                    "sources": response.get("sources", []),
                }
                
                if st.session_state.debug_mode:
                    metadata["debug_info"] = response
                    
                    # 1. Источники
                    with st.expander("Источники", expanded=False):
                        if response.get("sources"):
                            for source in response["sources"]:
                                st.caption(f"• {source['filename']} (релевантность: {source['relevance']:.2f})")
                        
                        # Информация о страницах и фрагментах
                        if response.get("search_results"):
                            pages_info = {}
                            fragment_counts = {}
                            
                            for result in response["search_results"]:
                                filename = result.get("metadata", {}).get("filename", "Unknown")
                                page = result.get("metadata", {}).get("source_page", "Unknown")
                                
                                if filename not in pages_info:
                                    pages_info[filename] = set()
                                    fragment_counts[filename] = 0
                                
                                if page != "Unknown":
                                    pages_info[filename].add(str(page))
                                fragment_counts[filename] += 1
                            
                            if pages_info:
                                st.caption("**Страницы и фрагменты:**")
                                for filename, pages in pages_info.items():
                                    pages_str = ", ".join(sorted(pages)) if pages else "не указаны"
                                    fragments = fragment_counts.get(filename, 0)
                                    st.caption(f"• {filename}: стр. {pages_str} ({fragments} фрагментов)")
                    
                    # 2. Анализ запроса
                    with st.expander("Анализ запроса", expanded=False):
                        query_analysis = response.get("query_analysis", {})
                        routing_result = response.get("routing_result", {})
                        
                        # Язык, тип, ключевые слова
                        if query_analysis:
                            lang = query_analysis.get("language", "неизвестен")
                            query_type = query_analysis.get("query_type", "неизвестен")
                            keywords = query_analysis.get("keywords", [])
                            
                            st.caption(f"**Язык:** {lang}")
                            st.caption(f"**Тип запроса:** {query_type}")
                            if keywords:
                                st.caption(f"**Ключевые слова:** {', '.join(keywords)}")
                        
                        # Решение маршрутизатора
                        if routing_result:
                            reasoning = routing_result.get("reasoning", "не указано")
                            confidence = routing_result.get("confidence", 0)
                            can_answer = routing_result.get("can_answer", False)
                            
                            st.caption(f"**Решение:** {'может ответить' if can_answer else 'не может ответить'}")
                            st.caption(f"**Уверенность:** {confidence:.2f}")
                            st.caption(f"**Обоснование:** {reasoning}")
                    
                    # 3. Производительность
                    with st.expander("Производительность", expanded=False):
                        # Время и тип ответа
                        time_resp = response.get("response_time", 0)
                        type_resp = response.get("response_type", "unknown")
                        st.caption(f"**Время и тип ответа:** {time_resp:.2f}с, тип: {type_resp}")
                        
                        # Результаты поиска
                        search_results = response.get("search_results", [])
                        sources_count = len(response.get("sources", []))
                        st.caption(f"**Результаты поиска:** найдено: {len(search_results)}, использовано: {sources_count}")
                        
                        # Показатели схожести
                        if search_results:
                            similarities = [r.get("similarity", 0) for r in search_results if "similarity" in r]
                            if similarities:
                                avg_sim = sum(similarities) / len(similarities)
                                max_sim = max(similarities)
                                st.caption(f"**Показатели схожести:** средняя: {avg_sim:.2f}, макс: {max_sim:.2f}")
                    
                    # 4. Отладка
                    with st.expander("Отладка", expanded=False):
                        st.json(response)
            
            assistant_message = {
                "role": "assistant", 
                "content": full_response,
                "metadata": metadata
            }
            st.session_state.chat_sessions[st.session_state.current_session]["messages"].append(assistant_message)
            
            conversation_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": prompt,
                "answer": full_response,
                "response_type": response.get("response_type", "unknown"),
                "response_time": response.get("response_time", 0),
                "sources": response.get("sources", [])
            }
            st.session_state.conversation_history.insert(0, conversation_entry)
elif page == "Настройки":
    st.title("Настройки системы")
    if not st.session_state.system_initialized:
        st.error("Система не инициализирована.")
    else:
        tab1, tab2, tab3 = st.tabs(["Параметры", "Экспорт", "Система"])
        with tab1:
            st.subheader("Параметры RAG системы")
            st.markdown("### Выбор моделей Ollama")
            available_models = st.session_state.rag_pipeline.config_manager.get_available_models()
            current_config = st.session_state.rag_pipeline.config_manager.get_current_config()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Модель LLM")
                if available_models['llm']:
                    current_llm_index = 0
                    if current_config.llm_model in available_models['llm']:
                        current_llm_index = available_models['llm'].index(current_config.llm_model)
                    selected_llm = st.selectbox(
                        "Выберите LLM модель",
                        available_models['llm'],
                        index=current_llm_index,
                        key="llm_model_select"
                    )
                    if selected_llm != current_config.llm_model:
                        if st.button("Применить LLM модель", key="apply_llm"):
                            if st.session_state.rag_pipeline.update_models(llm_model=selected_llm):
                                st.rerun()
                else:
                    st.warning("Не найдено доступных LLM моделей")
                    st.info("Установите LLM модель: ollama pull <model_name>")
            with col2:
                st.markdown("#### Модель Embedding")
                if available_models['embedding']:
                    current_embed_index = 0
                    if current_config.embedding_model in available_models['embedding']:
                        current_embed_index = available_models['embedding'].index(current_config.embedding_model)
                    selected_embedding = st.selectbox(
                        "Выберите Embedding модель",
                        available_models['embedding'],
                        index=current_embed_index,
                        key="embedding_model_select"
                    )
                    if selected_embedding != current_config.embedding_model:
                        if st.button("Применить Embedding модель", key="apply_embedding"):
                            if st.session_state.rag_pipeline.update_models(embedding_model=selected_embedding):
                                st.rerun()
                else:
                    st.warning("Не найдено доступных Embedding моделей")
                    st.info("Установите Embedding модель: ollama pull <model_name>")
            st.markdown("#### Текущие модели")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**LLM:** {current_config.llm_model}")
            with col2:
                st.info(f"**Embedding:** {current_config.embedding_model}")
            if st.button("Обновить список моделей", key="refresh_models"):
                st.rerun()
            st.markdown("#### Добавить новые модели")
            st.markdown("Для добавления новых моделей используйте:")
            st.code("ollama pull <model_name>", language="bash")
            st.markdown("**Популярные модели:**")
            st.markdown("- **LLM:** `llama3.2:latest`, `deepseek-r1:latest`")
            st.markdown("- **Embedding:** `nomic-embed-text:latest`, `mxbai-embed-large:latest`")
            st.markdown("---")
            st.markdown("### Настройки маршрутизатора")
            current_threshold = st.session_state.rag_pipeline.router.confidence_threshold
            new_threshold = st.slider(
                "Порог уверенности для маршрутизации",
                min_value=0.0,
                max_value=1.0,
                value=current_threshold,
                step=0.05,
                help="Более высокий порог = более строгая фильтрация"
            )
            if new_threshold != current_threshold:
                st.session_state.rag_pipeline.router.update_confidence_threshold(new_threshold)
                st.write(f"Порог уверенности обновлен: {new_threshold}")
            st.markdown("### Настройки обработки документов")
            chunk_size = st.number_input(
                "Размер фрагмента текста",
                min_value=500,
                max_value=2000,
                value=1000,
                step=100,
                help="Размер каждого фрагмента документа в символах"
            )
            chunk_overlap = st.number_input(
                "Перекрытие фрагментов",
                min_value=0,
                max_value=500,
                value=100,
                step=50,
                help="Количество символов перекрытия между фрагментами"
            )
            if st.button("Применить настройки обработки"):
                st.info("Настройки будут применены к новым документам")
            st.markdown("### Настройки поиска")
            search_k = st.number_input(
                "Количество результатов поиска",
                min_value=1,
                max_value=20,
                value=10,
                help="Количество наиболее релевантных фрагментов для поиска"
            )
            st.markdown("#### Расширенные настройки")
            with st.expander("Настройки уверенности", expanded=False):
                st.info("Настройте, насколько уверенно система должна отвечать на вопросы")
                high_confidence_threshold = st.slider(
                    "Порог высокой уверенности",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="При превышении этого порога система всегда пытается ответить"
                )
                min_context_length = st.number_input(
                    "Минимальная длина контекста",
                    min_value=10,
                    max_value=500,
                    value=100,
                    help="Минимальное количество символов контекста для попытки ответа"
                )
                st.caption("• Более низкие значения = более агрессивная система (больше попыток ответить)")
                st.caption("• Более высокие значения = более консервативная система (меньше ошибок)")
            st.markdown("### Настройки LLM")
            temperature = st.slider(
                "Температура генерации",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.1,
                help="Более высокая температура = более креативные ответы"
            )
            
        with tab2:
            st.subheader("Экспорт данных")
            if st.session_state.conversation_history:
                st.markdown("### Экспорт истории разговоров")
                if st.button("Экспортировать историю разговоров"):
                    export_text = st.session_state.rag_pipeline.export_conversation(
                        st.session_state.conversation_history
                    )
                    st.download_button(
                        label="Скачать историю",
                        data=export_text,
                        filename=f"rag_conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
            else:
                st.info("История разговоров пуста")
        with tab3:
            st.subheader("Системная информация")
            status = st.session_state.rag_pipeline.get_system_status()
            st.markdown("### Конфигурация моделей")
            current_config = st.session_state.rag_pipeline.config_manager.get_current_config()
            available_models = st.session_state.rag_pipeline.config_manager.get_available_models()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### LLM Модель")
                llm_info = st.session_state.rag_pipeline.config_manager.get_model_info(current_config.llm_model)
                if llm_info:
                    model_data = {
                        "Название": llm_info.get('name', 'N/A'),
                        "Размер": llm_info.get('size', 'N/A'),
                        "Изменено": llm_info.get('modified_at', 'N/A'),
                        "Статус": "Доступна" if current_config.llm_model in available_models['llm'] else "Недоступна"
                    }
                    st.json(model_data)
                else:
                    st.error("Информация о модели недоступна")
            with col2:
                st.markdown("#### Embedding Модель")
                embed_info = st.session_state.rag_pipeline.config_manager.get_model_info(current_config.embedding_model)
                if embed_info:
                    model_data = {
                        "Название": embed_info.get('name', 'N/A'),
                        "Размер": embed_info.get('size', 'N/A'),
                        "Изменено": embed_info.get('modified_at', 'N/A'),
                        "Статус": "Доступна" if current_config.embedding_model in available_models['embedding'] else "Недоступна"
                    }
                    st.json(model_data)
                else:
                    st.error("Информация о модели недоступна")
            st.markdown("#### Доступные модели")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("LLM модели", len(available_models['llm']))
                if available_models['llm']:
                    st.write("Список:")
                    for model in available_models['llm'][:5]:
                        st.write(f"  • {model}")
                    if len(available_models['llm']) > 5:
                        st.write(f"  ... и еще {len(available_models['llm']) - 5}")
            with col2:
                st.metric("Embedding модели", len(available_models['embedding']))
                if available_models['embedding']:
                    st.write("Список:")
                    for model in available_models['embedding'][:5]:
                        st.write(f"  • {model}")
                    if len(available_models['embedding']) > 5:
                        st.write(f"  ... и еще {len(available_models['embedding']) - 5}")
            st.markdown("---")
            st.markdown("### Векторная база данных")
            st.json(status["vector_store"])
            st.markdown("### LLM информация")
            st.json(status["llm"])
            st.markdown("### Маршрутизатор")
            st.json(status["router"])
            st.markdown("### Статистика пайплайна")
            st.json(status["pipeline_stats"])
            st.markdown("### Системные действия")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Переинициализировать систему"):
                    st.session_state.system_initialized = False
                    st.session_state.rag_pipeline = RAGPipeline()
                    st.write("Система переинициализирована")
                    st.rerun()
            with col2:
                if st.button("Сбросить все настройки"):
                    st.session_state.clear()
                    st.write("Настройки сброшены")
                    st.rerun()
