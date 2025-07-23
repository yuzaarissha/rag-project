import streamlit as st
import os
from datetime import datetime
from src.main import RAGPipeline
import json
st.set_page_config(
    page_title="RAG System",
    page_icon="",   
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
st.sidebar.title("RAG System")
st.sidebar.markdown("---")
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
page = st.sidebar.selectbox(
    "Выберите страницу",
    ["Главная", "Управление документами", "Чат", "Настройки"]
)
st.session_state.debug_mode = st.sidebar.checkbox("Режим отладки", value=st.session_state.debug_mode)
st.sidebar.markdown("---")
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
    st.title("RAG System - Система вопросов и ответов")
    st.markdown("""
    Добро пожаловать в RAG (Retrieval-Augmented Generation) систему!
    
    ## Возможности системы:
    - **Загрузка PDF документов** - поддержка больших файлов
    - **Умная маршрутизация** - определение релевантности запросов
    - **Многоязычность** - поддержка русского и казахского языков
    - **Локальная обработка** - работа без интернета через Ollama
    
    ## Технологический стек:
    - **LLM**: Автоматический выбор из доступных моделей Ollama
    - **Embeddings**: Автоматический выбор из доступных моделей Ollama
    - **Vector DB**: ChromaDB
    - **Interface**: Streamlit
    """)
    if not st.session_state.system_initialized:
        st.warning("Система не инициализирована. Проверьте систему в боковой панели.")
    else:
        config = st.session_state.rag_pipeline.config_manager.get_current_config()
        st.success("Система готова к работе!")
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
elif page == "Управление документами":
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
                        st.success(f"Успешно обработано {success_count} из {len(uploaded_files)} файлов")
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
                                        st.success("Векторная база очищена")
                                    else:
                                        st.error("Ошибка очистки")
                                        st.stop()
                                st.info(f"Переиндексация {len(pdf_files)} файлов...")
                                success = st.session_state.rag_pipeline.reindex_existing_documents(docs_dir)
                                if success:
                                    st.success("Переиндексация завершена успешно!")
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
                                    st.success("Физический файл найден")
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
                if "documents_to_delete" not in st.session_state:
                    st.session_state.documents_to_delete = []
                if "rename_dialog" not in st.session_state:
                    st.session_state.rename_dialog = {"show": False, "filename": "", "new_name": ""}
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
                                    st.success(f"Документ переименован: {old_filename} → {new_name}")
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
                                st.success(f"Успешно удалено {deleted_count} документов")
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
                        st.success("Векторная база очищена")
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
    st.title("Чат с документами")
    if not st.session_state.system_initialized:
        st.error("Система не инициализирована. Сначала проверьте систему.")
    else:
        status = st.session_state.rag_pipeline.get_system_status()
        if status["vector_store"]["total_documents"] == 0:
            st.warning("Документы не загружены. Перейдите в раздел 'Управление документами' для загрузки.")
        else:
            st.subheader(f"Загружено {status['vector_store']['total_documents']} фрагментов из {status['vector_store']['unique_files']} файлов")
            st.markdown("### Задайте вопрос")
            query = st.text_area(
                "",
                placeholder="Введите ваш вопрос здесь...\n\nНапример:\n• Что такое искусственный интеллект?\n• Какие основные принципы работы данной системы?\n• Расскажите подробнее о...",
                key="chat_input",
                height=120,
                label_visibility="collapsed",
                help="Используйте многострочный ввод для сложных или длинных вопросов"
            )
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                ask_button = st.button("Задать вопрос", type="primary", use_container_width=True)
            with col2:
                clear_history = st.button("Очистить историю", use_container_width=True)
            with col3:
                if st.button("Советы", use_container_width=True):
                    st.info("""
                    **Советы для лучших результатов:**
                    • Формулируйте вопросы конкретно
                    • Используйте ключевые слова из документов
                    • Задавайте вопросы на том же языке, что и документы
                    """)
            if clear_history:
                st.session_state.conversation_history = []
                st.rerun()
            if ask_button and query:
                with st.spinner("Обработка запроса..."):
                    response = st.session_state.rag_pipeline.process_query(
                        query, 
                        show_debug=st.session_state.debug_mode,
                        selected_documents=st.session_state.selected_documents
                    )
                conversation_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question": query,
                    "answer": response["answer"],
                    "response_type": response.get("response_type", "unknown"),
                    "response_time": response.get("response_time", 0),
                    "sources": response.get("sources", [])
                }
                st.session_state.conversation_history.insert(0, conversation_entry)
                st.markdown("## Ответ")
                with st.container():
                    st.markdown(response['answer'])
                st.markdown("---")
                st.markdown("### Дополнительная информация")
                if response.get("sources"):
                    with st.expander("Источники", expanded=False):
                        for source in response["sources"]:
                            st.write(f"• {source['filename']} (релевантность: {source['relevance']:.2f})")
                else:
                    with st.expander("Источники", expanded=False):
                        st.write("Источники не найдены")
                with st.expander("Детали ответа", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Время ответа", f"{response.get('response_time', 0):.2f} сек")
                        st.metric("Тип ответа", response.get('response_type', 'unknown'))
                    with col2:
                        if response.get("routing_result"):
                            confidence = response["routing_result"].get("confidence", 0)
                            st.metric("Уверенность", f"{confidence:.2f}")
                with st.expander("Детали обработки", expanded=False):
                    st.write(f"Найдено {len(response.get('search_results', []))} релевантных документов")
                    if response.get('search_results'):
                        distances = [r.get('distance', 1.0) for r in response['search_results']]
                        if distances:
                            best_distance = min(distances)
                            avg_distance = sum(distances) / len(distances)
                            st.write(f"**Качество поиска:** Лучшее совпадение: {1-best_distance:.2f}, Среднее: {1-avg_distance:.2f}")
                    if response.get('routing_result'):
                        routing_info = response['routing_result']
                        st.write(f"**Решение маршрутизатора:** {'Можно ответить' if routing_info.get('can_answer', False) else 'Недостаточно данных'}")
                        st.write(f"**Уверенность:** {routing_info.get('confidence', 0):.2f}")
                        context_length = len(routing_info.get('context', ''))
                        st.write(f"**Размер контекста:** {context_length} символов")
                        if routing_info.get('query_analysis'):
                            analysis = routing_info['query_analysis']
                            st.write(f"**Тип вопроса:** {analysis.get('query_type', 'unknown')}")
                            st.write(f"**Язык:** {analysis.get('language', 'unknown')}")
                            if analysis.get('keywords'):
                                st.write(f"**Ключевые слова:** {', '.join(analysis['keywords'])}")
                if st.session_state.debug_mode:
                    with st.expander("Полная отладочная информация", expanded=False):
                        st.json(response)
            if st.session_state.conversation_history:
                st.markdown("---")
                st.subheader("История разговоров")
                for i, entry in enumerate(st.session_state.conversation_history[:5]):
                    with st.expander(f"{entry['timestamp']} - {entry['question'][:50]}...", expanded=False):
                        st.markdown(f"**Вопрос:** {entry['question']}")
                        st.markdown("**Ответ:**")
                        st.info(entry['answer'])
                        st.markdown("**Дополнительная информация:**")
                        st.caption(f"Время: {entry['response_time']:.2f}s | Тип: {entry['response_type']}")
                        if entry.get('sources'):
                            st.caption("**Источники:**")
                            for source in entry['sources']:
                                st.caption(f"• {source['filename']}")
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
            st.markdown("- **LLM:** `llama3.2:latest`, `deepseek-r1:latest`, `qwen2.5:latest`")
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
                st.success(f"Порог уверенности обновлен: {new_threshold}")
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
                        file_name=f"rag_conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
            else:
                st.info("История разговоров пуста")
            st.markdown("### Экспорт статистики")
            if st.button("Экспортировать статистику"):
                status = st.session_state.rag_pipeline.get_system_status()
                stats_json = json.dumps(status, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Скачать статистику",
                    data=stats_json,
                    file_name=f"rag_system_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
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
                    st.success("Система переинициализирована")
                    st.rerun()
            with col2:
                if st.button("Сбросить все настройки"):
                    st.session_state.clear()
                    st.success("Настройки сброшены")
                    st.rerun()
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center;'>
<small>RAG System v1.5<br>
Powered by Ollama & Streamlit</small>
</div>
""", unsafe_allow_html=True)