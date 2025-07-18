"""
Streamlit Web Interface for RAG System
Provides user-friendly interface for document upload and querying
"""

import streamlit as st
import os
from datetime import datetime
from src.main import RAGPipeline
import json

# Page configuration
st.set_page_config(
    page_title="RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = False

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# Sidebar
st.sidebar.title("🧠 RAG System")
st.sidebar.markdown("---")

# System initialization
if not st.session_state.system_initialized:
    with st.sidebar:
        st.subheader("🔧 Инициализация системы")
        if st.button("Проверить систему", type="primary"):
            st.session_state.system_initialized = st.session_state.rag_pipeline.initialize_system()

# Navigation
page = st.sidebar.selectbox(
    "Выберите страницу",
    ["🏠 Главная", "📚 Управление документами", "💬 Чат", "📊 Статистика", "⚙️ Настройки"]
)

# Debug mode toggle
st.session_state.debug_mode = st.sidebar.checkbox("🐛 Режим отладки", value=st.session_state.debug_mode)

st.sidebar.markdown("---")

# System status in sidebar
if st.session_state.system_initialized:
    status = st.session_state.rag_pipeline.get_system_status()
    st.sidebar.subheader("📊 Статус системы")
    st.sidebar.metric("Документов", status["vector_store"]["total_documents"])
    st.sidebar.metric("Файлов", status["vector_store"]["unique_files"])
    st.sidebar.metric("Запросов", status["pipeline_stats"]["total_queries"])

# Main content
if page == "🏠 Главная":
    st.title("🧠 RAG System - Система вопросов и ответов")
    
    st.markdown("""
    Добро пожаловать в RAG (Retrieval-Augmented Generation) систему!
    
    ## 🚀 Возможности системы:
    - **Загрузка PDF документов** - поддержка больших файлов
    - **Умная маршрутизация** - определение релевантности запросов
    - **Многоязычность** - поддержка русского и казахского языков
    - **Локальная обработка** - работа без интернета через Ollama
    
    ## 🛠️ Технологический стек:
    - **LLM**: Llama 3.2 (через Ollama)
    - **Embeddings**: nomic-embed-text
    - **Vector DB**: ChromaDB
    - **Interface**: Streamlit
    """)
    
    if not st.session_state.system_initialized:
        st.warning("⚠️ Система не инициализирована. Проверьте систему в боковой панели.")
    else:
        st.success("✅ Система готова к работе!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 Быстрый старт")
            st.markdown("""
            1. Перейдите в **Управление документами**
            2. Загрузите PDF файлы
            3. Перейдите в **Чат** для вопросов
            """)
        
        with col2:
            st.subheader("💡 Советы")
            st.markdown("""
            - Задавайте конкретные вопросы
            - Используйте ключевые слова из документов
            - Включите режим отладки для детальной информации
            """)

elif page == "📚 Управление документами":
    st.title("📚 Управление документами")
    
    if not st.session_state.system_initialized:
        st.error("❌ Система не инициализирована. Сначала проверьте систему.")
    else:
        tab1, tab2, tab3 = st.tabs(["📤 Загрузка", "📋 Просмотр", "🗑️ Управление"])
        
        with tab1:
            st.subheader("Загрузка новых документов")
            
            # File upload method selection
            upload_method = st.radio(
                "Способ загрузки:",
                ["Загрузить файлы", "Загрузить из папки"]
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
            
            else:
                directory_path = st.text_input(
                    "Путь к папке с PDF файлами:",
                    placeholder="./data/documents"
                )
                
                if directory_path and st.button("Загрузить из папки", type="primary"):
                    st.session_state.rag_pipeline.load_documents_from_directory(directory_path)
        
        with tab2:
            st.subheader("Загруженные документы")
            
            status = st.session_state.rag_pipeline.get_system_status()
            vector_info = status["vector_store"]
            
            if vector_info["total_documents"] > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Всего фрагментов", vector_info["total_documents"])
                    st.metric("Уникальных файлов", vector_info["unique_files"])
                
                with col2:
                    if vector_info["filenames"]:
                        st.write("**Загруженные файлы:**")
                        for filename in vector_info["filenames"]:
                            st.write(f"- {filename}")
            else:
                st.info("Документы не загружены")
        
        with tab3:
            st.subheader("Управление данными")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Очистить все документы", type="secondary"):
                    if st.session_state.rag_pipeline.clear_all_data():
                        st.success("Все документы удалены")
                        st.rerun()
            
            with col2:
                st.info("Функция удаления отдельных файлов будет добавлена в следующей версии")

elif page == "💬 Чат":
    st.title("💬 Чат с документами")
    
    if not st.session_state.system_initialized:
        st.error("❌ Система не инициализирована. Сначала проверьте систему.")
    else:
        # Check if documents are loaded
        status = st.session_state.rag_pipeline.get_system_status()
        if status["vector_store"]["total_documents"] == 0:
            st.warning("⚠️ Документы не загружены. Перейдите в раздел 'Управление документами' для загрузки.")
        else:
            # Chat interface
            st.subheader(f"📚 Загружено {status['vector_store']['total_documents']} фрагментов из {status['vector_store']['unique_files']} файлов")
            
            # Query input
            query = st.text_input(
                "Введите ваш вопрос:",
                placeholder="Например: Что такое искусственный интеллект?",
                key="chat_input"
            )
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                ask_button = st.button("🔍 Задать вопрос", type="primary")
            
            with col2:
                clear_history = st.button("🗑️ Очистить историю")
            
            if clear_history:
                st.session_state.conversation_history = []
                st.rerun()
            
            # Process query
            if ask_button and query:
                with st.spinner("Обработка запроса..."):
                    response = st.session_state.rag_pipeline.process_query(
                        query, 
                        show_debug=st.session_state.debug_mode
                    )
                
                # Add to conversation history
                conversation_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question": query,
                    "answer": response["answer"],
                    "response_type": response.get("response_type", "unknown"),
                    "response_time": response.get("response_time", 0),
                    "sources": response.get("sources", [])
                }
                
                st.session_state.conversation_history.insert(0, conversation_entry)
                
                # Display response - ANSWER FIRST
                st.markdown("### 💡 Ответ:")
                st.markdown(response["answer"])
                
                # Technical information in expandable sections
                col1, col2 = st.columns(2)
                
                with col1:
                    # Show sources in expander
                    if response.get("sources"):
                        with st.expander("📚 Источники", expanded=False):
                            for source in response["sources"]:
                                st.write(f"**📄 {source['filename']}** (релевантность: {source['relevance']:.2f})")
                
                with col2:
                    # Show response info in expander
                    with st.expander("ℹ️ Информация об ответе", expanded=False):
                        st.write(f"**⏱️ Время ответа:** {response.get('response_time', 0):.2f} сек")
                        st.write(f"**🎯 Тип ответа:** {response.get('response_type', 'unknown')}")
                        if response.get("routing_result"):
                            confidence = response["routing_result"].get("confidence", 0)
                            st.write(f"**🎲 Уверенность:** {confidence:.2f}")
                
                # Show debug info only if debug mode is enabled
                if st.session_state.debug_mode:
                    with st.expander("🐛 Отладочная информация", expanded=False):
                        st.json(response)
            
            # Display conversation history
            if st.session_state.conversation_history:
                st.markdown("---")
                st.subheader("📜 История разговоров")
                
                for i, entry in enumerate(st.session_state.conversation_history[:5]):  # Show last 5
                    with st.expander(f"💬 {entry['timestamp']} - {entry['question'][:50]}...", expanded=False):
                        st.markdown(f"**❓ Вопрос:** {entry['question']}")
                        st.markdown("---")
                        st.markdown(f"**💡 Ответ:** {entry['answer']}")
                        
                        # Technical details in smaller text
                        st.markdown("---")
                        st.caption(f"⏱️ Время: {entry['response_time']:.2f}s | 🎯 Тип: {entry['response_type']}")
                        
                        if entry.get('sources'):
                            st.caption("**📚 Источники:**")
                            for source in entry['sources']:
                                st.caption(f"• {source['filename']}")

elif page == "📊 Статистика":
    st.title("📊 Статистика системы")
    
    if not st.session_state.system_initialized:
        st.error("❌ Система не инициализирована.")
    else:
        status = st.session_state.rag_pipeline.get_system_status()
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Всего документов",
                status["vector_store"]["total_documents"]
            )
        
        with col2:
            st.metric(
                "Уникальных файлов",
                status["vector_store"]["unique_files"]
            )
        
        with col3:
            st.metric(
                "Всего запросов",
                status["pipeline_stats"]["total_queries"]
            )
        
        with col4:
            avg_time = status["pipeline_stats"]["average_response_time"]
            st.metric(
                "Среднее время ответа",
                f"{avg_time:.2f}s"
            )
        
        st.markdown("---")
        
        # Detailed statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Статистика запросов")
            stats = status["pipeline_stats"]
            
            if stats["total_queries"] > 0:
                success_rate = (stats["successful_answers"] / stats["total_queries"]) * 100
                st.metric("Успешных ответов", f"{success_rate:.1f}%")
                st.metric("Успешных запросов", stats["successful_answers"])
                st.metric("Неудачных запросов", stats["failed_answers"])
            else:
                st.info("Статистика запросов пока недоступна")
        
        with col2:
            st.subheader("🧠 Информация о модели")
            llm_info = status["llm"]
            
            st.write(f"**Модель:** {llm_info.get('name', 'Unknown')}")
            st.write(f"**Статус:** {'✅ Доступна' if llm_info.get('available', False) else '❌ Недоступна'}")
            st.write(f"**Размер:** {llm_info.get('size', 'Unknown')}")
        
        st.markdown("---")
        
        # Files information
        st.subheader("📚 Информация о файлах")
        
        if status["vector_store"]["filenames"]:
            for filename in status["vector_store"]["filenames"]:
                st.write(f"📄 {filename}")
        else:
            st.info("Файлы не загружены")
        
        # Conversation history stats
        if st.session_state.conversation_history:
            st.markdown("---")
            st.subheader("💬 Статистика разговоров")
            
            total_conversations = len(st.session_state.conversation_history)
            
            # Response type distribution
            response_types = {}
            for conv in st.session_state.conversation_history:
                resp_type = conv.get('response_type', 'unknown')
                response_types[resp_type] = response_types.get(resp_type, 0) + 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Всего разговоров", total_conversations)
                
            with col2:
                st.write("**Распределение типов ответов:**")
                for resp_type, count in response_types.items():
                    percentage = (count / total_conversations) * 100
                    st.write(f"- {resp_type}: {count} ({percentage:.1f}%)")

elif page == "⚙️ Настройки":
    st.title("⚙️ Настройки системы")
    
    if not st.session_state.system_initialized:
        st.error("❌ Система не инициализирована.")
    else:
        tab1, tab2, tab3 = st.tabs(["🎛️ Параметры", "📤 Экспорт", "🔧 Система"])
        
        with tab1:
            st.subheader("Параметры RAG системы")
            
            # Router settings
            st.markdown("### 🧭 Настройки маршрутизатора")
            
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
            
            # Document processing settings
            st.markdown("### 📄 Настройки обработки документов")
            
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
                # This would require reinitializing the document processor
                st.info("Настройки будут применены к новым документам")
            
            # Search settings
            st.markdown("### 🔍 Настройки поиска")
            
            search_k = st.number_input(
                "Количество результатов поиска",
                min_value=1,
                max_value=10,
                value=5,
                help="Количество наиболее релевантных фрагментов для поиска"
            )
            
            # LLM settings
            st.markdown("### 🧠 Настройки LLM")
            
            temperature = st.slider(
                "Температура генерации",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.1,
                help="Более высокая температура = более креативные ответы"
            )
        
        with tab2:
            st.subheader("📤 Экспорт данных")
            
            # Export conversation history
            if st.session_state.conversation_history:
                st.markdown("### 💬 Экспорт истории разговоров")
                
                if st.button("Экспортировать историю разговоров"):
                    export_text = st.session_state.rag_pipeline.export_conversation(
                        st.session_state.conversation_history
                    )
                    
                    st.download_button(
                        label="📥 Скачать историю",
                        data=export_text,
                        file_name=f"rag_conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
            else:
                st.info("История разговоров пуста")
            
            # Export system statistics
            st.markdown("### 📊 Экспорт статистики")
            
            if st.button("Экспортировать статистику"):
                status = st.session_state.rag_pipeline.get_system_status()
                stats_json = json.dumps(status, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📥 Скачать статистику",
                    data=stats_json,
                    file_name=f"rag_system_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with tab3:
            st.subheader("🔧 Системная информация")
            
            # System status
            status = st.session_state.rag_pipeline.get_system_status()
            
            st.markdown("### 💾 Векторная база данных")
            st.json(status["vector_store"])
            
            st.markdown("### 🧠 LLM информация")
            st.json(status["llm"])
            
            st.markdown("### 🧭 Маршрутизатор")
            st.json(status["router"])
            
            st.markdown("### 📈 Статистика пайплайна")
            st.json(status["pipeline_stats"])
            
            # System actions
            st.markdown("### ⚡ Системные действия")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Переинициализировать систему"):
                    st.session_state.system_initialized = False
                    st.session_state.rag_pipeline = RAGPipeline()
                    st.success("Система переинициализирована")
                    st.rerun()
            
            with col2:
                if st.button("🧹 Сбросить все настройки"):
                    st.session_state.clear()
                    st.success("Настройки сброшены")
                    st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #666;'>
<small>RAG System v1.0<br>
Powered by Ollama & Streamlit</small>
</div>
""", unsafe_allow_html=True)