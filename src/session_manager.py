import logging
from typing import Dict, Any, List, Optional
import streamlit as st
import json
import os
from datetime import datetime
import uuid


class SessionManager:
    """Упрощенный менеджер сессий для работы со стандартным st.session_state.Локальная работа без персистентности между сессиями браузера."""

    def __init__(self):
        self .logger = logging .getLogger(__name__)
        self .sessions_dir = "./data/sessions"

        os .makedirs(self .sessions_dir, exist_ok=True)

        self .session_keys = [
            'chat_sessions',
            'current_session',
            'conversation_history',
            'selected_documents',
            'debug_mode',
            'chunk_size',
            'chunk_overlap',
            'search_k',
            'search_method',
            'distance_threshold',
            'confidence_threshold',
            'temperature',
            'max_tokens',
            'system_prompt_style'
        ]

    def initialize_defaults(self):
        """Инициализирует значения по умолчанию для всех настроек RAG системы"""
        defaults = {
            'chat_sessions': {"default": {"name": "Новый чат", "messages": [], "created_at": None}},
            'current_session': "default",
            'conversation_history': [],
            'selected_documents': "all",
            'debug_mode': False,
            'chunk_size': 512,
            'chunk_overlap': 25,
            'search_k': 10,
            'search_method': "mmr",
            'distance_threshold': 0.25,
            'confidence_threshold': 0.5,
            'temperature': 0.2,
            'max_tokens': 2000,
            'system_prompt_style': "Профессиональный"
        }

        for key, default_value in defaults .items():
            if key not in st .session_state:
                setattr(st .session_state, key, default_value)

        self .logger .info(
            "Настройки RAG системы инициализированы значениями по умолчанию")

    def get_setting(self, key: str, default=None):
        """Получает значение настройки из session_state"""
        return getattr(st .session_state, key, default)

    def set_setting(self, key: str, value):
        """Устанавливает значение настройки в session_state"""
        if key in self .session_keys:
            setattr(st .session_state, key, value)
            self .logger .debug(f"Настройка {key} обновлена: {value}")
        else:
            self .logger .warning(f"Неизвестная настройка: {key}")

    def get_all_settings(self) -> Dict[str, Any]:
        """Возвращает все текущие настройки RAG системы"""
        settings = {}
        for key in self .session_keys:
            settings[key] = getattr(st .session_state, key, None)
        return settings

    def reset_to_defaults(self):
        """Сбрасывает все настройки к значениям по умолчанию"""
        for key in self .session_keys:
            if hasattr(st .session_state, key):
                delattr(st .session_state, key)
        self .initialize_defaults()
        self .logger .info("Настройки сброшены к значениям по умолчанию")

    def validate_settings(self) -> Dict[str, bool]:
        """Проверяет валидность текущих настроек"""
        validation = {}

        chunk_size = self .get_setting('chunk_size', 512)
        validation['chunk_size'] = 256 <= chunk_size <= 1024

        chunk_overlap = self .get_setting('chunk_overlap', 25)
        validation['chunk_overlap'] = 5 <= chunk_overlap <= 50

        search_k = self .get_setting('search_k', 10)
        validation['search_k'] = 5 <= search_k <= 15

        distance_threshold = self .get_setting('distance_threshold', 0.25)
        validation['distance_threshold'] = 0.2 <= distance_threshold <= 0.9

        confidence_threshold = self .get_setting('confidence_threshold', 0.5)
        validation['confidence_threshold'] = 0.3 <= confidence_threshold <= 0.9

        temperature = self .get_setting('temperature', 0.2)
        validation['temperature'] = 0.0 <= temperature <= 1.0

        max_tokens = self .get_setting('max_tokens', 2000)
        validation['max_tokens'] = 200 <= max_tokens <= 4000

        return validation

    def get_system_prompts(self) -> Dict[str, str]:
        """Возвращает доступные стили системных промптов на основе ChatGPT"""
        return {
            "Профессиональный": "Отвечай структурированно, деловым стилем для рабочих задач. Используй четкую логику и факты.",
            "Академический": "Используй научный подход с акцентом на точность, источники и детальный анализ.",
            "Дружелюбный": "Общайся теплым, приближенным тоном для комфортного взаимодействия.",
            "Краткий": "Отвечай лаконично в стиле Хемингуэя - максимум информации минимумом слов.",
            "Подробный": "Давай развернутые объяснения с примерами, контекстом и дополнительными деталями.",
            "Формальный": "Используй официальный стиль для документооборота и формальных процедур.",
            "Объясняющий": "Объясняй сложные концепции простым языком как Фейнман - понятно и доступно.",
            "Технический": "Предоставляй точные технические инструкции и детали для специалистов.",
            "Креативный": "Используй элементы остроумия и творческий подход при сохранении информативности."
        }

    def get_session_info(self) -> Dict[str, Any]:
        """Возвращает информацию о текущей сессии"""
        settings = self .get_all_settings()
        validation = self .validate_settings()

        return {
            "session_type": "local",
            "total_settings": len(self .session_keys),
            "initialized_settings": len([k for k in self .session_keys if hasattr(st .session_state, k)]),
            "valid_settings": sum(validation .values()),
            "invalid_settings": len(validation)-sum(validation .values()),
            "current_system_prompt": self .get_setting('system_prompt_style', 'Профессиональный'),
            "rag_parameters": {
                "chunk_size": self .get_setting('chunk_size'),
                "search_method": self .get_setting('search_method'),
                "temperature": self .get_setting('temperature')
            }
        }

    def clear_session(self) -> bool:
        """Очищает текущую сессию"""
        try:
            for key in self .session_keys:
                if hasattr(st .session_state, key):
                    delattr(st .session_state, key)

            self .logger .info("Сессия очищена")
            return True

        except Exception as e:
            self .logger .error(f"Ошибка очистки сессии: {str(e)}")
            return False

    def save_session(self, session_name: str = None) -> Dict[str, Any]:
        """Сохраняет текущую сессию в файл (БЕЗ шифрования)"""
        try:
            if not session_name:
                session_name = f"session_{datetime .now().strftime('%Y%m%d_%H%M%S')}"

            session_data = {
                "id": str(uuid .uuid4()),
                "name": session_name,
                "created_at": datetime .now().isoformat(),
                "settings": self .get_all_settings(),
                "conversation_history": getattr(st .session_state, 'conversation_history', []),
                "version": "2.0"
            }

            session_file = os .path .join(
                self .sessions_dir, f"{session_name}.json")
            with open(session_file, 'w', encoding='utf-8')as f:
                json .dump(session_data, f, ensure_ascii=False, indent=2)

            self .logger .info(f"Сессия сохранена: {session_file}")
            return {
                "success": True,
                "session_id": session_data["id"],
                "session_name": session_name,
                "file_path": session_file
            }

        except Exception as e:
            self .logger .error(f"Ошибка сохранения сессии: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def load_session(self, session_name: str) -> Dict[str, Any]:
        """Загружает сессию из файла (БЕЗ расшифровки)"""
        try:
            session_file = os .path .join(
                self .sessions_dir, f"{session_name}.json")

            if not os .path .exists(session_file):
                return {
                    "success": False,
                    "error": f"Файл сессии не найден: {session_file}"
                }

            with open(session_file, 'r', encoding='utf-8')as f:
                session_data = json .load(f)

            if "settings" in session_data:
                for key, value in session_data["settings"].items():
                    if key in self .session_keys:
                        setattr(st .session_state, key, value)

            if "conversation_history" in session_data:
                st .session_state .conversation_history = session_data["conversation_history"]

            self .logger .info(f"Сессия загружена: {session_file}")
            return {
                "success": True,
                "session_data": session_data,
                "loaded_settings": len(session_data .get("settings", {})),
                "conversation_count": len(session_data .get("conversation_history", []))
            }

        except Exception as e:
            self .logger .error(f"Ошибка загрузки сессии: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_saved_sessions(self) -> List[Dict[str, Any]]:
        """Возвращает список сохраненных сессий"""
        try:
            sessions = []

            if not os .path .exists(self .sessions_dir):
                return sessions

            for filename in os .listdir(self .sessions_dir):
                if filename .endswith('.json'):
                    session_file = os .path .join(self .sessions_dir, filename)
                    try:
                        with open(session_file, 'r', encoding='utf-8')as f:
                            session_data = json .load(f)

                        sessions .append({
                            "filename": filename,
                            "name": session_data .get("name", filename .replace('.json', '')),
                            "created_at": session_data .get("created_at", "Unknown"),
                            "id": session_data .get("id", "Unknown"),
                            "conversation_count": len(session_data .get("conversation_history", [])),
                            "version": session_data .get("version", "1.0")
                        })

                    except Exception as e:
                        self .logger .warning(
                            f"Ошибка чтения сессии {filename}: {str(e)}")
                        continue

            sessions .sort(key=lambda x: x .get(
                "created_at", ""), reverse=True)
            return sessions

        except Exception as e:
            self .logger .error(f"Ошибка получения списка сессий: {str(e)}")
            return []

    def delete_session(self, session_name: str) -> bool:
        """Удаляет сохраненную сессию"""
        try:
            session_file = os .path .join(
                self .sessions_dir, f"{session_name}.json")

            if os .path .exists(session_file):
                os .remove(session_file)
                self .logger .info(f"Сессия удалена: {session_file}")
                return True
            else:
                self .logger .warning(f"Файл сессии не найден: {session_file}")
                return False

        except Exception as e:
            self .logger .error(f"Ошибка удаления сессии: {str(e)}")
            return False

    def auto_save_session(self) -> bool:
        """Автоматическое сохранение текущей сессии"""
        try:

            auto_name = f"auto_save_{datetime .now().strftime('%Y%m%d_%H%M%S')}"
            result = self .save_session(auto_name)

            if result["success"]:
                self .logger .info(f"Автосохранение выполнено: {auto_name}")
                return True
            else:
                self .logger .error(
                    f"Ошибка автосохранения: {result['error']}")
                return False

        except Exception as e:
            self .logger .error(f"Ошибка автосохранения: {str(e)}")
            return False

    def export_session_data(self, session_name: str) -> Optional[Dict[str, Any]]:
        """Экспортирует данные сессии для анализа"""
        try:
            session_file = os .path .join(
                self .sessions_dir, f"{session_name}.json")

            if not os .path .exists(session_file):
                return None

            with open(session_file, 'r', encoding='utf-8')as f:
                session_data = json .load(f)

            return session_data

        except Exception as e:
            self .logger .error(f"Ошибка экспорта данных сессии: {str(e)}")
            return None

    def cleanup_old_sessions(self, days_to_keep: int = 30) -> int:
        """Удаляет старые сессии (старше указанного количества дней)"""
        try:
            if not os .path .exists(self .sessions_dir):
                return 0

            from datetime import timedelta
            cutoff_date = datetime .now()-timedelta(days=days_to_keep)
            deleted_count = 0

            for filename in os .listdir(self .sessions_dir):
                if filename .endswith('.json'):
                    session_file = os .path .join(self .sessions_dir, filename)
                    try:
                        with open(session_file, 'r', encoding='utf-8')as f:
                            session_data = json .load(f)

                        created_at = session_data .get("created_at")
                        if created_at:
                            session_date = datetime .fromisoformat(created_at)
                            if session_date < cutoff_date:
                                os .remove(session_file)
                                deleted_count += 1
                                self .logger .info(
                                    f"Удалена старая сессия: {filename}")

                    except Exception as e:
                        self .logger .warning(
                            f"Ошибка при проверке файла {filename}: {str(e)}")
                        continue

            self .logger .info(
                f"Очистка завершена: удалено {deleted_count} старых сессий")
            return deleted_count

        except Exception as e:
            self .logger .error(f"Ошибка очистки старых сессий: {str(e)}")
            return 0
