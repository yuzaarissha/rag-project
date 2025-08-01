import logging
from typing import Dict, Any, List, Optional
import streamlit as st
import json
import os
from datetime import datetime, timedelta
import uuid
import time
import threading
import atexit


class SessionManager:
    """Упрощенный менеджер сессий для работы со стандартным st.session_state.Локальная работа без персистентности между сессиями браузера."""

    def __init__(self):
        self .logger = logging .getLogger(__name__)
        self .autosave_dir = "./data/saves"

        os .makedirs(self .autosave_dir, exist_ok=True)

        self .autosave_interval = 60
        self .max_autosave_files = 10
        self .autosave_timer = None
        self .last_state_hash = None

        atexit .register(self .cleanup_on_exit)

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
            'distance_threshold': 0.5,
            'confidence_threshold': 0.5,
            'temperature': 0.5,
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

            self .auto_save_session()
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

        distance_threshold = self .get_setting('distance_threshold', 0.5)
        validation['distance_threshold'] = 0.1 <= distance_threshold <= 0.9

        confidence_threshold = self .get_setting('confidence_threshold', 0.5)
        validation['confidence_threshold'] = 0.1 <= confidence_threshold <= 0.9

        temperature = self .get_setting('temperature', 0.5)
        validation['temperature'] = 0.0 <= temperature <= 1.0

        max_tokens = self .get_setting('max_tokens', 2000)
        validation['max_tokens'] = 200 <= max_tokens <= 4000

        return validation

    def get_system_prompts(self) -> Dict[str, str]:
        """Возвращает доступные стили системных промптов на основе исследований 2024"""
        return {
            "Профессиональный": "Отвечай структурированно в деловом стиле. Используй четкую логику, конкретные факты и цифры. Избегай эмоциональной окраски. Указывай уровень уверенности в ответе. Структурируй ответ по пунктам при необходимости.",
            "Академический": "Используй строго научный подход с объективным тоном. Обязательно ссылайся на конкретные факты из контекста. Применяй сложные конструкции предложений. Избегай сокращений и разговорной лексики. Четко разграничивай факты и выводы.",
            "Дружелюбный": "Общайся теплым, приближенным тоном для комфортного взаимодействия. Используй простые объяснения. Проявляй эмпатию к вопросам пользователя. При неполной информации мягко объясняй ограничения.",
            "Краткий": "Отвечай лаконично в стиле Хемингуэя - максимум информации минимумом слов. Используй короткие предложения. Сразу переходи к сути. Избегай вводных конструкций. Указывай только ключевые факты.",
            "Подробный": "Давай развернутые объяснения с примерами, контекстом и дополнительными деталями. Показывай пошаговое рассуждение. Объясняй взаимосвязи между фактами. Предоставляй background информацию для лучшего понимания.",
            "Формальный": "Используй официальный стиль для документооборота и формальных процедур. Применяй стандартную терминологию. Избегай личных местоимений. Структурируй ответ согласно формальным требованиям. Указывай степень достоверности информации.",
            "Объясняющий": "Объясняй сложные концепции простым языком как Фейнман - понятно и доступно. Используй аналогии и примеры из повседневной жизни. Разбивай сложные темы на простые компоненты. Проверяй понимание на каждом шаге.",
            "Технический": "Предоставляй точные технические инструкции с конкретными параметрами, кодом и спецификациями. Используй профессиональную терминологию. Структурируй информацию пошагово. Указывай предварительные требования и возможные ошибки.",
            "Креативный": "Используй элементы остроумия и творческий подход при сохранении информативности. Применяй нестандартные форматы подачи информации. Делай ответы запоминающимися. Сохраняй баланс между креативностью и точностью фактов."
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

    def get_full_session_state(self) -> Dict[str, Any]:
        """Получает полное состояние сессии включая все данные интерфейса"""
        full_state = {}

        for key in self .session_keys:
            if hasattr(st .session_state, key):
                full_state[key] = getattr(st .session_state, key)

        additional_keys = [
            'system_initialized',
            'rag_pipeline',
            'full_content_dialog',
            'show_rename_dialog',
            'show_dropdown_menu',
            'documents_to_delete',
            'rename_dialog'
        ]

        for key in additional_keys:
            if hasattr(st .session_state, key):
                value = getattr(st .session_state, key)

                if key == 'rag_pipeline' and hasattr(value, 'config_manager'):
                    full_state[key + '_config'] = {
                        'llm_model': value .config_manager .config .llm_model,
                        'embedding_model': value .config_manager .config .embedding_model
                    }
                else:
                    full_state[key] = value

        return full_state

    def calculate_state_hash(self, state: Dict[str, Any]) -> str:
        """Вычисляет хеш состояния для определения изменений"""
        try:
            import hashlib
            state_str = json .dumps(state, sort_keys=True, default=str)
            return hashlib .md5(state_str .encode()).hexdigest()
        except Exception:
            return str(time .time())

    def auto_save_session(self) -> bool:
        """Автоматическое сохранение сессии"""
        try:
            current_state = self .get_full_session_state()
            current_hash = self .calculate_state_hash(current_state)

            if current_hash != self .last_state_hash:
                timestamp = datetime .now().strftime('%Y%m%d_%H%M%S')
                autosave_name = f"autosave_{timestamp}"

                session_data = {
                    "id": str(uuid .uuid4()),
                    "name": autosave_name,
                    "created_at": datetime .now().isoformat(),
                    "type": "autosave",
                    "settings": current_state,
                    "version": "2.0",
                    "state_hash": current_hash
                }

                autosave_file = os .path .join(
                    self .autosave_dir, f"{autosave_name}.json")
                with open(autosave_file, 'w', encoding='utf-8')as f:
                    json .dump(session_data, f, ensure_ascii=False, indent=2)

                self .last_state_hash = current_hash
                self .logger .info(
                    f"Автосохранение выполнено: {autosave_name}")

                self .cleanup_old_autosaves()
                return True

            return False

        except Exception as e:
            self .logger .error(f"Ошибка автосохранения: {str(e)}")
            return False

    def start_autosave_timer(self):
        """Запускает таймер автосохранения"""
        try:
            if self .autosave_timer:
                self .autosave_timer .cancel()

            def autosave_task():
                self .auto_save_session()

                self .start_autosave_timer()

            self .autosave_timer = threading .Timer(
                self .autosave_interval, autosave_task)
            self .autosave_timer .daemon = True
            self .autosave_timer .start()

        except Exception as e:
            self .logger .error(f"Ошибка запуска автосохранения: {str(e)}")

    def stop_autosave_timer(self):
        """Останавливает таймер автосохранения"""
        if self .autosave_timer:
            self .autosave_timer .cancel()
            self .autosave_timer = None

    def cleanup_old_autosaves(self):
        """Удаляет старые автосохранения, оставляя только последние"""
        try:
            if not os .path .exists(self .autosave_dir):
                return 0

            autosaves = []
            for filename in os .listdir(self .autosave_dir):
                if filename .startswith('autosave_') and filename .endswith('.json'):
                    filepath = os .path .join(self .autosave_dir, filename)
                    try:
                        stat = os .stat(filepath)
                        autosaves .append((filepath, stat .st_mtime))
                    except Exception:
                        continue

            autosaves .sort(key=lambda x: x[1], reverse=True)

            deleted_count = 0
            for filepath, _ in autosaves[self .max_autosave_files:]:
                try:
                    os .remove(filepath)
                    deleted_count += 1
                    self .logger .info(
                        f"Удалено старое автосохранение: {os .path .basename(filepath)}")
                except Exception:
                    continue

            return deleted_count

        except Exception as e:
            self .logger .error(f"Ошибка очистки автосохранений: {str(e)}")
            return 0

    def get_latest_autosave(self) -> Optional[Dict[str, Any]]:
        """Возвращает последнее автосохранение"""
        try:
            if not os .path .exists(self .autosave_dir):
                return None

            latest_file = None
            latest_time = 0

            for filename in os .listdir(self .autosave_dir):
                if filename .startswith('autosave_') and filename .endswith('.json'):
                    filepath = os .path .join(self .autosave_dir, filename)
                    try:
                        stat = os .stat(filepath)
                        if stat .st_mtime > latest_time:
                            latest_time = stat .st_mtime
                            latest_file = filepath
                    except Exception:
                        continue

            if latest_file:
                with open(latest_file, 'r', encoding='utf-8')as f:
                    return json .load(f)

            return None

        except Exception as e:
            self .logger .error(
                f"Ошибка получения последнего автосохранения: {str(e)}")
            return None

    def restore_from_autosave(self) -> bool:
        """Восстанавливает состояние из последнего автосохранения"""
        try:
            latest_autosave = self .get_latest_autosave()
            if not latest_autosave:
                return False

            settings = latest_autosave .get("settings", {})

            for key, value in settings .items():
                if key in self .session_keys or key in [
                    'system_initialized', 'full_content_dialog',
                    'show_rename_dialog', 'show_dropdown_menu',
                    'documents_to_delete', 'rename_dialog'
                ]:
                    setattr(st .session_state, key, value)

            self .logger .info(
                f"Состояние восстановлено из автосохранения: {latest_autosave['name']}")
            return True

        except Exception as e:
            self .logger .error(
                f"Ошибка восстановления из автосохранения: {str(e)}")
            return False

    def cleanup_on_exit(self):
        """Очистка при выходе из приложения"""
        try:

            self .auto_save_session()

            self .stop_autosave_timer()

            self .cleanup_old_autosaves()

            self .logger .info("Очистка при выходе выполнена")

        except Exception as e:
            self .logger .error(f"Ошибка очистки при выходе: {str(e)}")

    def initialize_with_autorestore(self):
        """Инициализация с автовосстановлением"""

        self .initialize_defaults()

        if self .restore_from_autosave():
            self .logger .info("Состояние восстановлено из автосохранения")
        else:
            self .logger .info(
                "Автосохранение не найдено, используем defaults")

        self .start_autosave_timer()

        self .cleanup_old_autosaves()
