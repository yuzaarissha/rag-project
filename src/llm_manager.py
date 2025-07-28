import ollama
from typing import Dict, Any, Optional, List
import json
import logging
class LLMManager:
    def __init__(self, model_name: str = "llama3.2:latest"):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.system_prompt = """РОЛЬ: Экспертный ассистент-аналитик документов

ВАША МИССИЯ:
Предоставлять точные, профессиональные ответы на основе загруженных документов с максимальной полезностью для пользователя.

ОСНОВНЫЕ ПРИНЦИПЫ:

АНАЛИЗ И СИНТЕЗ:
• Тщательно изучайте весь предоставленный контекст
• Объединяйте информацию из разных частей документов
• Выявляйте ключевые связи и закономерности
• Структурируйте ответ логично и последовательно

МНОГОЯЗЫЧНОСТЬ:
• Автоматически определяйте язык вопроса
• Отвечайте на том же языке (русский/казахский/английский)
• Сохраняйте терминологию и стиль оригинальных документов

КАЧЕСТВО ОТВЕТОВ:
• Давайте конкретные, практичные ответы
• Избегайте общих фраз и абстракций
• Приводите конкретные факты, цифры, даты
• Структурируйте сложную информацию в понятном виде

ФОРМАТ ОТВЕТА:
Чистый, читаемый текст без технических элементов. Используйте структурирование (списки, абзацы) для лучшего восприятия.

СТРОГО ЗАПРЕЩЕНО:
• Добавлять технические ссылки: [Источник: file.pdf, chunk_id: abc123]
• Включать секции "Источники:" или "References:"
• Показывать внутренние размышления или сомнения
• Выдумывать информацию, не содержащуюся в документах
• Использовать расплывчатые формулировки

СТАНДАРТНЫЕ ОТВЕТЫ:
• При отсутствии информации: "Информация не найдена в предоставленных документах"
• При частичной информации: четко указывайте ограничения данных

ЦЕЛЬ: Быть максимально полезным, точным и профессиональным помощником."""
    def update_model(self, model_name: str) -> bool:
        try:
            test_response = ollama.generate(
                model=model_name,
                prompt="test",
                options={"num_predict": 1}
            )
            self.model_name = model_name
            return True
        except Exception as e:
            self.logger.error(f"Failed to update model to {model_name}: {str(e)}")
            return False
    def check_model_availability(self) -> bool:
        try:
            models_response = ollama.list()
            available_models = []
            if isinstance(models_response, dict) and 'models' in models_response:
                for model in models_response['models']:
                    if isinstance(model, dict) and 'name' in model:
                        available_models.append(model['name'])
            self.logger.info(f"Доступные модели: {available_models}")
            is_available = self.model_name in available_models
            if not is_available:
                self.logger.warning(f"Модель {self.model_name} не найдена в списке доступных моделей")
                self.logger.info("Попробуйте использовать точное название из 'ollama list'")
            return is_available
        except Exception as e:
            self.logger.error(f"Ошибка проверки доступности моделей: {str(e)}")
            try:
                test_response = ollama.generate(
                    model=self.model_name,
                    prompt="test",
                    options={"num_predict": 1}
                )
                self.logger.info(f"Модель {self.model_name} работает (прямое тестирование)")
                return True
            except Exception as test_error:
                self.logger.error(f"Модель {self.model_name} недоступна: {test_error}")
                return False
    def generate_response(self, prompt: str, context: str = "", temperature: float = 0.2) -> str:
        try:
            if context:
                full_prompt = f"""БАЗА ЗНАНИЙ:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{prompt}

ВАШ ЭКСПЕРТНЫЙ ОТВЕТ:"""
            else:
                full_prompt = f"""ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{prompt}

ВАШ ОТВЕТ:"""
            response = ollama.generate(
                model=self.model_name,
                prompt=full_prompt,
                system=self.system_prompt,
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 800,
                    "repeat_penalty": 1.1
                }
            )
            clean_response = self._clean_response(response['response'].strip())
            return clean_response
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "Извините, произошла ошибка при генерации ответа."
    def generate_router_decision(self, query: str, context: str) -> bool:
        try:
            router_prompt = f"""РОЛЬ: Интеллектуальный маршрутизатор RAG-системы

МИССИЯ: Определить релевантность найденного контекста для ответа на вопрос пользователя

КРИТЕРИИ ОЦЕНКИ:

ОТВЕЧАЙТЕ "Да" ЕСЛИ:
• Контекст содержит ПРЯМОЙ ответ на вопрос
• Контекст содержит ЧАСТИЧНУЮ информацию по теме
• Контекст позволяет сделать ОБОСНОВАННЫЕ выводы
• Контекст содержит СВЯЗАННЫЕ факты и данные
• Можно дать ПОЛЕЗНЫЙ ответ даже с ограниченными данными

ОТВЕЧАЙТЕ "Нет" ТОЛЬКО ЕСЛИ:
• Контекст ПОЛНОСТЬЮ не связан с темой вопроса
• Информация НЕ ПОМОЖЕТ даже частично ответить
• Контекст касается СОВЕРШЕННО ДРУГИХ предметов

ПРИМЕРЫ АНАЛИЗА:

Пример 1:
Контекст: "Налоговый кодекс РК устанавливает ставку подоходного налога 10%"
Вопрос: "Какая ставка подоходного налога?"
Анализ: Прямой ответ → Да

Пример 2:
Контекст: "В 2023 году объем инвестиций составил 2.5 млрд тенге"
Вопрос: "Какие налоги платят инвесторы?"
Анализ: Связанная тема, частичная релевантность → Да

Пример 3:
Контекст: "Рецепт приготовления борща включает свеклу и капусту"
Вопрос: "Какая ставка НДС на услуги?"
Анализ: Полностью разные темы → Нет

ВАШЕ РЕШЕНИЕ: Одно слово "Да" или "Нет"

КОНТЕКСТ: {context}
ВОПРОС: {query}
РЕШЕНИЕ:"""
            response = ollama.generate(
                model=self.model_name,
                prompt=router_prompt,
                options={
                    "temperature": 0.0,
                    "top_p": 0.1,
                    "num_predict": 10
                }
            )
            answer = response['response'].strip().lower()
            positive_indicators = ["да", "yes", "true", "1", "можно", "возможно", "есть", "имеется"]
            return any(indicator in answer for indicator in positive_indicators)
        except Exception as e:
            self.logger.error(f"Error in router decision: {str(e)}")
            return False
    def summarize_context(self, context: str, max_length: int = 1000) -> str:
        if len(context) <= max_length:
            return context
        try:
            summary_prompt = f"""РОЛЬ: Профессиональный аналитик-суммаризатор

ЗАДАЧА: Создать структурированное краткое изложение документа

ТРЕБОВАНИЯ:
• Максимум {max_length} символов
• Сохранить ВСЮ ключевую информацию
• Выделить основные факты, цифры, даты
• Убрать дублирования и избыточность
• Сохранить язык оригинального текста
• Структурировать по важности

СТРУКТУРА ИЗЛОЖЕНИЯ:
1. Главные факты и выводы
2. Ключевые данные (цифры, даты, имена)
3. Важные детали и контекст

ИСХОДНЫЙ ДОКУМЕНТ:
{context}

СТРУКТУРИРОВАННОЕ ИЗЛОЖЕНИЕ:"""
            response = ollama.generate(
                model=self.model_name,
                prompt=summary_prompt,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": max_length // 4
                }
            )
            return response['response'].strip()
        except Exception as e:
            self.logger.error(f"Error summarizing context: {str(e)}")
            return context[:max_length] + "..."
    def extract_key_topics(self, text: str) -> List[str]:
        try:
            topics_prompt = f"""РОЛЬ: Эксперт-аналитик семантических тем

ЗАДАЧА: Извлечь и классифицировать ключевые темы из документа

ТРЕБОВАНИЯ К ТЕМАМ:
• 5-10 наиболее важных тем
• Каждая тема: 1-3 ключевых слова
• Приоритет: существительные и профессиональные термины
• Избегать общих слов (процесс, система, документ)

ТИПЫ ТЕМ ДЛЯ ПОИСКА:
• Организации и учреждения
• Процедуры и регламенты  
• Правовые понятия и нормы
• Технические термины
• Численные показатели и даты
• Географические названия

АНАЛИЗИРУЕМЫЙ ТЕКСТ:
{text}

КЛЮЧЕВЫЕ ТЕМЫ (через запятую):"""
            response = ollama.generate(
                model=self.model_name,
                prompt=topics_prompt,
                options={
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "num_predict": 100
                }
            )
            topics_text = response['response'].strip()
            topics = [topic.strip() for topic in topics_text.split(',')]
            return topics[:10]
        except Exception as e:
            self.logger.error(f"Error extracting topics: {str(e)}")
            return []
    def get_model_info(self) -> Dict[str, Any]:
        try:
            models_response = ollama.list()
            if isinstance(models_response, dict) and 'models' in models_response:
                for model in models_response['models']:
                    if isinstance(model, dict) and model.get('name') == self.model_name:
                        return {
                            "name": model.get('name'),
                            "size": model.get('size', 'Unknown'),
                            "modified": model.get('modified_at', 'Unknown'),
                            "available": True
                        }
            return {
                "name": self.model_name,
                "available": False,
                "error": "Model not found in available models"
            }
        except Exception as e:
            return {
                "name": self.model_name, 
                "available": False,
                "error": str(e)
            }
    def test_connection(self) -> bool:
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt="Test",
                options={"num_predict": 5}
            )
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {str(e)}")
            self.logger.info("Убедитесь, что:")
            self.logger.info("1. Ollama запущен: ollama serve")
            self.logger.info("2. Модель загружена: ollama pull <model_name>")
            self.logger.info("3. Модель доступна: ollama list")
            return False
    def _clean_response(self, response: str) -> str:
        import re
        # Убираем внутренние размышления
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'\*думаю\*.*?\*/?думаю\*', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'\[думаю\].*?\[/?думаю\]', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Убираем форматирование источников
        cleaned = re.sub(r'\[Источник:.*?\]', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\[источник:.*?\]', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\[Source:.*?\]', '', cleaned, flags=re.IGNORECASE)
        
        # Убираем секции с источниками
        cleaned = re.sub(r'\*\*Источники:\*\*.*?(?=\n\n|\Z)', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'\*\*Sources:\*\*.*?(?=\n\n|\Z)', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'Источники:.*?(?=\n\n|\Z)', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'Sources:.*?(?=\n\n|\Z)', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Убираем строки со списками источников
        cleaned = re.sub(r'^[-•]\s*\[Источник:.*?\].*$', '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
        cleaned = re.sub(r'^[-•]\s*\[Source:.*?\].*$', '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
        
        # Очищаем лишние переносы строк
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()
        return cleaned