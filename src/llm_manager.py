import ollama
import streamlit as st
from typing import Dict, Any, Optional, List
import json
class LLMManager:
    def __init__(self, model_name: str = "llama3.2:latest"):
        self.model_name = model_name
        self.system_prompt = """Вы являетесь экспертным помощником для ответов на вопросы на основе предоставленного контекста. 

ОСНОВНЫЕ ПРИНЦИПЫ:
1. ПРИОРИТЕТ: Максимально используйте предоставленный контекст для формирования полного и точного ответа
2. АГРЕССИВНЫЙ ПОИСК: Ищите любую релевантную информацию в контексте, даже если она не очень очевидно связана с вопросом
3. СИНТЕЗ: Объединяйте информацию из разных частей контекста для создания комплексного ответа
4. СТРУКТУРА: Организуйте ответ логично - основная информация, детали, примеры
5. ЯЗЫК: Отвечайте на том же языке, что и вопрос (русский/казахский/английский)
6. ПОЛНОТА: Если в контексте есть связанная информация, включите её в ответ
7. ЧЕСТНОСТЬ: Если контекст действительно не содержит информации для ответа, сообщите об этом

ЗАПРЕЩЕНО:
- Показывать внутренние размышления или теги <think>
- Отвечать информацией не из контекста
- Давать слишком краткие ответы, если в контексте есть больше деталей

ФОРМАТ ОТВЕТА: Прямой, структурированный ответ без префиксов типа "На основе контекста..."""
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
            st.error(f"Failed to update model to {model_name}: {str(e)}")
            return False
    def check_model_availability(self) -> bool:
        try:
            models_response = ollama.list()
            available_models = []
            if isinstance(models_response, dict) and 'models' in models_response:
                for model in models_response['models']:
                    if isinstance(model, dict) and 'name' in model:
                        available_models.append(model['name'])
            st.info(f"Доступные модели: {available_models}")
            is_available = self.model_name in available_models
            if not is_available:
                st.warning(f"Модель {self.model_name} не найдена в списке доступных моделей")
                st.info("Попробуйте использовать точное название из 'ollama list'")
            return is_available
        except Exception as e:
            st.error(f"Ошибка проверки доступности моделей: {str(e)}")
            try:
                test_response = ollama.generate(
                    model=self.model_name,
                    prompt="test",
                    options={"num_predict": 1}
                )
                st.success(f"Модель {self.model_name} работает (прямое тестирование)")
                return True
            except Exception as test_error:
                st.error(f"Модель {self.model_name} недоступна: {test_error}")
                return False
    def generate_response(self, prompt: str, context: str = "", temperature: float = 0.2) -> str:
        try:
            if context:
                full_prompt = f"""Контекст: {context}

Вопрос: {prompt}

Ответ:"""
            else:
                full_prompt = prompt
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
            st.error(f"Error generating response: {str(e)}")
            return "Извините, произошла ошибка при генерации ответа."
    def generate_router_decision(self, query: str, context: str) -> bool:
        try:
            router_prompt = f"""Роль: Системный маршрутизатор
Задача: Определить, может ли система ответить на вопрос пользователя на основе предоставленного текста.

Инструкции:
- Если в тексте есть ЛЮБАЯ информация, связанная с вопросом, отвечайте "Да"
- Если текст может дать хотя бы частичный ответ, отвечайте "Да"
- Отвечайте "Нет" только если текст СОВСЕМ не связан с вопросом
- Ваш ответ должен содержать только одно слово: "Да" или "Нет"

Примеры:
Текст: "Столица Франции - Париж."
Вопрос: "Какая столица Франции?"
Ответ: Да

Текст: "Население США составляет более 330 миллионов человек."
Вопрос: "Какое население Китая?"
Ответ: Нет

Текст: {context}
Вопрос: {query}
Ответ:"""
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
            st.error(f"Error in router decision: {str(e)}")
            return False
    def summarize_context(self, context: str, max_length: int = 1000) -> str:
        if len(context) <= max_length:
            return context
        try:
            summary_prompt = f"""Задача: Кратко изложить основные моменты из следующего текста.
Требования:
- Сохранить ключевую информацию
- Убрать повторения
- Максимальная длина: {max_length} символов
- Язык ответа: такой же, как в исходном тексте

Текст:
{context}

Краткое изложение:"""
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
            st.error(f"Error summarizing context: {str(e)}")
            return context[:max_length] + "..."
    def extract_key_topics(self, text: str) -> List[str]:
        try:
            topics_prompt = f"""Задача: Извлечь ключевые темы из следующего текста.
Требования:
- Выделить 5-10 основных тем
- Каждая тема должна быть 1-3 слова
- Ответ в формате: тема1, тема2, тема3...

Текст:
{text}

Ключевые темы:"""
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
            st.error(f"Error extracting topics: {str(e)}")
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
            st.error(f"Connection test failed: {str(e)}")
            st.info("Убедитесь, что:")
            st.info("1. Ollama запущен: ollama serve")
            st.info("2. Модель загружена: ollama pull <model_name>")
            st.info("3. Модель доступна: ollama list")
            return False
    def _clean_response(self, response: str) -> str:
        import re
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'\*думаю\*.*?\*/?думаю\*', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'\[думаю\].*?\[/?думаю\]', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()
        return cleaned