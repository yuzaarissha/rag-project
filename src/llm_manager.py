import ollama
from typing import Dict, Any, Optional, List
import json
import logging


class LLMManager:
    def __init__(self, model_name: str = "qwen2.5-coder:latest"):
        self .model_name = model_name
        self .logger = logging .getLogger(__name__)
        self .system_prompt = """РОЛЬ: Экспертный ассистент-аналитик документов

ВАША МИССИЯ:
Предоставлять точные, профессиональные ответы на основе загруженных документов с максимальной полезностью для пользователя.

ОСНОВНЫЕ ПРИНЦИПЫ:

АНАЛИЗ И СИНТЕЗ:
• Тщательно изучайте весь предоставленный контекст
• Объединяйте информацию из разных частей документов
• Выявляйте ключевые связи и закономерности
• Структурируйте ответ логично и последовательно

ПОШАГОВОЕ РАССУЖДЕНИЕ (Chain-of-Thought):
• При сложных вопросах рассуждайте пошагово
• Сначала определите ключевые факты из контекста
• Затем проанализируйте их взаимосвязи
• Сделайте обоснованные выводы на основе анализа
• Покажите логическую цепочку от фактов к ответу

ПРОВЕРКА КОНТЕКСТА (Context Validation):
• Проверяйте релевантность каждого факта из контекста для вопроса
• Используйте только информацию, прямо или косвенно связанную с вопросом
• Игнорируйте нерелевантные части контекста
• Всегда основывайте ответы на проверенных фактах из документов

ОЦЕНКА УВЕРЕННОСТИ (Uncertainty Reflection):
• Указывайте уровень уверенности в ответе (высокий/средний/низкий)
• При неполной информации четко обозначайте ограничения
• Если данных недостаточно, честно признавайте это
• Различайте факты и предположения в своих ответах

МНОГОЯЗЫЧНОСТЬ:
• Автоматически определяйте язык вопроса
• Отвечайте на том же языке (русский/английский)
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
• Выдумывать информацию, не содержащуюся в документах
• Использовать расплывчатые формулировки без обоснования

СТАНДАРТНЫЕ ОТВЕТЫ:
• При отсутствии информации: "Информация не найдена в предоставленных документах"
• При частичной информации: "На основе доступных данных могу сказать следующее... (уверенность: средняя/низкая)"
• При высокой уверенности: четко структурированный ответ с конкретными фактами

ЦЕЛЬ: Быть максимально полезным, точным и профессиональным помощником с прозрачной оценкой достоверности ответов."""

    def update_model(self, model_name: str) -> bool:
        try:
            test_response = ollama .generate(
                model=model_name,
                prompt="test",
                options={"num_predict": 1}
            )
            self .model_name = model_name
            return True
        except Exception as e:
            self .logger .error(
                f"Failed to update model to {model_name}: {str(e)}")
            return False

    def check_model_availability(self) -> bool:
        try:
            models_response = ollama .list()
            available_models = []
            if isinstance(models_response, dict) and 'models' in models_response:
                for model in models_response['models']:
                    if isinstance(model, dict) and 'name' in model:
                        available_models .append(model['name'])
            self .logger .info(f"Доступные модели: {available_models}")
            is_available = self .model_name in available_models
            if not is_available:
                self .logger .warning(
                    f"Модель {self .model_name} не найдена в списке доступных моделей")
                self .logger .info(
                    "Попробуйте использовать точное название из 'ollama list'")
            return is_available
        except Exception as e:
            self .logger .error(
                f"Ошибка проверки доступности моделей: {str(e)}")
            try:
                test_response = ollama .generate(
                    model=self .model_name,
                    prompt="test",
                    options={"num_predict": 1}
                )
                self .logger .info(
                    f"Модель {self .model_name} работает (прямое тестирование)")
                return True
            except Exception as test_error:
                self .logger .error(
                    f"Модель {self .model_name} недоступна: {test_error}")
                return False

    def generate_response(self, prompt: str, context: str = "", temperature: float = 0.2,
                          max_tokens: int = 2000, system_prompt_style: str = "Профессиональный") -> str:
        try:

            system_prompts = self ._get_system_prompts()
            style_prompt = system_prompts .get(
                system_prompt_style, system_prompts["Профессиональный"])

            combined_system_prompt = f"{self .system_prompt}\n\nСТИЛЬ ОБЩЕНИЯ: {style_prompt}"

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
            response = ollama .generate(
                model=self .model_name,
                prompt=full_prompt,
                system=combined_system_prompt,
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": max_tokens,
                    "repeat_penalty": 1.1
                }
            )
            clean_response = self ._clean_response(
                response['response'].strip())
            return clean_response
        except Exception as e:
            self .logger .error(f"Error generating response: {str(e)}")
            return "Извините, произошла ошибка при генерации ответа."

    def generate_router_decision(self, query: str, context: str) -> Dict[str, Any]:
        """Улучшенный маршрутизатор RAG 2024 с многоуровневой оценкой"""
        try:
            router_prompt = f"""РОЛЬ: Интеллектуальный маршрутизатор RAG-системы (версия 2024)

МИССИЯ: Определить релевантность и качество найденного контекста для ответа на вопрос пользователя

МНОГОУРОВНЕВАЯ ОЦЕНКА:

УРОВЕНЬ 1 - ПРЯМАЯ РЕЛЕВАНТНОСТЬ (100%):
• Контекст содержит ПРЯМОЙ и ПОЛНЫЙ ответ на вопрос
• Вся необходимая информация присутствует
• Можно дать исчерпывающий ответ

УРОВЕНЬ 2 - ВЫСОКАЯ РЕЛЕВАНТНОСТЬ (75-99%):
• Контекст содержит БОЛЬШУЮ ЧАСТЬ нужной информации
• Можно дать качественный ответ с незначительными пробелами
• Информация напрямую связана с вопросом

УРОВЕНЬ 3 - СРЕДНЯЯ РЕЛЕВАНТНОСТЬ (50-74%):
• Контекст содержит ЧАСТИЧНУЮ информацию по теме
• Можно дать ответ, но с оговорками об ограничениях
• Информация косвенно связана с вопросом

УРОВЕНЬ 4 - НИЗКАЯ РЕЛЕВАНТНОСТЬ (25-49%):
• Контекст содержит МИНИМАЛЬНУЮ полезную информацию
• Можно дать только общий или предварительный ответ
• Связь с вопросом слабая, но существует

УРОВЕНЬ 5 - НЕ РЕЛЕВАНТНО (0-24%):
• Контекст НЕ СОДЕРЖИТ полезной информации для ответа
• Темы не связаны или противоречат вопросу
• Невозможно дать содержательный ответ

ИНСТРУКЦИИ:
1. Определите уровень релевантности (1-5)
2. Укажите процент соответствия (0-100%)
3. Выделите полезные части контекста
4. Укажите что именно можно/нельзя ответить

КОНТЕКСТ: {context}

ВОПРОС: {query}

АНАЛИЗ:
Уровень релевантности: [1-5]
Процент соответствия: [0-100]%
Полезные части: [конкретные факты/фразы]
Возможности ответа: [что можно ответить]
Ограничения: [чего не хватает]

РЕШЕНИЕ: [Использовать/Отклонить контекст]"""

            response = ollama .generate(
                model=self .model_name,
                prompt=router_prompt,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 300
                }
            )

            result_text = response['response'].strip()

            relevance_level = 3
            relevance_percentage = 50
            decision = True

            lines = result_text .split('\n')
            for line in lines:
                if 'Уровень релевантности:' in line:
                    import re
                    match = re .search(r'(\d)', line)
                    if match:
                        relevance_level = int(match .group(1))

                if 'Процент соответствия:' in line:
                    match = re .search(r'(\d+)', line)
                    if match:
                        relevance_percentage = int(match .group(1))

                if 'РЕШЕНИЕ:' in line:
                    decision = 'Использовать' in line or 'использовать' in line

            if relevance_percentage < 25:
                decision = False

            if relevance_percentage >= 75:
                quality = "высокое"
            elif relevance_percentage >= 50:
                quality = "среднее"
            elif relevance_percentage >= 25:
                quality = "низкое"
            else:
                quality = "неприемлемое"

            return {
                "decision": decision,
                "relevance_level": relevance_level,
                "relevance_percentage": relevance_percentage,
                "quality": quality,
                "detailed_analysis": result_text,
                "should_use_context": decision
            }

        except Exception as e:
            self .logger .error(f"Error in advanced router decision: {str(e)}")
            return {
                "decision": True,
                "relevance_level": 3,
                "relevance_percentage": 50,
                "quality": "среднее",
                "detailed_analysis": "Ошибка анализа маршрутизатора",
                "should_use_context": True
            }

    def generate_simple_router_decision(self, query: str, context: str) -> bool:
        """Упрощенная версия маршрутизатора для обратной совместимости"""
        try:
            advanced_result = self .generate_router_decision(query, context)
            return advanced_result["decision"]
        except Exception as e:
            self .logger .error(f"Error in simple router decision: {str(e)}")
            return True

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
            response = ollama .generate(
                model=self .model_name,
                prompt=summary_prompt,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": max_length // 4
                }
            )
            return response['response'].strip()
        except Exception as e:
            self .logger .error(f"Error summarizing context: {str(e)}")
            return context[:max_length]+"..."

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
            response = ollama .generate(
                model=self .model_name,
                prompt=topics_prompt,
                options={
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "num_predict": 100
                }
            )
            topics_text = response['response'].strip()
            topics = [topic .strip()for topic in topics_text .split(',')]
            return topics[:10]
        except Exception as e:
            self .logger .error(f"Error extracting topics: {str(e)}")
            return []

    def evaluate_context_relevance(self, query: str, context: str) -> Dict[str, Any]:
        """Новый метод 2024: оценка релевантности контекста для RAGAS-подобной оценки"""
        try:
            relevance_prompt = f"""РОЛЬ: Эксперт по оценке релевантности контекста

ЗАДАЧА: Оценить какие предложения из контекста действительно полезны для ответа на вопрос

КРИТЕРИИ ОЦЕНКИ:
• ПРЯМАЯ СВЯЗЬ (3 балла): Предложение содержит прямой ответ на вопрос
• КОСВЕННАЯ СВЯЗЬ (2 балла): Предложение содержит связанную информацию
• ОБЩИЙ КОНТЕКСТ (1 балл): Предложение дает полезный background
• НЕРЕЛЕВАНТНО (0 баллов): Предложение не связано с вопросом

ИНСТРУКЦИИ:
1. Разбейте контекст на отдельные предложения
2. Оцените каждое предложение по шкале 0-3
3. Укажите общий процент релевантности контекста
4. Выделите наиболее полезные предложения

КОНТЕКСТ:
{context}

ВОПРОС: {query}

АНАЛИЗ РЕЛЕВАНТНОСТИ:
Предложение 1: [оценка 0-3] - [краткое обоснование]
Предложение 2: [оценка 0-3] - [краткое обоснование]
...

ОБЩАЯ РЕЛЕВАНТНОСТЬ: [процент]%
НАИБОЛЕЕ ПОЛЕЗНЫЕ ПРЕДЛОЖЕНИЯ: [номера предложений]"""

            response = ollama .generate(
                model=self .model_name,
                prompt=relevance_prompt,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 500
                }
            )

            result_text = response['response'].strip()

            relevance_percentage = 0
            lines = result_text .split('\n')
            for line in lines:
                if 'ОБЩАЯ РЕЛЕВАНТНОСТЬ' in line:
                    import re
                    match = re .search(r'(\d+)%', line)
                    if match:
                        relevance_percentage = int(match .group(1))
                        break

            return {
                "relevance_score": relevance_percentage / 100.0,
                "detailed_analysis": result_text,
                "is_relevant": relevance_percentage > 30
            }

        except Exception as e:
            self .logger .error(
                f"Error evaluating context relevance: {str(e)}")
            return {
                "relevance_score": 0.5,
                "detailed_analysis": "Ошибка оценки релевантности",
                "is_relevant": True
            }

    def assess_confidence(self, query: str, context: str, answer: str) -> Dict[str, Any]:
        """Новый метод 2024: оценка уверенности в ответе для Uncertainty Reflection"""
        try:
            confidence_prompt = f"""РОЛЬ: Аналитик достоверности ответов

ЗАДАЧА: Оценить уверенность в ответе на основе качества контекста и полноты информации

ШКАЛА УВЕРЕННОСТИ:
• ОЧЕНЬ ВЫСОКАЯ (90-100%): Контекст содержит прямые и полные ответы
• ВЫСОКАЯ (70-89%): Достаточно данных для обоснованных выводов
• СРЕДНЯЯ (50-69%): Частичная информация, есть пробелы
• НИЗКАЯ (30-49%): Минимальная информация, много предположений
• ОЧЕНЬ НИЗКАЯ (0-29%): Крайне недостаточно данных

КРИТЕРИИ ОЦЕНКИ:
1. Полнота информации в контексте
2. Прямота связи контекста с вопросом  
3. Качество и специфичность ответа
4. Наличие конкретных фактов и цифр
5. Согласованность информации

ВОПРОС: {query}

КОНТЕКСТ: {context}

ОТВЕТ: {answer}

АНАЛИЗ УВЕРЕННОСТИ:
1. Полнота контекста: [оценка и обоснование]
2. Качество ответа: [оценка и обоснование]
3. Конкретность данных: [оценка и обоснование]

ИТОГОВАЯ УВЕРЕННОСТЬ: [процент]%
ОБОСНОВАНИЕ: [краткое объяснение уровня уверенности]"""

            response = ollama .generate(
                model=self .model_name,
                prompt=confidence_prompt,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 400
                }
            )

            result_text = response['response'].strip()

            confidence_percentage = 50
            lines = result_text .split('\n')
            for line in lines:
                if 'ИТОГОВАЯ УВЕРЕННОСТЬ' in line:
                    import re
                    match = re .search(r'(\d+)%', line)
                    if match:
                        confidence_percentage = int(match .group(1))
                        break

            if confidence_percentage >= 90:
                level = "очень высокая"
            elif confidence_percentage >= 70:
                level = "высокая"
            elif confidence_percentage >= 50:
                level = "средняя"
            elif confidence_percentage >= 30:
                level = "низкая"
            else:
                level = "очень низкая"

            return {
                "confidence_score": confidence_percentage / 100.0,
                "confidence_level": level,
                "detailed_analysis": result_text,
                "should_warn_user": confidence_percentage < 50
            }

        except Exception as e:
            self .logger .error(f"Error assessing confidence: {str(e)}")
            return {
                "confidence_score": 0.5,
                "confidence_level": "средняя",
                "detailed_analysis": "Ошибка оценки уверенности",
                "should_warn_user": False
            }

    def get_model_info(self) -> Dict[str, Any]:
        try:
            models_response = ollama .list()
            if isinstance(models_response, dict) and 'models' in models_response:
                for model in models_response['models']:
                    if isinstance(model, dict) and model .get('name') == self .model_name:
                        return {
                            "name": model .get('name'),
                            "size": model .get('size', 'Unknown'),
                            "modified": model .get('modified_at', 'Unknown'),
                            "available": True
                        }
            return {
                "name": self .model_name,
                "available": False,
                "error": "Model not found in available models"
            }
        except Exception as e:
            return {
                "name": self .model_name,
                "available": False,
                "error": str(e)
            }

    def test_connection(self) -> bool:
        try:
            response = ollama .generate(
                model=self .model_name,
                prompt="Test",
                options={"num_predict": 5}
            )
            return True
        except Exception as e:
            self .logger .error(f"Connection test failed: {str(e)}")
            self .logger .info("Убедитесь, что:")
            self .logger .info("1. Ollama запущен: ollama serve")
            self .logger .info("2. Модель загружена: ollama pull <model_name>")
            self .logger .info("3. Модель доступна: ollama list")
            return False

    def _clean_response(self, response: str) -> str:
        import re
        cleaned = re .sub(r'<think>.*?</think>', '', response,
                          flags=re .DOTALL | re .IGNORECASE)
        cleaned = re .sub(r'\*думаю\*.*?\*/?думаю\*', '',
                          cleaned, flags=re .DOTALL | re .IGNORECASE)
        cleaned = re .sub(r'\[думаю\].*?\[/?думаю\]', '',
                          cleaned, flags=re .DOTALL | re .IGNORECASE)

        cleaned = re .sub(r'\[Источник:.*?\]', '',
                          cleaned, flags=re .IGNORECASE)
        cleaned = re .sub(r'\[источник:.*?\]', '',
                          cleaned, flags=re .IGNORECASE)
        cleaned = re .sub(r'\[Source:.*?\]', '', cleaned, flags=re .IGNORECASE)

        cleaned = re .sub(r'\*\*Источники:\*\*.*?(?=\n\n|\Z)',
                          '', cleaned, flags=re .DOTALL | re .IGNORECASE)
        cleaned = re .sub(r'\*\*Sources:\*\*.*?(?=\n\n|\Z)', '',
                          cleaned, flags=re .DOTALL | re .IGNORECASE)
        cleaned = re .sub(r'Источники:.*?(?=\n\n|\Z)', '',
                          cleaned, flags=re .DOTALL | re .IGNORECASE)
        cleaned = re .sub(r'Sources:.*?(?=\n\n|\Z)', '',
                          cleaned, flags=re .DOTALL | re .IGNORECASE)

        cleaned = re .sub(r'^[-•]\s*\[Источник:.*?\].*$',
                          '', cleaned, flags=re .MULTILINE | re .IGNORECASE)
        cleaned = re .sub(r'^[-•]\s*\[Source:.*?\].*$', '',
                          cleaned, flags=re .MULTILINE | re .IGNORECASE)

        cleaned = re .sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned .strip()
        return cleaned

    def _get_system_prompts(self) -> Dict[str, str]:
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
