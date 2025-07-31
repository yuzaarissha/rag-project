from typing import Dict, Any, Optional, List
import logging
from .llm_manager import LLMManager
from .vector_store import VectorStore


class SmartRouter:
    def __init__(self, llm_manager: LLMManager, vector_store: VectorStore):
        self .logger = logging .getLogger(__name__)
        self .llm_manager = llm_manager
        self .vector_store = vector_store
        self .confidence_threshold = 0.15

    def analyze_query(self, query: str) -> Dict[str, Any]:
        query_analysis = {
            "length": len(query),
            "word_count": len(query .split()),
            "is_question": query .strip().endswith('?'),
            "language": self ._detect_language(query),
            "keywords": self ._extract_keywords(query),
            "query_type": self ._classify_query_type(query)
        }
        return query_analysis

    def _detect_language(self, text: str) -> str:
        russian_chars = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
        text_lower = text .lower()
        russian_count = sum(1 for char in text_lower if char in russian_chars)
        if russian_count > 0:
            return "russian"
        else:
            return "english"

    def _extract_keywords(self, query: str) -> List[str]:
        stop_words = {
            'что', 'как', 'где', 'когда', 'почему', 'какой', 'какая', 'какие',
            'это', 'то', 'его', 'её', 'их', 'и', 'или', 'но', 'а', 'в', 'на',
            'по', 'за', 'для', 'с', 'из', 'к', 'от', 'о', 'об', 'про',
            'what', 'how', 'where', 'when', 'why', 'which', 'is', 'are', 'the', 'a', 'an'
        }
        words = query .lower().split()
        keywords = [word .strip('.,!?;:')for word in words if word .strip(
            '.,!?;:')not in stop_words]
        return keywords

    def _classify_query_type(self, query: str) -> str:
        query_lower = query .lower()
        if any(word in query_lower for word in ['что такое', 'определение', 'объясни', 'what is', 'define']):
            return "definition"
        elif any(word in query_lower for word in ['как', 'how', 'способ', 'method']):
            return "instruction"
        elif any(word in query_lower for word in ['где', 'where', 'location']):
            return "location"
        elif any(word in query_lower for word in ['когда', 'when', 'время', 'time']):
            return "temporal"
        elif any(word in query_lower for word in ['почему', 'why', 'причина', 'reason']):
            return "causal"
        elif any(word in query_lower for word in ['сколько', 'how many', 'количество', 'count']):
            return "quantitative"
        else:
            return "general"

    def route_query(self, query: str, initial_search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        query_analysis = self .analyze_query(query)
        if not initial_search_results:
            return {
                "can_answer": False,
                "reason": "No relevant documents found",
                "context": "",
                "confidence": 0.0,
                "query_analysis": query_analysis
            }
        best_distance = min(result['distance']
                            for result in initial_search_results)
        confidence = max(0, 1 - best_distance)
        context = "\n\n".join([result['content']
                               for result in initial_search_results])
        if confidence >= 0.4:
            can_answer = True
        elif confidence >= 0.2 and len(context .strip()) > 100:
            can_answer = True
        elif confidence >= self .confidence_threshold and len(context .strip()) > 50:
            can_answer = self .llm_manager .generate_router_decision(
                query, context)
        elif len(context .strip()) > 200:
            can_answer = True
        else:
            can_answer = False
        routing_result = {
            "can_answer": can_answer,
            "confidence": confidence,
            "context": context,
            "num_sources": len(initial_search_results),
            "query_analysis": query_analysis,
            "reasoning": self ._generate_routing_reasoning(query_analysis, confidence, can_answer)
        }
        return routing_result

    def _generate_routing_reasoning(self, query_analysis: Dict[str, Any], confidence: float, can_answer: bool) -> str:
        language = query_analysis .get('language', 'unknown')
        query_type = query_analysis .get('query_type', 'general')
        if can_answer:
            if language == 'russian':
                return f"Найдены релевантные документы (уверенность: {confidence:.2f}). Тип вопроса: {query_type}. Могу ответить на основе локальных данных."
            else:
                return f"Relevant documents found (confidence: {confidence:.2f}). Query type: {query_type}. Can answer from local data."
        else:
            if language == 'russian':
                return f"Недостаточно релевантных данных (уверенность: {confidence:.2f}). Тип вопроса: {query_type}. Требуется дополнительная информация."
            else:
                return f"Insufficient relevant data (confidence: {confidence:.2f}). Query type: {query_type}. Additional information needed."

    def enhance_context(self, context: str, query: str) -> str:
        try:
            if len(context) > 3000:
                context = self .llm_manager .summarize_context(
                    context, max_length=2000)
            return context
        except Exception as e:
            self .logger .error(f"Error enhancing context: {str(e)}")
            return context

    def get_routing_metrics(self) -> Dict[str, Any]:
        return {
            "confidence_threshold": self .confidence_threshold,
            "total_queries": 0,
            "successful_routes": 0,
            "failed_routes": 0
        }

    def update_confidence_threshold(self, new_threshold: float) -> None:
        if 0.0 <= new_threshold <= 1.0:
            self .confidence_threshold = new_threshold
        else:
            self .logger .error(
                "Confidence threshold must be between 0.0 and 1.0")

    def explain_routing_decision(self, routing_result: Dict[str, Any]) -> str:
        can_answer = routing_result .get('can_answer', False)
        confidence = routing_result .get('confidence', 0.0)
        num_sources = routing_result .get('num_sources', 0)
        query_analysis = routing_result .get('query_analysis', {})
        explanation = f"""
**Решение маршрутизатора:**
- Можно ответить: {'Да'if can_answer else 'Нет'}
- Уверенность: {confidence:.2f}
- Количество источников: {num_sources}
- Тип вопроса: {query_analysis .get('query_type', 'неизвестно')}
- Язык: {query_analysis .get('language', 'неизвестно')}
- Ключевые слова: {','.join(query_analysis .get('keywords', []))}

**Обоснование:** {routing_result .get('reasoning', 'Нет обоснования')}
        """
        return explanation .strip()
