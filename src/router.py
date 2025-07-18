"""
Smart Router for RAG system
Determines whether to use local knowledge or search for more information
"""

from typing import Dict, Any, Optional, List
import streamlit as st
from .llm_manager import LLMManager
from .vector_store import VectorStore


class SmartRouter:
    def __init__(self, llm_manager: LLMManager, vector_store: VectorStore):
        """
        Initialize Smart Router
        
        Args:
            llm_manager: LLM Manager instance
            vector_store: Vector Store instance
        """
        self.llm_manager = llm_manager
        self.vector_store = vector_store
        self.confidence_threshold = 0.3  # Similarity threshold for routing
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze user query to understand its characteristics
        
        Args:
            query: User question
            
        Returns:
            Query analysis results
        """
        query_analysis = {
            "length": len(query),
            "word_count": len(query.split()),
            "is_question": query.strip().endswith('?'),
            "language": self._detect_language(query),
            "keywords": self._extract_keywords(query),
            "query_type": self._classify_query_type(query)
        }
        
        return query_analysis
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected language
        """
        # Simple heuristic - can be improved with proper language detection
        russian_chars = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
        kazakh_chars = set('әғқңөұүһі')
        
        text_lower = text.lower()
        russian_count = sum(1 for char in text_lower if char in russian_chars)
        kazakh_count = sum(1 for char in text_lower if char in kazakh_chars)
        
        if kazakh_count > 0:
            return "kazakh"
        elif russian_count > 0:
            return "russian"
        else:
            return "english"
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from query
        
        Args:
            query: User question
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction - remove common words
        stop_words = {
            'что', 'как', 'где', 'когда', 'почему', 'какой', 'какая', 'какие',
            'это', 'то', 'его', 'её', 'их', 'и', 'или', 'но', 'а', 'в', 'на',
            'по', 'за', 'для', 'с', 'из', 'к', 'от', 'о', 'об', 'про',
            'what', 'how', 'where', 'when', 'why', 'which', 'is', 'are', 'the', 'a', 'an'
        }
        
        words = query.lower().split()
        keywords = [word.strip('.,!?;:') for word in words if word.strip('.,!?;:') not in stop_words]
        
        return keywords
    
    def _classify_query_type(self, query: str) -> str:
        """
        Classify the type of query
        
        Args:
            query: User question
            
        Returns:
            Query type
        """
        query_lower = query.lower()
        
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
        """
        Route query based on analysis and initial search results
        
        Args:
            query: User question
            initial_search_results: Initial search results from vector store
            
        Returns:
            Routing decision with context
        """
        # Analyze query
        query_analysis = self.analyze_query(query)
        
        # Check if we have relevant documents
        if not initial_search_results:
            return {
                "can_answer": False,
                "reason": "No relevant documents found",
                "context": "",
                "confidence": 0.0,
                "query_analysis": query_analysis
            }
        
        # Calculate relevance confidence
        avg_distance = sum(result['distance'] for result in initial_search_results) / len(initial_search_results)
        confidence = max(0, 1 - avg_distance)  # Convert distance to confidence
        
        # Get context from search results
        context = "\n\n".join([result['content'] for result in initial_search_results])
        
        # Use LLM to make routing decision
        if confidence >= self.confidence_threshold:
            can_answer = self.llm_manager.generate_router_decision(query, context)
        else:
            can_answer = False
        
        routing_result = {
            "can_answer": can_answer,
            "confidence": confidence,
            "context": context,
            "num_sources": len(initial_search_results),
            "query_analysis": query_analysis,
            "reasoning": self._generate_routing_reasoning(query_analysis, confidence, can_answer)
        }
        
        return routing_result
    
    def _generate_routing_reasoning(self, query_analysis: Dict[str, Any], confidence: float, can_answer: bool) -> str:
        """
        Generate human-readable reasoning for routing decision
        
        Args:
            query_analysis: Analysis of the query
            confidence: Confidence score
            can_answer: Whether we can answer
            
        Returns:
            Reasoning text
        """
        language = query_analysis.get('language', 'unknown')
        query_type = query_analysis.get('query_type', 'general')
        
        if can_answer:
            if language == 'russian':
                return f"Найдены релевантные документы (уверенность: {confidence:.2f}). Тип вопроса: {query_type}. Могу ответить на основе локальных данных."
            elif language == 'kazakh':
                return f"Сәйкес құжаттар табылды (сенімділік: {confidence:.2f}). Сұрақ түрі: {query_type}. Жергілікті деректер негізінде жауап бере аламын."
            else:
                return f"Relevant documents found (confidence: {confidence:.2f}). Query type: {query_type}. Can answer from local data."
        else:
            if language == 'russian':
                return f"Недостаточно релевантных данных (уверенность: {confidence:.2f}). Тип вопроса: {query_type}. Требуется дополнительная информация."
            elif language == 'kazakh':
                return f"Сәйкес деректер жеткіліксіз (сенімділік: {confidence:.2f}). Сұрақ түрі: {query_type}. Қосымша ақпарат қажет."
            else:
                return f"Insufficient relevant data (confidence: {confidence:.2f}). Query type: {query_type}. Additional information needed."
    
    def enhance_context(self, context: str, query: str) -> str:
        """
        Enhance context by removing irrelevant parts and highlighting important information
        
        Args:
            context: Original context
            query: User query
            
        Returns:
            Enhanced context
        """
        try:
            # If context is too long, summarize it
            if len(context) > 2000:
                context = self.llm_manager.summarize_context(context, max_length=1500)
            
            return context
            
        except Exception as e:
            st.error(f"Error enhancing context: {str(e)}")
            return context
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about routing decisions
        
        Returns:
            Routing metrics
        """
        # This would be enhanced with actual metrics tracking in production
        return {
            "confidence_threshold": self.confidence_threshold,
            "total_queries": 0,  # Would track in session state
            "successful_routes": 0,  # Would track in session state
            "failed_routes": 0  # Would track in session state
        }
    
    def update_confidence_threshold(self, new_threshold: float) -> None:
        """
        Update confidence threshold for routing
        
        Args:
            new_threshold: New threshold value (0.0 to 1.0)
        """
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
        else:
            st.error("Confidence threshold must be between 0.0 and 1.0")
    
    def explain_routing_decision(self, routing_result: Dict[str, Any]) -> str:
        """
        Provide detailed explanation of routing decision
        
        Args:
            routing_result: Result from route_query
            
        Returns:
            Detailed explanation
        """
        can_answer = routing_result.get('can_answer', False)
        confidence = routing_result.get('confidence', 0.0)
        num_sources = routing_result.get('num_sources', 0)
        query_analysis = routing_result.get('query_analysis', {})
        
        explanation = f"""
**Решение маршрутизатора:**
- Можно ответить: {'Да' if can_answer else 'Нет'}
- Уверенность: {confidence:.2f}
- Количество источников: {num_sources}
- Тип вопроса: {query_analysis.get('query_type', 'неизвестно')}
- Язык: {query_analysis.get('language', 'неизвестно')}
- Ключевые слова: {', '.join(query_analysis.get('keywords', []))}

**Обоснование:** {routing_result.get('reasoning', 'Нет обоснования')}
        """
        
        return explanation.strip()