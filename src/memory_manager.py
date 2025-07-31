from typing import Dict, Any, List, Optional, Tuple, Set
import json
import logging
from datetime import datetime, timedelta
from collections import deque, Counter
import re
from difflib import SequenceMatcher
from functools import lru_cache
import hashlib


class ConversationMemory:

    def __init__(self, max_history: int = 10, context_window: int = 3):
        self .max_history = max_history
        self .context_window = context_window
        self .conversation_history = deque(maxlen=max_history)
        self .current_session_id = None
        self .logger = logging .getLogger(__name__)

        self .topic_evolution = []
        self .semantic_clusters = {}
        self .conversation_embeddings = []

        self ._context_cache = {}
        self ._similarity_cache = {}
        self ._topic_cache = {}
        self .advanced_stats = {
            "semantic_similarity_used": 0,
            "topic_transitions_detected": 0,
            "context_summarizations": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        self .current_topics = set()
        self .topic_transitions = []
        self .topic_weights = {}
        self .topic_clusters = {}
        self .stats = {
            "total_interactions": 0,
            "context_used_count": 0,
            "topic_switches": 0,
            "average_context_length": 0.0
        }

    def start_new_session(self, session_id: str):
        self .current_session_id = session_id
        self .conversation_history .clear()
        self .current_topics .clear()
        self .topic_transitions .clear()
        self .logger .info(f"Started new conversation session: {session_id}")

    def add_interaction(self, query: str, response: str, sources: List[Dict[str, Any]],
                        metadata: Dict[str, Any] = None):
        try:
            query_topics = self ._extract_topics_with_weights(query)
            response_topics = self ._extract_topics_with_weights(response)
            all_topics = {**query_topics, **response_topics}
            for topic, weight in all_topics .items():
                if topic in self .topic_weights:
                    self .topic_weights[topic] = (
                        self .topic_weights[topic]+weight)/2
                else:
                    self .topic_weights[topic] = weight
            interaction = {
                "timestamp": datetime .now().isoformat(),
                "session_id": self .current_session_id,
                "query": query,
                "response": response,
                "sources": sources,
                "query_topics": set(query_topics .keys()),
                "response_topics": set(response_topics .keys()),
                "topic_weights": all_topics,
                "interaction_length": len(query)+len(response),
                "source_count": len(sources),
                "metadata": metadata or {},
                "semantic_hash": self ._generate_semantic_hash(query, response)
            }

            self .conversation_history .append(interaction)
            if len(self .conversation_embeddings) >= self .max_history:
                self .conversation_embeddings .pop(0)
            self .conversation_embeddings .append(interaction["semantic_hash"])
            new_topics = set(all_topics .keys())
            topic_similarity = self ._calculate_topic_similarity(
                self .current_topics, new_topics)

            if topic_similarity < 0.3:
                if self .current_topics:
                    transition = {
                        "from": list(self .current_topics),
                        "to": list(new_topics),
                        "similarity": topic_similarity,
                        "transition_type": self ._classify_topic_transition(self .current_topics, new_topics),
                        "timestamp": datetime .now().isoformat()
                    }
                    self .topic_transitions .append(transition)
                    self .stats["topic_switches"] += 1
                    self .advanced_stats["topic_transitions_detected"] += 1
                self ._update_topic_clusters(self .current_topics, new_topics)

            self .current_topics = new_topics

            self .stats["total_interactions"] += 1
            self .logger .info(
                f"Added interaction to memory. Topics: {list(new_topics)[:5]}...")

        except Exception as e:
            self .logger .error(
                f"Failed to add interaction to memory: {str(e)}")

    def _generate_semantic_hash(self, query: str, response: str) -> str:
        combined_text = f"{query} {response}".lower()
        key_terms = sorted(self ._extract_topics(combined_text))
        return hashlib .md5("".join(key_terms[:10]).encode()).hexdigest()

    def _calculate_topic_similarity(self, topics1: Set[str], topics2: Set[str]) -> float:
        if not topics1 or not topics2:
            return 0.0
        intersection = len(topics1 .intersection(topics2))
        union = len(topics1 .union(topics2))
        jaccard = intersection / union if union > 0 else 0.0
        semantic_bonus = 0.0
        for topic1 in topics1:
            for topic2 in topics2:
                if topic1 != topic2:
                    similarity = SequenceMatcher(None, topic1, topic2).ratio()
                    if similarity > 0.6:
                        semantic_bonus += similarity * 0.1

        return min(jaccard + semantic_bonus, 1.0)

    def _classify_topic_transition(self, old_topics: Set[str], new_topics: Set[str]) -> str:
        if not old_topics:
            return "initial"

        overlap_ratio = len(old_topics .intersection(
            new_topics))/len(old_topics .union(new_topics))

        if overlap_ratio > 0.7:
            return "continuation"
        elif overlap_ratio > 0.3:
            return "evolution"
        else:
            return "shift"

    def _update_topic_clusters(self, old_topics: Set[str], new_topics: Set[str]):
        try:
            for old_topic in old_topics:
                for new_topic in new_topics:
                    if old_topic != new_topic:
                        cluster_key = tuple(sorted([old_topic, new_topic]))
                        if cluster_key in self .topic_clusters:
                            self .topic_clusters[cluster_key] += 1
                        else:
                            self .topic_clusters[cluster_key] = 1
        except Exception as e:
            self .logger .warning(f"Topic clustering failed: {str(e)}")

    def _calculate_interaction_relevance(self, current_query: str, current_topics: Dict[str, float],
                                         current_semantic_hash: str, interaction: Dict[str, Any]) -> Dict[str, float]:
        try:
            interaction_topics = interaction["query_topics"].union(
                interaction["response_topics"])
            topic_similarity = self ._calculate_topic_similarity(
                set(current_topics .keys()), interaction_topics)
            interaction_hash = interaction .get("semantic_hash", "")
            semantic_similarity = (
                0.5 if interaction_hash == current_semantic_hash and interaction_hash else
                self ._calculate_semantic_similarity(
                    current_query, interaction["query"])
            )
            recency_score = self ._calculate_recency_score(
                interaction["timestamp"])
            source_overlap = self ._calculate_source_overlap(
                current_query, interaction)

            return {
                "topic_similarity": topic_similarity,
                "semantic_similarity": semantic_similarity,
                "recency_score": recency_score,
                "source_overlap": source_overlap
            }

        except Exception as e:
            self .logger .warning(f"Relevance calculation failed: {str(e)}")
            return {"topic_similarity": 0.0, "semantic_similarity": 0.0, "recency_score": 0.0, "source_overlap": 0.0}

    @lru_cache(maxsize=200)
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        try:
            self .advanced_stats["semantic_similarity_used"] += 1
            words1 = set(text1 .lower().split())
            words2 = set(text2 .lower().split())

            if not words1 or not words2:
                return 0.0

            intersection = len(words1 .intersection(words2))
            union = len(words1 .union(words2))
            jaccard = intersection / union if union > 0 else 0.0
            length_ratio = min(len(text1), len(text2)) / \
                max(len(text1), len(text2))

            return jaccard * 0.8 + length_ratio * 0.2

        except Exception as e:
            self .logger .warning(
                f"Semantic similarity calculation failed: {str(e)}")
            return 0.0

    def _calculate_source_overlap(self, current_query: str, interaction: Dict[str, Any]) -> float:
        try:
            sources = interaction .get("sources", [])
            return 0.1 if sources else 0.0
        except Exception:
            return 0.0

    def _build_smart_context(self, relevant_interactions: List[Dict[str, Any]], max_length: int) -> str:
        try:
            if not relevant_interactions:
                return ""

            context_parts = []
            current_length = 0
            for item in relevant_interactions:
                interaction = item["interaction"]
                relevance = item["relevance"]
                response_preview_length = min(
                    300 if relevance > 0.5 else 150, len(interaction["response"]))

                context_part = f"Предыдущий вопрос: {interaction['query']}\n"
                context_part += f"Ответ: {interaction['response'][:response_preview_length]}"
                if len(interaction["response"]) > response_preview_length:
                    context_part += "..."
                context_part += "\n"
                if relevance > 0.4 and interaction["sources"]:
                    source_files = [s .get("filename", "unknown")
                                    for s in interaction["sources"][:2]]
                    context_part += f"Источники: {','.join(source_files)}\n"

                context_part += "---\n"

                if current_length + len(context_part) <= max_length:
                    context_parts .append(context_part)
                    current_length += len(context_part)
                else:
                    if len(context_parts) == 0:
                        context_parts .append(
                            context_part[:max_length - 50]+"...\n")
                    break
            context = "КОНТЕКСТ РАЗГОВОРА:\n" + \
                "".join(context_parts)if context_parts else ""

            if len(context) > max_length * 0.9:
                context = self ._summarize_context(context, max_length)
                self .advanced_stats["context_summarizations"] += 1

            return context

        except Exception as e:
            self .logger .error(f"Smart context building failed: {str(e)}")
            return ""

    def _summarize_context(self, context: str, max_length: int) -> str:
        try:
            lines = context .split('\n')
            important_lines = []

            for line in lines:
                if any(keyword in line .lower()for keyword in ['вопрос:', 'ответ:', 'источники:']):
                    important_lines .append(line)

            summarized = "\n".join(important_lines)

            if len(summarized) > max_length:
                summarized = summarized[:max_length -
                                        50]+"\n[Контекст сокращен...]\n"

            return summarized

        except Exception as e:
            self .logger .warning(f"Context summarization failed: {str(e)}")
            return context[:max_length]

    def _analyze_conversation_flow(self, current_query: str, response: str) -> List[str]:
        try:
            suggestions = []
            if any(word in current_query .lower()for word in ['что такое', 'определение', 'объясни']):
                suggestions .extend([
                    "Какие есть примеры этого?",
                    "Как это применяется на практике?",
                    "С чем это связано?"
                ])
            elif any(word in current_query .lower()for word in ['как', 'процедура', 'алгоритм']):
                suggestions .extend([
                    "Какие есть альтернативные способы?",
                    "Какие могут быть проблемы?",
                    "Кто отвечает за это?"
                ])
            elif any(word in current_query .lower()for word in ['сколько', 'количество', 'число']):
                suggestions .extend([
                    "Как изменялись эти показатели?",
                    "С чем сравнить эти цифры?",
                    "Какие факторы влияют на это?"
                ])

            return suggestions[:2]

        except Exception as e:
            self .logger .warning(
                f"Conversation flow analysis failed: {str(e)}")
            return []

    def _suggest_topic_evolution(self) -> List[str]:
        try:
            suggestions = []

            if len(self .topic_transitions) > 1:
                topic_counts = Counter()
                for transition in self .topic_transitions[-3:]:
                    topic_counts .update(transition["to"])
                common_topics = topic_counts .most_common(2)
                for topic, count in common_topics:
                    if count > 1:
                        suggestions .append(
                            f"Какие еще аспекты {topic} важны?")

            return suggestions[:1]

        except Exception as e:
            self .logger .warning(f"Topic evolution analysis failed: {str(e)}")
            return []

    def _find_advanced_question_patterns(self) -> List[str]:
        try:
            if len(self .conversation_history) < 2:
                return []
            recent_queries = [interaction["query"]
                              for interaction in list(self .conversation_history)[-5:]]
            pattern_suggestions = {
                "definition": ["Что означает этот термин?", "Дайте определение"],
                "procedure": ["Опишите процедуру", "Какие этапы включает?"],
                "comparison": ["В чем разница?", "Сравните варианты"],
                "examples": ["Приведите конкретные примеры", "Какие есть случаи?"],
                "causes": ["Какие причины этого?", "Что влияет на это?"],
                "consequences": ["К чему это приводит?", "Какие последствия?"]
            }
            pattern_counts = {pattern: 0 for pattern in pattern_suggestions}

            for query in recent_queries:
                query_lower = query .lower()
                if any(word in query_lower for word in ['что такое', 'определение', 'означает']):
                    pattern_counts["definition"] += 1
                elif any(word in query_lower for word in ['как', 'процедура', 'этапы']):
                    pattern_counts["procedure"] += 1
                elif any(word in query_lower for word in ['разница', 'отличие', 'сравн']):
                    pattern_counts["comparison"] += 1
                elif any(word in query_lower for word in ['пример', 'случай']):
                    pattern_counts["examples"] += 1
                elif any(word in query_lower for word in ['почему', 'причина']):
                    pattern_counts["causes"] += 1
            most_common_pattern = max(pattern_counts, key=pattern_counts .get)
            if pattern_counts[most_common_pattern] > 0:
                return pattern_suggestions[most_common_pattern][:2]

            return []

        except Exception as e:
            self .logger .warning(
                f"Advanced pattern analysis failed: {str(e)}")
            return []

    def _rank_and_deduplicate_suggestions(self, suggestions: List[str], current_topics: Dict[str, float]) -> List[str]:
        try:
            seen = set()
            unique_suggestions = []

            for suggestion in suggestions:
                suggestion_clean = suggestion .strip().lower()
                if suggestion_clean not in seen and len(suggestion .strip()) > 5:
                    seen .add(suggestion_clean)
                    unique_suggestions .append(suggestion .strip())
            ranked_suggestions = []
            topic_keywords = set(current_topics .keys())
            for suggestion in unique_suggestions:
                suggestion_words = set(suggestion .lower().split())
                if topic_keywords .intersection(suggestion_words):
                    ranked_suggestions .append(suggestion)
            for suggestion in unique_suggestions:
                if suggestion not in ranked_suggestions:
                    ranked_suggestions .append(suggestion)

            return ranked_suggestions

        except Exception as e:
            self .logger .warning(f"Suggestion ranking failed: {str(e)}")
            return list(dict .fromkeys(suggestions))

    def _calculate_avg_interaction_length(self) -> float:
        if not self .conversation_history:
            return 0.0

        total_length = sum(
            len(interaction["query"])+len(interaction["response"])
            for interaction in self .conversation_history
        )
        return total_length / len(self .conversation_history)

    def _calculate_topic_coherence(self) -> float:
        if len(self .topic_transitions) < 2:
            return 1.0
        coherence_scores = []
        for i in range(len(self .topic_transitions)-1):
            current_topics = set(self .topic_transitions[i]["to"])
            next_topics = set(self .topic_transitions[i + 1]["to"])
            similarity = self ._calculate_topic_similarity(
                current_topics, next_topics)
            coherence_scores .append(similarity)

        return sum(coherence_scores)/len(coherence_scores)if coherence_scores else 1.0

    def get_conversation_context(self, current_query: str, max_context_length: int = 1500) -> str:
        try:
            if not self .conversation_history:
                return ""
            cache_key = hashlib .md5(
                f"{current_query}_{max_context_length}".encode()).hexdigest()
            if cache_key in self ._context_cache:
                self .advanced_stats["cache_hits"] += 1
                return self ._context_cache[cache_key]

            self .advanced_stats["cache_misses"] += 1

            current_topics = self ._extract_topics_with_weights(current_query)
            current_semantic_hash = self ._generate_semantic_hash(
                current_query, "")

            relevant_interactions = []
            recent_interactions = list(
                self .conversation_history)[-min(self .context_window * 2, len(self .conversation_history)):]

            for interaction in recent_interactions:
                relevance_factors = self ._calculate_interaction_relevance(
                    current_query, current_topics, current_semantic_hash, interaction
                )

                combined_relevance = (
                    relevance_factors["topic_similarity"]*0.4 +
                    relevance_factors["semantic_similarity"]*0.3 +
                    relevance_factors["recency_score"]*0.2 +
                    relevance_factors["source_overlap"]*0.1
                )

                if combined_relevance > 0.15:
                    relevant_interactions .append({
                        "interaction": interaction,
                        "relevance": combined_relevance,
                        "factors": relevance_factors
                    })
            relevant_interactions .sort(
                key=lambda x: x["relevance"], reverse=True)
            context = self ._build_smart_context(
                relevant_interactions, max_context_length)
            self ._context_cache[cache_key] = context

            if context:
                self .stats["context_used_count"] += 1
                self .stats["average_context_length"] = (
                    (self .stats["average_context_length"] *
                     (self .stats["context_used_count"]-1)+len(context))
                    / self .stats["context_used_count"]
                )

            self .logger .info(
                f"Generated enhanced context with {len(relevant_interactions)} interactions, {len(context)} chars")
            return context

        except Exception as e:
            self .logger .error(
                f"Failed to generate conversation context: {str(e)}")
            return ""

    def get_follow_up_suggestions(self, current_query: str, response: str) -> List[str]:
        try:
            suggestions = []
            current_topics = self ._extract_topics_with_weights(
                current_query + ""+response)
            entities = self ._extract_entities_enhanced(response)
            sorted_topics = sorted(
                current_topics .items(), key=lambda x: x[1], reverse=True)
            for topic, weight in sorted_topics[:3]:
                if weight > 0.3:
                    suggestions .extend([
                        f"Расскажите больше о {topic}",
                        f"Какие есть примеры {topic}?",
                        f"Как {topic} связано с другими вопросами?",
                        f"Какие проблемы связаны с {topic}?"
                    ])
            for entity in entities[:3]:
                suggestions .extend([
                    f"Что еще известно о {entity}?",
                    f"Где можно найти информацию о {entity}?",
                    f"Какие документы содержат информацию о {entity}?"
                ])
            flow_suggestions = self ._analyze_conversation_flow(
                current_query, response)
            suggestions .extend(flow_suggestions)
            evolution_suggestions = self ._suggest_topic_evolution()
            suggestions .extend(evolution_suggestions)
            if len(self .conversation_history) > 1:
                pattern_suggestions = self ._find_advanced_question_patterns()
                suggestions .extend(pattern_suggestions)
            ranked_suggestions = self ._rank_and_deduplicate_suggestions(
                suggestions, current_topics)
            return ranked_suggestions[:6]

        except Exception as e:
            self .logger .error(
                f"Failed to generate follow-up suggestions: {str(e)}")
            return []

    def detect_conversation_shift(self, current_query: str) -> Dict[str, Any]:
        try:
            if not self .conversation_history:
                return {"shift_detected": False, "confidence": 0.0}

            current_topics = self ._extract_topics(current_query)
            recent_topics = set()
            for interaction in list(self .conversation_history)[-2:]:
                recent_topics .update(interaction["query_topics"])
                recent_topics .update(interaction["response_topics"])
            overlap = len(current_topics .intersection(recent_topics))
            total_topics = len(current_topics .union(recent_topics))

            if total_topics == 0:
                shift_confidence = 0.0
            else:
                shift_confidence = 1.0 - (overlap / total_topics)

            shift_detected = shift_confidence > 0.7

            if shift_detected:
                self .logger .info(
                    f"Conversation shift detected with confidence {shift_confidence:.2f}")

            return {
                "shift_detected": shift_detected,
                "confidence": shift_confidence,
                "current_topics": list(current_topics),
                "recent_topics": list(recent_topics),
                "new_topics": list(current_topics - recent_topics)
            }

        except Exception as e:
            self .logger .error(
                f"Failed to detect conversation shift: {str(e)}")
            return {"shift_detected": False, "confidence": 0.0, "error": str(e)}

    def _extract_topics(self, text: str) -> set:
        topics_with_weights = self ._extract_topics_with_weights(text)
        return set(topics_with_weights .keys())

    def _extract_topics_with_weights(self, text: str) -> Dict[str, float]:
        try:
            cache_key = hashlib .md5(text .encode()).hexdigest()
            if cache_key in self ._topic_cache:
                return self ._topic_cache[cache_key]
            clean_text = re .sub(r'[^\w\s]', '', text .lower())
            words = clean_text .split()
            stop_words = {
                'что', 'как', 'где', 'когда', 'почему', 'какой', 'это', 'то', 'на', 'в',
                'и', 'с', 'для', 'по', 'из', 'за', 'при', 'до', 'после', 'через',
                'может', 'можно', 'нужно', 'есть', 'был', 'будет', 'иметь', 'быть',
                'очень', 'также', 'даже', 'если', 'или', 'так', 'все', 'его', 'она',
                'они', 'вы', 'мы', 'их', 'него', 'нее', 'них', 'тем', 'чем', 'чего',
                'кого', 'кому', 'кем', 'о', 'об', 'про', 'под', 'над', 'между'
            }
            word_freq = Counter(word for word in words if len(
                word) > 3 and word not in stop_words)
            max_freq = max(word_freq .values())if word_freq else 1
            topics_with_weights = {}
            for word, freq in word_freq .items():
                if freq >= 1:
                    weight = freq / max_freq
                    if len(word) > 6:
                        weight *= 1.2
                    topics_with_weights[word] = min(weight, 1.0)
            noun_phrases = re .findall(
                r'\b[А-Яа-я]{4,}\s+[А-Яа-я]{4,}(?:\s+[А-Яа-я]{4,})?\b', text)
            for phrase in noun_phrases:
                clean_phrase = phrase .lower().strip()
                if clean_phrase and len(clean_phrase) > 6:
                    topics_with_weights[clean_phrase] = min(
                        0.8 + (len(clean_phrase .split())*0.1), 1.0)
            domain_patterns = [
                r'\b(закон|статья|документ|процедура|требование|норматив)\w*\b',
                r'\b(система|технология|архитектура|интерфейс|компонент)\w*\b',
                r'\b(организация|департамент|управление|отдел|служба)\w*\b'
            ]

            for pattern in domain_patterns:
                matches = re .findall(pattern, text .lower())
                for match in matches:
                    if match not in stop_words:
                        topics_with_weights[match] = min(
                            topics_with_weights .get(match, 0)+0.3, 1.0)
            self ._topic_cache[cache_key] = topics_with_weights
            return topics_with_weights

        except Exception as e:
            self .logger .error(
                f"Topic extraction with weights failed: {str(e)}")
            return {}

    def _extract_entities(self, text: str) -> List[str]:
        return self ._extract_entities_enhanced(text)[:10]

    def _extract_entities_enhanced(self, text: str) -> List[str]:
        try:
            entities = []
            entity_patterns = [
                r'\b[А-Я][а-я]+(?:\s+[А-Я][а-я]+){1,3}\b',
                r'\b[А-Я]{2,}\b',

                r'\b\d{4}\b',
                r'\b\d+(?:\.\d+)?\s*%\b',
                r'\b\d+(?:\s*(?:млн|тыс|тысяч|миллион|миллиард))?\b',
                r'\b\d+(?:\.\d+)?\s*(?:рублей?|долларов?|евро|тенге)\b',

                r'\b\d{1,2}[\.\-]\d{1,2}[\.\-]\d{2,4}\b',
                r'\b(?:январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь)\s+\d{4}\b',
                r'\b(?:статья|пункт|раздел|глава)\s+\d+(?:\.\d+)*\b',
                r'\b№\s*\d+(?:[-\/]\d+)*\b',

                r'\b[A-Z]{2,}(?:\-[A-Z0-9]+)*\b',
                r'\b\w+\.\w+(?:\.\w+)*\b'
            ]

            for pattern in entity_patterns:
                matches = re .findall(pattern, text)
                entities .extend(matches)
            quoted_text = re .findall(r'["«](.*?)["»]', text)
            entities .extend(quoted_text)
            cleaned_entities = []
            seen = set()

            for entity in entities:
                clean_entity = entity .strip()
                if (clean_entity and
                    len(clean_entity) > 2 and
                    clean_entity .lower()not in seen and
                        not clean_entity .isdigit()):

                    seen .add(clean_entity .lower())
                    cleaned_entities .append(clean_entity)
            cleaned_entities .sort(key=len, reverse=True)

            return cleaned_entities[:15]

        except Exception as e:
            self .logger .error(f"Enhanced entity extraction failed: {str(e)}")
            return []

    def _calculate_recency_score(self, timestamp_str: str) -> float:
        try:
            timestamp = datetime .fromisoformat(timestamp_str)
            time_diff = datetime .now()-timestamp
            hours_passed = time_diff .total_seconds()/3600
            recency_score = max(0.0, 1.0 - (hours_passed / 24.0))

            return recency_score

        except Exception as e:
            self .logger .error(f"Recency calculation failed: {str(e)}")
            return 0.0

    def _find_common_question_patterns(self) -> List[str]:
        return self ._find_advanced_question_patterns()

    def get_memory_stats(self) -> Dict[str, Any]:
        total_cache_requests = self .advanced_stats["cache_hits"] + \
            self .advanced_stats["cache_misses"]
        cache_hit_rate = (
            self .advanced_stats["cache_hits"]/total_cache_requests
            if total_cache_requests > 0 else 0.0
        )

        return {
            "interactions_stored": len(self .conversation_history),
            "current_session": self .current_session_id,
            "active_topics": list(self .current_topics),
            "topic_transitions": len(self .topic_transitions),
            "topic_clusters": len(self .topic_clusters),
            "weighted_topics_count": len(self .topic_weights),
            "cache_performance": {
                "context_cache_size": len(self ._context_cache),
                "similarity_cache_size": len(self ._similarity_cache),
                "topic_cache_size": len(self ._topic_cache),
                "cache_hit_rate": round(cache_hit_rate, 3)
            },
            "advanced_stats": self .advanced_stats .copy(),
            "basic_stats": self .stats .copy(),
            "conversation_quality": {
                "avg_interaction_length": self ._calculate_avg_interaction_length(),
                "topic_coherence_score": self ._calculate_topic_coherence(),
                "conversation_depth": len(self .topic_evolution)
            }
        }

    def export_conversation(self) -> List[Dict[str, Any]]:
        return list(self .conversation_history)

    def clear_memory(self):
        self .conversation_history .clear()
        self .current_topics .clear()
        self .topic_transitions .clear()
        self .topic_evolution .clear()
        self .semantic_clusters .clear()
        self .conversation_embeddings .clear()
        self .topic_weights .clear()
        self .topic_clusters .clear()
        self ._context_cache .clear()
        self ._similarity_cache .clear()
        self ._topic_cache .clear()
        self .stats = {
            "total_interactions": 0,
            "context_used_count": 0,
            "topic_switches": 0,
            "average_context_length": 0.0
        }
        self .advanced_stats = {
            "semantic_similarity_used": 0,
            "topic_transitions_detected": 0,
            "context_summarizations": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        self .current_session_id = None
        self .logger .info("Conversation memory and caches cleared")
