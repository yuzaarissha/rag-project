from typing import Dict, Any, List, Optional, Tuple
import re
from datetime import datetime
import logging
import unicodedata
from collections import Counter
import hashlib


class QueryProcessor:

    def __init__(self, llm_manager=None):
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(__name__)

        self.spell_corrections = {
            "тскт": "текст", "докумнт": "документ", "информцаия": "информация",
            "сколко": "сколько", "откуд": "откуда", "зачем": "зачем",
            "гдe": "где", "што": "что", "чьо": "что", "какой-то": "какой то",

            "зокон": "закон", "статъя": "статья", "порядак": "порядок",
            "процедуро": "процедура", "документооборот": "документооборот",
            "регламент": "регламент", "положени": "положение",


            "система": "система", "процес": "процесс", "методь": "метод",
            "алгоритьм": "алгоритм", "структуро": "структура",
            "организацыя": "организация", "учреждени": "учреждение"
        }

        self.domain_expansions = {
            "закон": ["законодательство", "нормативный акт", "правовой акт"],
            "документ": ["файл", "материал", "текст", "бумага"],
            "процедура": ["процесс", "алгоритм", "порядок", "методика"],
            "организация": ["учреждение", "предприятие", "компания", "структура"],
            "информация": ["данные", "сведения", "факты", "материалы"],
            "система": ["механизм", "структура", "схема", "комплекс"]
        }

        self.stop_words = {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а',
            'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
            'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
            'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже',
            'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был',
            'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там',
            'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть',
            'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб',
            'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж',
            'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем',
            'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее',
            'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при',
            'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше',
            'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много',
            'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед',
            'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более',
            'всегда', 'конечно', 'всю', 'между'
        }

        self.intent_patterns = {
            "definition": [
                r"что такое",
                r"определение",
                r"объясни",
                r"что означает",
                r"define",
                r"what is",
                r"explain"
            ],
            "comparison": [
                r"разница между",
                r"отличие",
                r"сравни",
                r"чем отличается",
                r"difference between",
                r"compare"
            ],
            "procedure": [
                r"как делать",
                r"как сделать",
                r"процедура",
                r"пошагово",
                r"инструкция",
                r"how to",
                r"step by step"
            ],
            "quantitative": [
                r"сколько",
                r"количество",
                r"how many",
                r"how much",
                r"число",
                r"count"
            ],
            "temporal": [
                r"когда",
                r"время",
                r"дата",
                r"период",
                r"when",
                r"time"
            ],
            "location": [
                r"где",
                r"место",
                r"адрес",
                r"where",
                r"location"
            ],
            "causal": [
                r"почему",
                r"причина",
                r"зачем",
                r"why",
                r"because",
                r"reason"
            ]
        }

    def preprocess_query(self, query: str, conversation_context: str = None) -> Dict[str, Any]:
        try:
            start_time = datetime.now()

            cleaned_query = self._clean_query(query)

            corrected_query = self._spell_correct(cleaned_query)

            language = self._detect_language(corrected_query)

            intent = self._classify_intent(corrected_query)

            keywords = self._extract_keywords(corrected_query)

            expanded_queries = self._expand_query(
                corrected_query, intent, keywords)

            if conversation_context:
                contextualized_query = self._integrate_conversation_context(
                    corrected_query, conversation_context
                )
            else:
                contextualized_query = corrected_query

            processing_time = (datetime.now() - start_time).total_seconds()

            result = {
                "original_query": query,
                "cleaned_query": cleaned_query,
                "corrected_query": corrected_query,
                "final_query": contextualized_query,
                "language": language,
                "intent": intent,
                "keywords": keywords,
                "expanded_queries": expanded_queries,
                "processing_time": processing_time,
                "has_conversation_context": conversation_context is not None
            }

            self.logger.info(
                f"Query preprocessed: {intent} intent, {language} language, {len(expanded_queries)} expansions")
            return result

        except Exception as e:
            self.logger.error(f"Query preprocessing failed: {str(e)}")
            return {
                "original_query": query,
                "final_query": query,
                "language": "unknown",
                "intent": "general",
                "keywords": [],
                "expanded_queries": [query],
                "error": str(e)
            }

    def _clean_query(self, query: str) -> str:
        query = re.sub(r'\s+', ' ', query.strip())

        query = re.sub(r'[^\w\s\?\!\.\,\:\;\-]', '', query)

        if not query.endswith('?') and any(word in query.lower() for word in ['что', 'как', 'где', 'когда', 'почему', 'сколько']):
            query += '?'

        return query

    def _spell_correct(self, query: str) -> str:
        words = query.split()
        corrected_words = []

        for word in words:
            word_lower = word.lower()
            if word_lower in self.spell_corrections:
                corrected_words.append(self.spell_corrections[word_lower])
            else:
                corrected_words.append(word)

        return ' '.join(corrected_words)

    def _detect_language(self, query: str) -> str:
        query_lower = query.lower()

        russian_chars = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
        english_chars = 'abcdefghijklmnopqrstuvwxyz'

        russian_count = sum(1 for char in query_lower if char in russian_chars)
        english_count = sum(1 for char in query_lower if char in english_chars)

        if russian_count > english_count:
            return "russian"
        elif english_count > 0:
            return "english"
        else:
            return "russian"

    def _classify_intent(self, query: str) -> str:
        query_lower = query.lower()

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent

        return "general"

    def _extract_keywords(self, query: str) -> List[str]:
        clean_query = re.sub(r'[^\w\s]', ' ', query.lower())
        words = clean_query.split()

        keywords = [
            word for word in words
            if word not in self.stop_words and len(word) > 2
        ]

        return keywords

    def _expand_query(self, query: str, intent: str, keywords: List[str]) -> List[str]:
        expansions = [query]

        try:
            if intent == "definition":
                for keyword in keywords:
                    expansions.append(f"определение {keyword}")
                    expansions.append(f"{keyword} это")
                    expansions.append(f"что означает {keyword}")

            elif intent == "procedure":
                for keyword in keywords:
                    expansions.append(f"инструкция {keyword}")
                    expansions.append(f"пошагово {keyword}")
                    expansions.append(f"процедура {keyword}")

            elif intent == "quantitative":
                for keyword in keywords:
                    expansions.append(f"количество {keyword}")
                    expansions.append(f"число {keyword}")
                    expansions.append(f"сколько {keyword}")

            synonyms = {
                "документ": ["файл", "текст", "материал"],
                "информация": ["данные", "сведения", "факты"],
                "процесс": ["процедура", "алгоритм", "метод"],
                "система": ["механизм", "структура", "схема"]
            }

            for keyword in keywords:
                if keyword in synonyms:
                    for synonym in synonyms[keyword]:
                        expanded = query.replace(keyword, synonym)
                        if expanded != query:
                            expansions.append(expanded)

            seen = set()
            unique_expansions = []
            for exp in expansions:
                if exp not in seen:
                    seen.add(exp)
                    unique_expansions.append(exp)

            return unique_expansions[:5]

        except Exception as e:
            self.logger.warning(f"Query expansion failed: {str(e)}")
            return [query]

    def _integrate_conversation_context(self, query: str, context: str) -> str:
        try:
            if not context or len(context.strip()) < 10:
                return query

            context_keywords = self._extract_keywords(context)
            query_keywords = self._extract_keywords(query)

            overlap = set(context_keywords) & set(query_keywords)

            if overlap:
                return f"В контексте {' '.join(overlap)}: {query}"
            else:
                return query

        except Exception as e:
            self.logger.warning(f"Context integration failed: {str(e)}")
            return query

    def get_processing_stats(self) -> Dict[str, Any]:
        return {
            "spell_corrections_available": len(self.spell_corrections),
            "intent_patterns": len(self.intent_patterns),
            "stop_words": len(self.stop_words),
            "supported_languages": ["russian", "english"]
        }
