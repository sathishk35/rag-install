"""
Query Optimizer for RAG System
Enhances queries for better retrieval performance and accuracy
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
from collections import defaultdict

@dataclass
class OptimizedQuery:
    """Container for optimized query information"""
    original_query: str
    optimized_query: str
    query_type: str  # 'code', 'documentation', 'troubleshooting', 'general'
    intent: str
    domain_hints: List[str]
    expanded_terms: List[str]
    filters: Dict[str, Any]
    confidence: float

class QueryOptimizer:
    """Optimizes queries for better retrieval performance"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Domain-specific terminology expansions
        self.domain_synonyms = {
            # Software/BSP terms
            "driver": ["device driver", "kernel module", "peripheral driver"],
            "bootloader": ["u-boot", "boot loader", "bootstrap"],
            "kernel": ["linux kernel", "rtos kernel", "os kernel"],
            "firmware": ["embedded software", "microcode", "flash software"],

            # Hardware terms
            "pcb": ["printed circuit board", "circuit board", "board"],
            "fpga": ["field programmable gate array", "programmable logic"],
            "asic": ["application specific integrated circuit", "custom chip"],
            "rf": ["radio frequency", "wireless"],

            # Radar/Defense terms
            "radar": ["radio detection and ranging", "surveillance system"],
            "ew": ["electronic warfare", "jamming", "electronic countermeasures"],
            "signal processing": ["dsp", "digital signal processing", "signal analysis"],

            # General technical terms
            "bug": ["defect", "issue", "problem", "error"],
            "fix": ["patch", "correction", "resolution", "solution"],
            "api": ["application programming interface", "interface", "function"],
            "optimize": ["improve", "enhance", "tune", "refactor"],
        }

        # Programming language synonyms
        self.language_terms = {
            "c": ["c programming", "c language", "c code"],
            "cpp": ["c++", "c plus plus", "cxx"],
            "python": ["py", "python script", "python code"],
            "matlab": ["matlab script", "m-file"],
        }

        # Query intent patterns
        self.intent_patterns = {
            'code_search': [
                r'\b(function|class|method|api|implementation|code)\b',
                r'\b(how to implement|show me|example of)\b',
                r'\.(c|cpp|py|h|hpp)\b',
            ],
            'documentation': [
                r'\b(document|specification|manual|guide|readme|doc)\b',
                r'\b(what is|explain|describe|definition)\b',
                r'\.(pdf|docx|md)\b',
            ],
            'troubleshooting': [
                r'\b(error|bug|issue|problem|fix|debug|crash|fail)\b',
                r'\b(why|how to fix|resolve|solution)\b',
                r'\b(not working|doesn\'t work|broken)\b',
            ],
            'architecture': [
                r'\b(architecture|design|structure|overview|system)\b',
                r'\b(how does|workflow|pipeline|flow)\b',
            ],
            'configuration': [
                r'\b(config|configure|setup|install|settings)\b',
                r'\b(how to setup|installation|deployment)\b',
            ]
        }

        # Code-specific patterns
        self.code_patterns = {
            'function_name': r'\b[a-z_][a-z0-9_]*\(\)',
            'class_name': r'\b[A-Z][a-zA-Z0-9]*\b',
            'file_path': r'[\w/]+\.(c|cpp|h|hpp|py|m)',
            'api_call': r'\b[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*',
        }

        # Domain detection patterns
        self.domain_patterns = {
            'drivers': [r'\b(driver|device|peripheral|bus|i2c|spi|uart|usb)\b'],
            'embedded': [r'\b(embedded|firmware|microcontroller|mcu|rtos|bare.?metal)\b'],
            'radar': [r'\b(radar|signal.?processing|waveform|tracking|detection)\b'],
            'ew': [r'\b(electronic.?warfare|jamming|ecm|eccm|intercept)\b'],
            'satellite': [r'\b(satellite|telemetry|orbit|ground.?station|spacecraft)\b'],
            'ate': [r'\b(ate|automatic.?test|test.?equipment|testing)\b'],
            'rf': [r'\b(rf|radio.?frequency|antenna|amplifier|filter|mixer)\b'],
            'digital': [r'\b(fpga|verilog|vhdl|asic|digital.?design)\b'],
        }

        self.logger.info("QueryOptimizer initialized")

    async def optimize(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> OptimizedQuery:
        """
        Main optimization method

        Args:
            query: Original user query
            user_context: Optional user context (domain, clearance, etc.)

        Returns:
            OptimizedQuery object with enhanced query information
        """
        try:
            # Step 1: Detect query intent
            query_type, intent = self._detect_intent(query)

            # Step 2: Extract domain hints
            domain_hints = self._extract_domain_hints(query)

            # Step 3: Expand query with synonyms
            expanded_terms = self._expand_query_terms(query, query_type)

            # Step 4: Build optimized query
            optimized_query = self._build_optimized_query(query, expanded_terms, query_type)

            # Step 5: Generate filters based on context
            filters = self._generate_filters(query_type, domain_hints, user_context)

            # Step 6: Calculate confidence
            confidence = self._calculate_confidence(query, query_type)

            result = OptimizedQuery(
                original_query=query,
                optimized_query=optimized_query,
                query_type=query_type,
                intent=intent,
                domain_hints=domain_hints,
                expanded_terms=expanded_terms,
                filters=filters,
                confidence=confidence
            )

            self.logger.info(f"Query optimized: type={query_type}, domains={domain_hints}")
            return result

        except Exception as e:
            self.logger.error(f"Error optimizing query: {e}")
            # Return original query if optimization fails
            return OptimizedQuery(
                original_query=query,
                optimized_query=query,
                query_type="general",
                intent="unknown",
                domain_hints=[],
                expanded_terms=[],
                filters={},
                confidence=0.5
            )

    def _detect_intent(self, query: str) -> Tuple[str, str]:
        """Detect the intent and type of the query"""
        query_lower = query.lower()

        # Check each intent pattern
        intent_scores = defaultdict(int)
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    intent_scores[intent] += 1

        # Get highest scoring intent
        if intent_scores:
            primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            return primary_intent, primary_intent

        return "general", "general_query"

    def _extract_domain_hints(self, query: str) -> List[str]:
        """Extract domain hints from the query"""
        detected_domains = []
        query_lower = query.lower()

        for domain, patterns in self.domain_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    detected_domains.append(domain)
                    break

        return detected_domains

    def _expand_query_terms(self, query: str, query_type: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        expanded = []
        query_lower = query.lower()

        # Expand domain-specific terms
        for term, synonyms in self.domain_synonyms.items():
            if term in query_lower:
                expanded.extend(synonyms)

        # Expand programming language terms for code queries
        if query_type == 'code_search':
            for lang, variants in self.language_terms.items():
                if lang in query_lower:
                    expanded.extend(variants)

        return list(set(expanded))  # Remove duplicates

    def _build_optimized_query(self, original_query: str, expanded_terms: List[str], query_type: str) -> str:
        """Build the optimized query string"""
        # Start with original query
        optimized = original_query

        # For code queries, enhance with programming context
        if query_type == 'code_search':
            # Extract potential function/class names
            code_elements = []
            for pattern_name, pattern in self.code_patterns.items():
                matches = re.findall(pattern, original_query)
                code_elements.extend(matches)

            if code_elements:
                optimized = f"{original_query} {' '.join(code_elements)}"

        # Add most relevant expanded terms (limit to top 3)
        if expanded_terms:
            top_terms = expanded_terms[:3]
            optimized = f"{optimized} {' '.join(top_terms)}"

        return optimized.strip()

    def _generate_filters(self, query_type: str, domain_hints: List[str], user_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate filters for vector search based on query analysis"""
        filters = {}

        # Add query type filter
        if query_type == 'code_search':
            filters['content_type'] = ['code', 'script']
            filters['file_extensions'] = ['.c', '.cpp', '.h', '.hpp', '.py', '.m']
        elif query_type == 'documentation':
            filters['content_type'] = ['document', 'pdf', 'docx']
            filters['file_extensions'] = ['.pdf', '.docx', '.md', '.txt']

        # Add domain filters
        if domain_hints:
            filters['domains'] = domain_hints

        # Add user context filters
        if user_context:
            if 'domains' in user_context:
                if 'domains' in filters:
                    # Intersect with detected domains
                    filters['domains'] = list(set(filters['domains']) & set(user_context['domains']))
                else:
                    filters['domains'] = user_context['domains']

            if 'security_clearance' in user_context:
                filters['security_clearance'] = user_context['security_clearance']

        return filters

    def _calculate_confidence(self, query: str, query_type: str) -> float:
        """Calculate confidence score for the optimization"""
        confidence = 0.5  # Base confidence

        # Increase confidence for specific patterns
        if query_type != 'general':
            confidence += 0.2

        # Increase confidence for longer queries
        word_count = len(query.split())
        if word_count > 5:
            confidence += 0.1
        if word_count > 10:
            confidence += 0.1

        # Increase confidence if technical terms detected
        technical_terms = ['function', 'class', 'api', 'driver', 'kernel', 'radar', 'signal']
        if any(term in query.lower() for term in technical_terms):
            confidence += 0.1

        return min(confidence, 1.0)

    async def decompose_query(self, complex_query: str) -> List[str]:
        """
        Decompose complex multi-part queries into simpler sub-queries

        Args:
            complex_query: A complex query that may contain multiple questions

        Returns:
            List of simpler sub-queries
        """
        # Split on common separators
        separators = [' and ', ' AND ', ' & ', ' also ', ' plus ']

        sub_queries = [complex_query]
        for separator in separators:
            new_queries = []
            for query in sub_queries:
                new_queries.extend(query.split(separator))
            sub_queries = new_queries

        # Clean up and filter
        sub_queries = [q.strip() for q in sub_queries if len(q.strip()) > 10]

        # If no decomposition happened, return original
        if len(sub_queries) <= 1:
            return [complex_query]

        return sub_queries

    def extract_technical_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract technical entities from the query"""
        entities = {
            'functions': [],
            'classes': [],
            'files': [],
            'apis': [],
            'keywords': []
        }

        # Extract function names (e.g., function_name())
        functions = re.findall(r'\b([a-z_][a-z0-9_]*)\s*\(', query, re.IGNORECASE)
        entities['functions'] = functions

        # Extract class names (PascalCase)
        classes = re.findall(r'\b([A-Z][a-zA-Z0-9]*)\b', query)
        entities['classes'] = classes

        # Extract file paths
        files = re.findall(r'[\w/]+\.(c|cpp|h|hpp|py|m|txt|md)', query)
        entities['files'] = files

        # Extract API-like patterns (object.method)
        apis = re.findall(r'\b([a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*)\b', query, re.IGNORECASE)
        entities['apis'] = apis

        return entities

    def suggest_alternative_queries(self, query: str, query_type: str) -> List[str]:
        """Suggest alternative phrasings of the query"""
        alternatives = []

        if query_type == 'code_search':
            alternatives.append(f"implementation of {query}")
            alternatives.append(f"example code for {query}")
            alternatives.append(f"{query} source code")
        elif query_type == 'documentation':
            alternatives.append(f"documentation for {query}")
            alternatives.append(f"{query} reference manual")
            alternatives.append(f"{query} specifications")
        elif query_type == 'troubleshooting':
            alternatives.append(f"how to fix {query}")
            alternatives.append(f"{query} solution")
            alternatives.append(f"debugging {query}")

        return alternatives[:3]  # Return top 3
