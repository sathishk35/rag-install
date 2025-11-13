"""
Advanced RAG Techniques
Implements HyDE, query decomposition, and other advanced retrieval strategies
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class DecomposedQuery:
    """Decomposed query component"""
    sub_query: str
    intent: str
    priority: float
    dependencies: List[str]

@dataclass
class MultiTeamDomain:
    """Multi-team domain routing information"""
    team: str
    domain: str
    sub_domains: List[str]
    confidence: float
    keywords: List[str]

class HyDERetriever:
    """
    Hypothetical Document Embeddings (HyDE) Retrieval
    Generates a hypothetical answer, embeds it, and uses it for retrieval
    """

    def __init__(self, llm_client, embedding_model, vector_client):
        """
        Args:
            llm_client: LLM client for generating hypothetical documents
            embedding_model: Embedding model
            vector_client: Vector database client
        """
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.vector_client = vector_client
        self.logger = logging.getLogger(__name__)

        # HyDE prompt template
        self.hyde_prompt_template = """Given the following question, write a detailed technical answer as if you were responding from documentation or code comments. Be specific and technical.

Question: {question}

Detailed technical answer:"""

    async def retrieve(self,
                      query: str,
                      collection_name: str,
                      top_k: int = 10,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve using HyDE method

        Args:
            query: User query
            collection_name: Vector collection name
            top_k: Number of results
            filters: Optional filters

        Returns:
            List of retrieved documents
        """
        try:
            # Step 1: Generate hypothetical document
            hypothetical_doc = await self._generate_hypothetical_document(query)

            if not hypothetical_doc:
                self.logger.warning("Failed to generate hypothetical document, falling back to standard retrieval")
                return await self._standard_retrieve(query, collection_name, top_k, filters)

            # Step 2: Embed the hypothetical document
            hyde_embedding = self.embedding_model.encode(hypothetical_doc).tolist()

            # Step 3: Search using the hypothetical embedding
            search_params = {
                'collection_name': collection_name,
                'query_vector': hyde_embedding,
                'limit': top_k
            }

            if filters:
                search_params['query_filter'] = filters

            results = self.vector_client.search(**search_params)

            # Convert to dict format
            documents = []
            for hit in results:
                documents.append({
                    'id': str(hit.id),
                    'content': hit.payload.get('content', ''),
                    'score': hit.score,
                    'metadata': hit.payload,
                    'retrieval_method': 'hyde'
                })

            self.logger.info(f"HyDE retrieval returned {len(documents)} documents")
            return documents

        except Exception as e:
            self.logger.error(f"HyDE retrieval error: {e}")
            return await self._standard_retrieve(query, collection_name, top_k, filters)

    async def _generate_hypothetical_document(self, query: str) -> Optional[str]:
        """Generate hypothetical document using LLM"""
        try:
            # Format prompt
            prompt = self.hyde_prompt_template.format(question=query)

            # Generate using LLM
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.7
            )

            return response.get('text', '').strip()

        except Exception as e:
            self.logger.error(f"Error generating hypothetical document: {e}")
            return None

    async def _standard_retrieve(self,
                                 query: str,
                                 collection_name: str,
                                 top_k: int,
                                 filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback to standard retrieval"""
        query_embedding = self.embedding_model.encode(query).tolist()

        search_params = {
            'collection_name': collection_name,
            'query_vector': query_embedding,
            'limit': top_k
        }

        if filters:
            search_params['query_filter'] = filters

        results = self.vector_client.search(**search_params)

        documents = []
        for hit in results:
            documents.append({
                'id': str(hit.id),
                'content': hit.payload.get('content', ''),
                'score': hit.score,
                'metadata': hit.payload,
                'retrieval_method': 'standard'
            })

        return documents


class QueryDecomposer:
    """
    Decomposes complex queries into simpler sub-queries
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Optional LLM client for intelligent decomposition
        """
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)

        # Decomposition prompt
        self.decomposition_prompt = """Break down the following complex question into 3-5 simple, specific sub-questions that together would help answer the main question.

Main Question: {question}

Sub-questions:
1."""

    async def decompose(self, complex_query: str) -> List[DecomposedQuery]:
        """
        Decompose complex query into sub-queries

        Args:
            complex_query: Complex query to decompose

        Returns:
            List of DecomposedQuery objects
        """
        try:
            if self.llm_client:
                return await self._llm_decompose(complex_query)
            else:
                return self._rule_based_decompose(complex_query)

        except Exception as e:
            self.logger.error(f"Query decomposition error: {e}")
            # Return original query as single component
            return [DecomposedQuery(
                sub_query=complex_query,
                intent='general',
                priority=1.0,
                dependencies=[]
            )]

    async def _llm_decompose(self, query: str) -> List[DecomposedQuery]:
        """Use LLM to decompose query"""
        try:
            prompt = self.decomposition_prompt.format(question=query)
            response = await self.llm_client.generate(prompt=prompt, max_tokens=300)

            # Parse response into sub-queries
            text = response.get('text', '')
            sub_queries = self._parse_sub_queries(text)

            # Create DecomposedQuery objects
            decomposed = []
            for i, sq in enumerate(sub_queries):
                decomposed.append(DecomposedQuery(
                    sub_query=sq,
                    intent='unknown',
                    priority=1.0 - (i * 0.1),  # Decrease priority for later queries
                    dependencies=[]
                ))

            return decomposed if decomposed else [DecomposedQuery(
                sub_query=query, intent='general', priority=1.0, dependencies=[]
            )]

        except Exception as e:
            self.logger.error(f"LLM decomposition error: {e}")
            return [DecomposedQuery(
                sub_query=query, intent='general', priority=1.0, dependencies=[]
            )]

    def _rule_based_decompose(self, query: str) -> List[DecomposedQuery]:
        """Rule-based decomposition"""
        # Split on common separators
        separators = [' and ', ' AND ', ' & ', ' also ', ' plus ', ' as well as ']

        sub_queries = [query]
        for separator in separators:
            new_queries = []
            for q in sub_queries:
                new_queries.extend(q.split(separator))
            sub_queries = new_queries

        # Clean and filter
        sub_queries = [q.strip() for q in sub_queries if len(q.strip()) > 10]

        if not sub_queries:
            sub_queries = [query]

        return [
            DecomposedQuery(
                sub_query=sq,
                intent='general',
                priority=1.0 / (i + 1),
                dependencies=[]
            )
            for i, sq in enumerate(sub_queries)
        ]

    def _parse_sub_queries(self, text: str) -> List[str]:
        """Parse sub-queries from LLM output"""
        import re

        # Look for numbered list items
        pattern = r'\d+\.\s*(.+?)(?=\d+\.|$)'
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)

        if matches:
            return [m.strip() for m in matches if len(m.strip()) > 10]

        # Fallback: split on newlines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return [line for line in lines if len(line) > 10][:5]


class MultiTeamDomainRouter:
    """
    Routes queries to appropriate team domains
    Designed for multi-team organizations (software, hardware, specialized)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Team domain definitions
        self.team_domains = {
            'software': {
                'bsp': {
                    'keywords': ['bootloader', 'u-boot', 'kernel', 'device tree', 'boot', 'init'],
                    'sub_domains': ['bootloader', 'kernel', 'device_tree']
                },
                'driver': {
                    'keywords': ['driver', 'device driver', 'peripheral', 'i2c', 'spi', 'uart', 'usb', 'pci'],
                    'sub_domains': ['peripheral_drivers', 'bus_drivers', 'network_drivers', 'char_drivers']
                },
                'application': {
                    'keywords': ['application', 'user space', 'gui', 'interface', 'middleware', 'service'],
                    'sub_domains': ['user_space', 'middleware', 'ui']
                },
                'embedded': {
                    'keywords': ['embedded', 'firmware', 'microcontroller', 'mcu', 'rtos', 'bare metal'],
                    'sub_domains': ['rtos', 'bare_metal', 'firmware']
                }
            },
            'hardware': {
                'boards': {
                    'keywords': ['pcb', 'board', 'schematic', 'layout', 'circuit board', 'hardware design'],
                    'sub_domains': ['pcb', 'schematics', 'layout', 'assembly']
                },
                'digital': {
                    'keywords': ['fpga', 'asic', 'verilog', 'vhdl', 'rtl', 'synthesis', 'digital design'],
                    'sub_domains': ['fpga', 'asic', 'verilog', 'vhdl']
                },
                'rf': {
                    'keywords': ['rf', 'radio frequency', 'antenna', 'amplifier', 'filter', 'mixer', 'oscillator'],
                    'sub_domains': ['antenna', 'amplifier', 'filter', 'mixer']
                },
                'systems': {
                    'keywords': ['power', 'thermal', 'mechanical', 'cooling', 'power supply'],
                    'sub_domains': ['power', 'thermal', 'mechanical']
                }
            },
            'specialized': {
                'radar': {
                    'keywords': ['radar', 'signal processing', 'waveform', 'tracking', 'detection', 'dsp'],
                    'sub_domains': ['signal_processing', 'waveform', 'tracking', 'detection']
                },
                'satellite': {
                    'keywords': ['satellite', 'telemetry', 'orbit', 'ground station', 'spacecraft', 'tracking'],
                    'sub_domains': ['telemetry', 'orbit', 'ground_station', 'tracking']
                },
                'ew': {
                    'keywords': ['electronic warfare', 'ew', 'jamming', 'ecm', 'eccm', 'intercept'],
                    'sub_domains': ['jamming', 'intercept', 'analysis', 'countermeasures']
                }
            }
        }

    def route_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> List[MultiTeamDomain]:
        """
        Route query to appropriate team domains

        Args:
            query: User query
            user_context: Optional user context (team, clearance, etc.)

        Returns:
            List of MultiTeamDomain objects sorted by confidence
        """
        query_lower = query.lower()
        matches = []

        # Score each domain
        for team, domains in self.team_domains.items():
            for domain, config in domains.items():
                score = self._calculate_domain_score(query_lower, config['keywords'])

                if score > 0:
                    matches.append(MultiTeamDomain(
                        team=team,
                        domain=domain,
                        sub_domains=config['sub_domains'],
                        confidence=score,
                        keywords=[kw for kw in config['keywords'] if kw in query_lower]
                    ))

        # Sort by confidence
        matches.sort(key=lambda x: x.confidence, reverse=True)

        # Filter by user context if provided
        if user_context and 'allowed_teams' in user_context:
            allowed_teams = user_context['allowed_teams']
            matches = [m for m in matches if m.team in allowed_teams]

        # Return top 3 matches
        return matches[:3]

    def _calculate_domain_score(self, query: str, keywords: List[str]) -> float:
        """Calculate relevance score for a domain"""
        score = 0.0

        for keyword in keywords:
            if keyword in query:
                # Exact match scores higher
                score += 1.0

                # Bonus for word boundaries (whole word match)
                import re
                if re.search(r'\b' + re.escape(keyword) + r'\b', query):
                    score += 0.5

        # Normalize by keyword count
        if keywords:
            score = score / len(keywords)

        return min(score, 1.0)

    def get_domain_filters(self, routed_domains: List[MultiTeamDomain]) -> Dict[str, Any]:
        """
        Generate filters for vector search based on routed domains

        Args:
            routed_domains: List of MultiTeamDomain objects

        Returns:
            Dictionary of filters for vector search
        """
        if not routed_domains:
            return {}

        # Collect all domains and sub-domains
        domains = []
        for rd in routed_domains:
            domains.append(rd.domain)
            domains.extend(rd.sub_domains)

        return {
            'domain': list(set(domains)),
            'team': list(set(rd.team for rd in routed_domains))
        }


class ParentChildChunker:
    """
    Parent-Child chunking strategy
    Stores small chunks for retrieval, large chunks for context
    """

    def __init__(self,
                 parent_chunk_size: int = 2000,
                 parent_overlap: int = 200,
                 child_chunk_size: int = 400,
                 child_overlap: int = 50):
        """
        Args:
            parent_chunk_size: Size of parent chunks (tokens)
            parent_overlap: Overlap between parent chunks
            child_chunk_size: Size of child chunks (tokens)
            child_overlap: Overlap between child chunks
        """
        self.parent_chunk_size = parent_chunk_size
        self.parent_overlap = parent_overlap
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        self.logger = logging.getLogger(__name__)

    def create_chunks(self, document: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Create parent-child chunk hierarchy

        Args:
            document: Document text
            metadata: Optional metadata

        Returns:
            List of chunk dictionaries with parent-child relationships
        """
        chunks = []

        # Create parent chunks
        parent_chunks = self._chunk_text(
            document,
            self.parent_chunk_size,
            self.parent_overlap
        )

        # For each parent, create child chunks
        for parent_idx, parent_text in enumerate(parent_chunks):
            parent_id = f"parent_{parent_idx}"

            # Create child chunks
            child_chunks = self._chunk_text(
                parent_text,
                self.child_chunk_size,
                self.child_overlap
            )

            # Add chunks with relationships
            for child_idx, child_text in enumerate(child_chunks):
                child_id = f"child_{parent_idx}_{child_idx}"

                chunks.append({
                    'id': child_id,
                    'content': child_text,
                    'chunk_type': 'child',
                    'parent_id': parent_id,
                    'parent_content': parent_text,
                    'metadata': {
                        **(metadata or {}),
                        'parent_idx': parent_idx,
                        'child_idx': child_idx,
                        'total_parents': len(parent_chunks),
                        'total_children': len(child_chunks)
                    }
                })

        self.logger.info(
            f"Created {len(chunks)} child chunks from "
            f"{len(parent_chunks)} parent chunks"
        )

        return chunks

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Simple text chunking"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks

    def retrieve_with_parent(self,
                           child_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Given child chunk results, return parent chunks for context

        Args:
            child_results: Results from child chunk search

        Returns:
            Results with parent context
        """
        results_with_parent = []

        for result in child_results:
            # Replace child content with parent content for more context
            result_copy = result.copy()
            result_copy['retrieved_chunk'] = result['content']
            result_copy['content'] = result.get('parent_content', result['content'])

            results_with_parent.append(result_copy)

        return results_with_parent
