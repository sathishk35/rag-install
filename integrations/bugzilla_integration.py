"""
Bugzilla Integration for RAG System
Syncs closed bugs and defects into the RAG knowledge base
"""

import logging
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
import re

@dataclass
class BugData:
    """Container for bug information"""
    bug_id: str
    product: str
    component: str
    summary: str
    description: str
    status: str
    resolution: str
    severity: str
    priority: str
    assigned_to: str
    reporter: str
    created_time: datetime
    modified_time: datetime
    resolved_time: Optional[datetime]
    comments: List[Dict[str, Any]]
    attachments: List[Dict[str, Any]]
    depends_on: List[str]
    blocks: List[str]
    keywords: List[str]
    url: str

class BugzillaIntegration:
    """
    Integration with Bugzilla for syncing bug data into RAG system
    """

    def __init__(self,
                 bugzilla_url: str,
                 api_key: Optional[str] = None,
                 username: Optional[str] = None,
                 password: Optional[str] = None):
        """
        Args:
            bugzilla_url: Base URL of Bugzilla instance
            api_key: API key for authentication (preferred)
            username: Username for authentication
            password: Password for authentication
        """
        self.bugzilla_url = bugzilla_url.rstrip('/')
        self.api_key = api_key
        self.username = username
        self.password = password
        self.logger = logging.getLogger(__name__)

        # API endpoints
        self.bug_endpoint = f"{self.bugzilla_url}/rest/bug"
        self.comment_endpoint = f"{self.bugzilla_url}/rest/bug/{{}}/comment"
        self.attachment_endpoint = f"{self.bugzilla_url}/rest/bug/{{}}/attachment"

        # Product/Component to Domain mapping
        self.domain_mapping = {
            'BSP': 'drivers',
            'Driver': 'drivers',
            'Kernel': 'drivers',
            'Embedded': 'embedded',
            'Firmware': 'embedded',
            'RTOS': 'embedded',
            'Application': 'general',
            'ATE': 'ate',
            'Radar': 'radar',
            'EW': 'ew',
            'Satellite': 'satellite',
            'RF': 'rf',
            'Digital': 'digital',
            'FPGA': 'digital',
            'Hardware': 'general',
        }

        # Security classification mapping
        self.security_mapping = {
            'blocker': 'restricted',
            'critical': 'confidential',
            'major': 'confidential',
            'normal': 'internal',
            'minor': 'internal',
            'trivial': 'internal',
            'enhancement': 'internal',
        }

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        headers = {'Content-Type': 'application/json'}

        if self.api_key:
            headers['X-BUGZILLA-API-KEY'] = self.api_key

        return headers

    async def fetch_bugs(self,
                        status: Optional[List[str]] = None,
                        resolution: Optional[List[str]] = None,
                        product: Optional[List[str]] = None,
                        component: Optional[List[str]] = None,
                        changed_after: Optional[datetime] = None,
                        limit: int = 100) -> List[BugData]:
        """
        Fetch bugs from Bugzilla

        Args:
            status: List of bug statuses (e.g., ['RESOLVED', 'CLOSED'])
            resolution: List of resolutions (e.g., ['FIXED'])
            product: List of products to filter
            component: List of components to filter
            changed_after: Only bugs changed after this date
            limit: Maximum number of bugs to fetch

        Returns:
            List of BugData objects
        """
        try:
            # Build query parameters
            params = {
                'limit': limit,
                'include_fields': [
                    'id', 'product', 'component', 'summary', 'description',
                    'status', 'resolution', 'severity', 'priority',
                    'assigned_to', 'reporter', 'creation_time',
                    'last_change_time', 'depends_on', 'blocks', 'keywords'
                ]
            }

            if status:
                params['status'] = status
            if resolution:
                params['resolution'] = resolution
            if product:
                params['product'] = product
            if component:
                params['component'] = component
            if changed_after:
                params['last_change_time'] = changed_after.strftime('%Y-%m-%dT%H:%M:%SZ')

            # Fetch bugs
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.bug_endpoint,
                    params=params,
                    headers=self._get_auth_headers()
                ) as response:
                    if response.status != 200:
                        self.logger.error(f"Bugzilla API error: {response.status}")
                        return []

                    data = await response.json()
                    bugs_raw = data.get('bugs', [])

            # Parse bugs
            bugs = []
            for bug_raw in bugs_raw:
                bug = await self._parse_bug(bug_raw)
                if bug:
                    bugs.append(bug)

            self.logger.info(f"Fetched {len(bugs)} bugs from Bugzilla")
            return bugs

        except Exception as e:
            self.logger.error(f"Error fetching bugs: {e}")
            return []

    async def _parse_bug(self, bug_data: Dict[str, Any]) -> Optional[BugData]:
        """Parse raw bug data into BugData object"""
        try:
            bug_id = str(bug_data['id'])

            # Fetch comments and attachments
            comments = await self._fetch_comments(bug_id)
            attachments = await self._fetch_attachments(bug_id)

            return BugData(
                bug_id=bug_id,
                product=bug_data.get('product', ''),
                component=bug_data.get('component', ''),
                summary=bug_data.get('summary', ''),
                description=bug_data.get('description', ''),
                status=bug_data.get('status', ''),
                resolution=bug_data.get('resolution', ''),
                severity=bug_data.get('severity', ''),
                priority=bug_data.get('priority', ''),
                assigned_to=bug_data.get('assigned_to', ''),
                reporter=bug_data.get('reporter', ''),
                created_time=datetime.fromisoformat(bug_data.get('creation_time', '').replace('Z', '+00:00')),
                modified_time=datetime.fromisoformat(bug_data.get('last_change_time', '').replace('Z', '+00:00')),
                resolved_time=None,  # Could parse from history if needed
                comments=comments,
                attachments=attachments,
                depends_on=bug_data.get('depends_on', []),
                blocks=bug_data.get('blocks', []),
                keywords=bug_data.get('keywords', []),
                url=f"{self.bugzilla_url}/show_bug.cgi?id={bug_id}"
            )

        except Exception as e:
            self.logger.error(f"Error parsing bug: {e}")
            return None

    async def _fetch_comments(self, bug_id: str) -> List[Dict[str, Any]]:
        """Fetch comments for a bug"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.comment_endpoint.format(bug_id),
                    headers=self._get_auth_headers()
                ) as response:
                    if response.status != 200:
                        return []

                    data = await response.json()
                    comments = data.get('bugs', {}).get(bug_id, {}).get('comments', [])
                    return comments

        except Exception as e:
            self.logger.error(f"Error fetching comments for bug {bug_id}: {e}")
            return []

    async def _fetch_attachments(self, bug_id: str) -> List[Dict[str, Any]]:
        """Fetch attachments for a bug"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.attachment_endpoint.format(bug_id),
                    headers=self._get_auth_headers()
                ) as response:
                    if response.status != 200:
                        return []

                    data = await response.json()
                    attachments = data.get('bugs', {}).get(bug_id, [])
                    return attachments

        except Exception as e:
            self.logger.error(f"Error fetching attachments for bug {bug_id}: {e}")
            return []

    def format_bug_as_document(self, bug: BugData) -> str:
        """
        Format bug as a document for ingestion into RAG

        Args:
            bug: BugData object

        Returns:
            Formatted document string
        """
        # Build document
        doc_parts = [
            f"# Bug #{bug.bug_id}: {bug.summary}",
            f"\n**Product**: {bug.product}",
            f"**Component**: {bug.component}",
            f"**Status**: {bug.status}",
            f"**Resolution**: {bug.resolution}",
            f"**Severity**: {bug.severity}",
            f"**Priority**: {bug.priority}",
            f"**Reported by**: {bug.reporter}",
            f"**Assigned to**: {bug.assigned_to}",
            f"**Created**: {bug.created_time.strftime('%Y-%m-%d')}",
            f"**Modified**: {bug.modified_time.strftime('%Y-%m-%d')}",
            f"\n## Description\n{bug.description}",
        ]

        # Add resolution comment (usually the last meaningful comment)
        if bug.comments:
            resolution_comment = self._extract_resolution_comment(bug.comments)
            if resolution_comment:
                doc_parts.append(f"\n## Resolution\n{resolution_comment}")

        # Add code changes mentioned in comments
        code_changes = self._extract_code_changes(bug.comments)
        if code_changes:
            doc_parts.append(f"\n## Code Changes\n{code_changes}")

        # Add related bugs
        if bug.depends_on:
            doc_parts.append(f"\n**Depends on**: {', '.join(bug.depends_on)}")
        if bug.blocks:
            doc_parts.append(f"\n**Blocks**: {', '.join(bug.blocks)}")

        # Add keywords
        if bug.keywords:
            doc_parts.append(f"\n**Keywords**: {', '.join(bug.keywords)}")

        # Add bug URL
        doc_parts.append(f"\n**Bug URL**: {bug.url}")

        return '\n'.join(doc_parts)

    def _extract_resolution_comment(self, comments: List[Dict[str, Any]]) -> Optional[str]:
        """Extract the resolution/fix comment from bug comments"""
        # Look for comments containing resolution keywords
        resolution_keywords = ['fixed', 'resolved', 'solution', 'patch', 'commit']

        for comment in reversed(comments):  # Start from most recent
            text = comment.get('text', '').lower()
            if any(keyword in text for keyword in resolution_keywords):
                return comment.get('text', '')

        # If no resolution comment found, return last comment
        if comments:
            return comments[-1].get('text', '')

        return None

    def _extract_code_changes(self, comments: List[Dict[str, Any]]) -> str:
        """Extract code changes mentioned in comments"""
        code_patterns = [
            r'commit\s+([a-f0-9]{7,40})',  # Git commit hashes
            r'diff\s+.*',  # Diff references
            r'file:\s+[\w/]+\.(c|cpp|h|hpp|py)',  # File paths
            r'function:\s+\w+',  # Function names
        ]

        code_mentions = []
        for comment in comments:
            text = comment.get('text', '')
            for pattern in code_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                code_mentions.extend(matches)

        if code_mentions:
            return '\n'.join(f"- {mention}" for mention in code_mentions[:10])

        return ''

    def determine_domain(self, bug: BugData) -> str:
        """Determine domain from bug product/component"""
        # Check product mapping
        for key, domain in self.domain_mapping.items():
            if key.lower() in bug.product.lower() or key.lower() in bug.component.lower():
                return domain

        return 'general'

    def determine_classification(self, bug: BugData) -> str:
        """Determine security classification from bug severity"""
        severity = bug.severity.lower()
        return self.security_mapping.get(severity, 'internal')

    async def sync_closed_bugs(self,
                               rag_pipeline,
                               since_date: Optional[datetime] = None,
                               products: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Sync closed bugs into RAG system

        Args:
            rag_pipeline: RAG pipeline instance for ingestion
            since_date: Only sync bugs closed after this date
            products: List of products to sync

        Returns:
            Dictionary with sync statistics
        """
        stats = {
            'fetched': 0,
            'ingested': 0,
            'skipped': 0,
            'errors': 0
        }

        try:
            # Default to last 30 days if not specified
            if since_date is None:
                since_date = datetime.now() - timedelta(days=30)

            # Fetch closed/resolved bugs
            bugs = await self.fetch_bugs(
                status=['RESOLVED', 'CLOSED', 'VERIFIED'],
                resolution=['FIXED'],
                product=products,
                changed_after=since_date,
                limit=500
            )

            stats['fetched'] = len(bugs)

            # Ingest each bug
            for bug in bugs:
                try:
                    # Format bug as document
                    document_content = self.format_bug_as_document(bug)

                    # Determine domain and classification
                    domain = self.determine_domain(bug)
                    classification = self.determine_classification(bug)

                    # Ingest into RAG
                    success = await rag_pipeline.ingest_document(
                        content=document_content,
                        security_classification=classification,
                        domain=domain,
                        metadata={
                            'source': 'bugzilla',
                            'bug_id': bug.bug_id,
                            'product': bug.product,
                            'component': bug.component,
                            'status': bug.status,
                            'resolution': bug.resolution,
                            'severity': bug.severity,
                            'keywords': bug.keywords,
                            'url': bug.url
                        }
                    )

                    if success:
                        stats['ingested'] += 1
                    else:
                        stats['skipped'] += 1

                except Exception as e:
                    self.logger.error(f"Error ingesting bug {bug.bug_id}: {e}")
                    stats['errors'] += 1

            self.logger.info(
                f"Bugzilla sync completed: "
                f"fetched={stats['fetched']}, "
                f"ingested={stats['ingested']}, "
                f"errors={stats['errors']}"
            )

            return stats

        except Exception as e:
            self.logger.error(f"Error syncing bugs: {e}")
            return stats

    async def setup_periodic_sync(self,
                                  rag_pipeline,
                                  interval_hours: int = 24,
                                  products: Optional[List[str]] = None):
        """
        Setup periodic sync of bugs

        Args:
            rag_pipeline: RAG pipeline instance
            interval_hours: Sync interval in hours
            products: List of products to sync
        """
        self.logger.info(f"Starting periodic Bugzilla sync every {interval_hours} hours")

        while True:
            try:
                # Calculate since_date (last sync interval)
                since_date = datetime.now() - timedelta(hours=interval_hours + 1)

                # Perform sync
                stats = await self.sync_closed_bugs(rag_pipeline, since_date, products)

                self.logger.info(f"Periodic sync completed: {stats}")

                # Wait for next interval
                await asyncio.sleep(interval_hours * 3600)

            except Exception as e:
                self.logger.error(f"Error in periodic sync: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(300)  # 5 minutes
