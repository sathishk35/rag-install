"""
Security Manager for RAG System
Handles access control, data classification, and audit logging
"""

import logging
import hashlib
import jwt
import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import re
import psycopg2

@dataclass
class UserClearance:
    """User security clearance information"""
    user_id: str
    security_level: str
    domains: List[str]
    access_groups: List[str]
    valid_until: datetime
    created_at: datetime
    updated_at: datetime

@dataclass
class AccessDecision:
    """Access control decision"""
    allowed: bool
    reason: str
    filtered_content: Optional[str] = None
    classification_level: Optional[str] = None

class SecurityManager:
    """Manages security and access control for RAG system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Security level hierarchy (lower number = higher clearance)
        self.security_levels = {
            'classified': 4,
            'restricted': 3,
            'confidential': 2,
            'internal': 1,
            'public': 0
        }
        
        # Domain-specific access control
        self.domain_restrictions = {
            'classified': ['radar', 'ew', 'classified_drivers'],
            'restricted': ['radar', 'ew', 'ate', 'restricted_drivers'],
            'confidential': ['drivers', 'embedded', 'radar', 'ate'],
            'internal': ['drivers', 'embedded', 'general', 'ate'],
            'public': ['general', 'public_docs']
        }
        
        # Content classification patterns
        self.classification_patterns = {
            'classified': [
                r'\b(secret|classified|top.?secret)\b',
                r'\b(itar|export.?control)\b',
                r'\b(defense.?classified)\b'
            ],
            'restricted': [
                r'\b(restricted|confidential)\b',
                r'\b(proprietary|internal.?use)\b',
                r'\b(defense.?restricted)\b'
            ],
            'confidential': [
                r'\b(confidential|internal)\b',
                r'\b(company.?confidential)\b',
                r'\b(not.?for.?public)\b'
            ]
        }
        
        # Sensitive data patterns to redact
        self.sensitive_patterns = {
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'api_key': r'\b[A-Za-z0-9]{32,}\b',
            'password': r'(?i)(password|pwd|pass)[\s=:]+[^\s]+',
            'secret_key': r'(?i)(secret|key|token)[\s=:]+[A-Za-z0-9+/=]{20,}'
        }
        
        # Initialize database connection
        self.db_connection = self._init_database()
        
        # Create user access table if not exists
        self._ensure_access_tables()
    
    def _init_database(self) -> psycopg2.extensions.connection:
        """Initialize database connection"""
        try:
            conn = psycopg2.connect(
                host=self.config["database"]["host"],
                port=self.config["database"]["port"],
                database=self.config["database"]["database"],
                user=self.config["database"]["user"]
            )
            self.logger.info("Security Manager connected to database")
            return conn
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _ensure_access_tables(self):
        """Ensure security tables exist"""
        try:
            cursor = self.db_connection.cursor()
            
            # Create security audit table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS security_audit (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT,
                    decision TEXT NOT NULL,
                    reason TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB
                )
            """)
            
            # Create access violations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS access_violations (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    resolved BOOLEAN DEFAULT FALSE,
                    metadata JSONB
                )
            """)
            
            # Create user sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    security_level TEXT NOT NULL,
                    domains TEXT[],
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    ip_address TEXT,
                    user_agent TEXT,
                    active BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_security_audit_user ON security_audit(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_security_audit_timestamp ON security_audit(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_violations_user ON access_violations(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_user ON user_sessions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(active)")
            
            self.db_connection.commit()
            self.logger.info("Security tables ensured")
            
        except Exception as e:
            self.logger.error(f"Error ensuring security tables: {e}")
            raise
    
    async def get_user_clearance(self, user_id: str) -> Dict[str, Any]:
        """Get user security clearance"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT security_clearance, domains, created_at, updated_at
                FROM user_access 
                WHERE user_id = %s
            """, (user_id,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'user_id': user_id,
                    'security_clearance': result[0],
                    'domains': result[1] or [],
                    'created_at': result[2],
                    'updated_at': result[3]
                }
            else:
                # Default clearance for new users
                return {
                    'user_id': user_id,
                    'security_clearance': 'public',
                    'domains': ['general'],
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                }
                
        except Exception as e:
            self.logger.error(f"Error getting user clearance for {user_id}: {e}")
            return {
                'user_id': user_id,
                'security_clearance': 'public',
                'domains': ['general'],
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
    
    async def set_user_clearance(self, 
                               user_id: str, 
                               security_level: str, 
                               domains: List[str],
                               admin_user: str) -> bool:
        """Set user security clearance"""
        try:
            if security_level not in self.security_levels:
                raise ValueError(f"Invalid security level: {security_level}")
            
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO user_access (user_id, security_clearance, domains, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id) 
                DO UPDATE SET 
                    security_clearance = EXCLUDED.security_clearance,
                    domains = EXCLUDED.domains,
                    updated_at = EXCLUDED.updated_at
            """, (user_id, security_level, domains, datetime.now()))
            
            self.db_connection.commit()
            
            # Log the clearance change
            await self._audit_log(
                admin_user, 
                'set_user_clearance',
                f"user:{user_id}",
                'allowed',
                f"Set clearance to {security_level} with domains {domains}"
            )
            
            self.logger.info(f"Set user {user_id} clearance to {security_level}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting user clearance: {e}")
            return False
    
    def get_allowed_security_levels(self, user_security_level: str) -> List[str]:
        """Get security levels that user can access"""
        user_level = self.security_levels.get(user_security_level, 0)
        
        allowed_levels = []
        for level, level_value in self.security_levels.items():
            if level_value <= user_level:
                allowed_levels.append(level)
        
        return allowed_levels
    
    async def check_document_access(self, 
                                  user_id: str, 
                                  document_path: str,
                                  document_classification: str,
                                  document_domain: str) -> AccessDecision:
        """Check if user can access a specific document"""
        try:
            user_clearance = await self.get_user_clearance(user_id)
            
            # Check security level
            user_level = self.security_levels.get(user_clearance['security_clearance'], 0)
            doc_level = self.security_levels.get(document_classification, 4)
            
            if user_level < doc_level:
                await self._audit_log(
                    user_id, 
                    'document_access_denied',
                    document_path,
                    'denied',
                    f"Insufficient clearance: user {user_clearance['security_clearance']} < document {document_classification}"
                )
                
                return AccessDecision(
                    allowed=False,
                    reason=f"Insufficient security clearance",
                    classification_level=document_classification
                )
            
            # Check domain access
            if document_domain not in user_clearance.get('domains', []):
                await self._audit_log(
                    user_id,
                    'document_access_denied',
                    document_path,
                    'denied',
                    f"Domain not authorized: {document_domain}"
                )
                
                return AccessDecision(
                    allowed=False,
                    reason=f"Domain access not authorized",
                    classification_level=document_classification
                )
            
            # Access granted
            await self._audit_log(
                user_id,
                'document_access_granted',
                document_path,
                'allowed',
                f"Access granted to {document_classification} document in {document_domain}"
            )
            
            return AccessDecision(
                allowed=True,
                reason="Access granted",
                classification_level=document_classification
            )
            
        except Exception as e:
            self.logger.error(f"Error checking document access: {e}")
            return AccessDecision(
                allowed=False,
                reason="Security check error",
                classification_level=document_classification
            )
    
    async def classify_content(self, content: str) -> str:
        """Automatically classify content based on patterns"""
        content_lower = content.lower()
        
        # Check for classification markers in order of sensitivity
        for classification, patterns in self.classification_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    return classification
        
        # Default classification
        return 'internal'
    
    async def sanitize_content(self, 
                             content: str, 
                             user_clearance: Dict[str, Any],
                             aggressive: bool = False) -> str:
        """Sanitize content based on user clearance"""
        sanitized_content = content
        
        try:
            # Get content classification
            content_classification = await self.classify_content(content)
            
            # Check if user can see this classification level
            user_level = self.security_levels.get(user_clearance['security_clearance'], 0)
            content_level = self.security_levels.get(content_classification, 4)
            
            if user_level < content_level:
                return "[REDACTED - Insufficient Clearance]"
            
            # Remove sensitive patterns if aggressive mode or lower clearance
            if aggressive or user_clearance['security_clearance'] in ['public', 'internal']:
                for pattern_name, pattern in self.sensitive_patterns.items():
                    sanitized_content = re.sub(
                        pattern, 
                        f"[REDACTED_{pattern_name.upper()}]", 
                        sanitized_content,
                        flags=re.IGNORECASE
                    )
            
            return sanitized_content
            
        except Exception as e:
            self.logger.error(f"Error sanitizing content: {e}")
            return "[REDACTED - Sanitization Error]"
    
    async def create_session(self, 
                           user_id: str, 
                           session_duration_hours: int = 8) -> str:
        """Create a secure user session"""
        try:
            user_clearance = await self.get_user_clearance(user_id)
            
            # Generate session ID
            session_data = f"{user_id}:{datetime.now().isoformat()}"
            session_id = hashlib.sha256(session_data.encode()).hexdigest()
            
            # Store session in database
    async def create_session(self, 
                           user_id: str, 
                           session_duration_hours: int = 8) -> str:
        """Create a secure user session"""
        try:
            user_clearance = await self.get_user_clearance(user_id)
            
            # Generate session ID
            session_data = f"{user_id}:{datetime.now().isoformat()}"
            session_id = hashlib.sha256(session_data.encode()).hexdigest()
            
            # Store session in database
            expires_at = datetime.now() + timedelta(hours=session_duration_hours)
            
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO user_sessions 
                (session_id, user_id, security_level, domains, expires_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                session_id,
                user_id,
                user_clearance['security_clearance'],
                user_clearance['domains'],
                expires_at
            ))
            
            self.db_connection.commit()
            
            await self._audit_log(user_id, 'session_created', session_id, 'allowed', 'New session created')
            
            self.logger.info(f"Created session for user {user_id}: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Error creating session: {e}")
            raise
    
    async def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate and refresh user session"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT user_id, security_level, domains, expires_at, last_activity
                FROM user_sessions 
                WHERE session_id = %s AND active = true
            """, (session_id,))
            
            result = cursor.fetchone()
            if not result:
                return None
            
            user_id, security_level, domains, expires_at, last_activity = result
            
            # Check if session has expired
            if expires_at < datetime.now():
                await self._invalidate_session(session_id, "Session expired")
                return None
            
            # Update last activity
            cursor.execute("""
                UPDATE user_sessions 
                SET last_activity = %s
                WHERE session_id = %s
            """, (datetime.now(), session_id))
            
            self.db_connection.commit()
            
            return {
                'session_id': session_id,
                'user_id': user_id,
                'security_clearance': security_level,
                'domains': domains,
                'expires_at': expires_at,
                'last_activity': last_activity
            }
            
        except Exception as e:
            self.logger.error(f"Error validating session: {e}")
            return None
    
    async def _invalidate_session(self, session_id: str, reason: str):
        """Invalidate a user session"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                UPDATE user_sessions 
                SET active = false
                WHERE session_id = %s
            """, (session_id,))
            
            self.db_connection.commit()
            
            # Get user_id for audit log
            cursor.execute("SELECT user_id FROM user_sessions WHERE session_id = %s", (session_id,))
            result = cursor.fetchone()
            if result:
                await self._audit_log(result[0], 'session_invalidated', session_id, 'allowed', reason)
            
        except Exception as e:
            self.logger.error(f"Error invalidating session: {e}")
    
    async def logout_session(self, session_id: str) -> bool:
        """Logout and invalidate session"""
        try:
            await self._invalidate_session(session_id, "User logout")
            return True
        except Exception as e:
            self.logger.error(f"Error during logout: {e}")
            return False
    
    async def detect_security_violations(self, 
                                       user_id: str, 
                                       query: str,
                                       retrieved_docs: List[str]) -> List[Dict[str, Any]]:
        """Detect potential security violations"""
        violations = []
        
        try:
            user_clearance = await self.get_user_clearance(user_id)
            
            # Check for suspicious query patterns
            suspicious_patterns = [
                r'(?i)(password|secret|key|token)',
                r'(?i)(classified|confidential|restricted)',
                r'(?i)(admin|root|sudo)',
                r'(?i)(database|db|sql|select)',
                r'(?i)(bypass|hack|exploit)',
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, query):
                    violations.append({
                        'type': 'suspicious_query',
                        'severity': 'medium',
                        'description': f'Query contains suspicious pattern: {pattern}',
                        'pattern': pattern
                    })
            
            # Check for access to documents above clearance
            for doc_path in retrieved_docs:
                # Simple heuristic - check if path contains security indicators
                if any(keyword in doc_path.lower() for keyword in ['classified', 'secret', 'restricted']):
                    if user_clearance['security_clearance'] in ['public', 'internal']:
                        violations.append({
                            'type': 'unauthorized_access_attempt',
                            'severity': 'high',
                            'description': f'Attempted access to restricted document: {doc_path}',
                            'document': doc_path
                        })
            
            # Check for unusual access patterns
            if len(retrieved_docs) > 50:
                violations.append({
                    'type': 'bulk_access',
                    'severity': 'medium',
                    'description': f'Large number of documents accessed: {len(retrieved_docs)}',
                    'count': len(retrieved_docs)
                })
            
            # Log violations
            for violation in violations:
                await self._log_violation(user_id, violation)
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Error detecting security violations: {e}")
            return []
    
    async def _log_violation(self, user_id: str, violation: Dict[str, Any]):
        """Log security violation"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO access_violations 
                (user_id, violation_type, description, severity, metadata)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                user_id,
                violation['type'],
                violation['description'],
                violation['severity'],
                json.dumps(violation)
            ))
            
            self.db_connection.commit()
            
            self.logger.warning(f"Security violation logged for user {user_id}: {violation['type']}")
            
        except Exception as e:
            self.logger.error(f"Error logging violation: {e}")
    
    async def _audit_log(self, 
                       user_id: str, 
                       action: str, 
                       resource: str,
                       decision: str, 
                       reason: str,
                       metadata: Dict[str, Any] = None):
        """Log security audit event"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO security_audit 
                (user_id, action, resource, decision, reason, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                user_id, action, resource, decision, reason,
                json.dumps(metadata) if metadata else None
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"Error writing audit log: {e}")
    
    async def get_user_activity_summary(self, 
                                      user_id: str, 
                                      days: int = 7) -> Dict[str, Any]:
        """Get user activity summary for security monitoring"""
        try:
            cursor = self.db_connection.cursor()
            
            # Get recent queries
            cursor.execute("""
                SELECT COUNT(*) as query_count,
                       COUNT(DISTINCT DATE(timestamp)) as active_days
                FROM query_log 
                WHERE user_id = %s 
                AND timestamp > NOW() - INTERVAL '%s days'
            """, (user_id, days))
            
            query_stats = cursor.fetchone()
            
            # Get security events
            cursor.execute("""
                SELECT action, COUNT(*) as count
                FROM security_audit 
                WHERE user_id = %s 
                AND timestamp > NOW() - INTERVAL '%s days'
                GROUP BY action
            """, (user_id, days))
            
            security_events = dict(cursor.fetchall())
            
            # Get violations
            cursor.execute("""
                SELECT violation_type, severity, COUNT(*) as count
                FROM access_violations 
                WHERE user_id = %s 
                AND timestamp > NOW() - INTERVAL '%s days'
                GROUP BY violation_type, severity
            """, (user_id, days))
            
            violations = [
                {'type': row[0], 'severity': row[1], 'count': row[2]}
                for row in cursor.fetchall()
            ]
            
            # Calculate risk score
            risk_score = self._calculate_user_risk_score(
                query_stats[0] if query_stats else 0,
                len(violations),
                security_events
            )
            
            return {
                'user_id': user_id,
                'period_days': days,
                'query_count': query_stats[0] if query_stats else 0,
                'active_days': query_stats[1] if query_stats else 0,
                'security_events': security_events,
                'violations': violations,
                'risk_score': risk_score,
                'last_activity': self._get_last_activity(user_id)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user activity summary: {e}")
            return {}
    
    def _calculate_user_risk_score(self, 
                                 query_count: int, 
                                 violation_count: int,
                                 security_events: Dict[str, int]) -> float:
        """Calculate user risk score (0-1, higher is riskier)"""
        risk_score = 0.0
        
        # Query volume risk
        if query_count > 1000:  # Very high query volume
            risk_score += 0.3
        elif query_count > 500:
            risk_score += 0.2
        elif query_count > 100:
            risk_score += 0.1
        
        # Violation risk
        if violation_count > 0:
            risk_score += min(violation_count * 0.1, 0.4)
        
        # Failed access attempts
        failed_access = security_events.get('document_access_denied', 0)
        if failed_access > 10:
            risk_score += 0.3
        elif failed_access > 5:
            risk_score += 0.2
        
        return min(risk_score, 1.0)
    
    def _get_last_activity(self, user_id: str) -> Optional[datetime]:
        """Get user's last activity timestamp"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT MAX(timestamp) 
                FROM query_log 
                WHERE user_id = %s
            """, (user_id,))
            
            result = cursor.fetchone()
            return result[0] if result and result[0] else None
            
        except Exception as e:
            self.logger.error(f"Error getting last activity: {e}")
            return None
    
    async def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data"""
        try:
            cursor = self.db_connection.cursor()
            
            # Active users in last 24 hours
            cursor.execute("""
                SELECT COUNT(DISTINCT user_id) 
                FROM query_log 
                WHERE timestamp > NOW() - INTERVAL '24 hours'
            """)
            active_users_24h = cursor.fetchone()[0]
            
            # Recent violations
            cursor.execute("""
                SELECT violation_type, severity, COUNT(*) as count
                FROM access_violations 
                WHERE timestamp > NOW() - INTERVAL '7 days'
                AND resolved = false
                GROUP BY violation_type, severity
                ORDER BY count DESC
            """)
            recent_violations = [
                {'type': row[0], 'severity': row[1], 'count': row[2]}
                for row in cursor.fetchall()
            ]
            
            # Security events summary
            cursor.execute("""
                SELECT action, decision, COUNT(*) as count
                FROM security_audit 
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                GROUP BY action, decision
                ORDER BY count DESC
            """)
            security_events = [
                {'action': row[0], 'decision': row[1], 'count': row[2]}
                for row in cursor.fetchall()
            ]
            
            # Top accessing users
            cursor.execute("""
                SELECT user_id, COUNT(*) as queries
                FROM query_log 
                WHERE timestamp > NOW() - INTERVAL '7 days'
                GROUP BY user_id
                ORDER BY queries DESC
                LIMIT 10
            """)
            top_users = [
                {'user_id': row[0], 'queries': row[1]}
                for row in cursor.fetchall()
            ]
            
            # Classification distribution
            cursor.execute("""
                SELECT security_classification, COUNT(*) as count
                FROM documents
                GROUP BY security_classification
            """)
            classification_dist = dict(cursor.fetchall())
            
            return {
                'timestamp': datetime.now().isoformat(),
                'active_users_24h': active_users_24h,
                'recent_violations': recent_violations,
                'security_events_24h': security_events,
                'top_users_7d': top_users,
                'document_classification': classification_dist,
                'system_status': 'operational'  # Could be enhanced with actual health checks
            }
            
        except Exception as e:
            self.logger.error(f"Error generating security dashboard: {e}")
            return {'error': str(e)}
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                UPDATE user_sessions 
                SET active = false
                WHERE expires_at < NOW() AND active = true
            """)
            
            expired_count = cursor.rowcount
            self.db_connection.commit()
            
            if expired_count > 0:
                self.logger.info(f"Cleaned up {expired_count} expired sessions")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired sessions: {e}")
    
    def close(self):
        """Close database connection"""
        try:
            if self.db_connection:
                self.db_connection.close()
        except Exception as e:
            self.logger.error(f"Error closing security manager: {e}")