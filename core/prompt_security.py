"""
Prompt Security Module
Protects against prompt injection, jailbreaking, and other prompt-based attacks
"""

import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ThreatLevel(Enum):
    """Threat level classification"""
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"
    CRITICAL = "critical"

@dataclass
class SecurityAnalysis:
    """Results of security analysis"""
    is_safe: bool
    threat_level: ThreatLevel
    threats_detected: List[str]
    confidence: float
    sanitized_input: Optional[str] = None
    details: Dict[str, Any] = None

class PromptInjectionDetector:
    """
    Detects and prevents prompt injection attacks
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Prompt injection patterns
        self.injection_patterns = [
            # Instruction override attempts
            (r'ignore\s+(previous|above|all|the)\s+(instructions?|rules?|prompts?)', ThreatLevel.CRITICAL),
            (r'forget\s+(previous|all|everything|your)\s+(context|instructions?|prompts?)', ThreatLevel.CRITICAL),
            (r'disregard\s+(previous|above|all|the)\s+(instructions?|rules?)', ThreatLevel.CRITICAL),

            # Role hijacking
            (r'you\s+are\s+now\s+(a|an|\w+)', ThreatLevel.DANGEROUS),
            (r'act\s+as\s+(a|an|\w+)', ThreatLevel.DANGEROUS),
            (r'pretend\s+(you|to)\s+(are|be)', ThreatLevel.DANGEROUS),
            (r'roleplay\s+as', ThreatLevel.DANGEROUS),

            # System prompt manipulation
            (r'<\|.*?\|>', ThreatLevel.CRITICAL),  # Special tokens
            (r'\[SYSTEM\]', ThreatLevel.CRITICAL),
            (r'system:', ThreatLevel.DANGEROUS),
            (r'assistant:', ThreatLevel.SUSPICIOUS),
            (r'human:', ThreatLevel.SUSPICIOUS),

            # Instruction injection
            (r'new\s+instructions?:', ThreatLevel.CRITICAL),
            (r'updated\s+rules?:', ThreatLevel.CRITICAL),
            (r'override\s+(instructions?|rules?|settings?)', ThreatLevel.CRITICAL),

            # Jailbreak attempts
            (r'jailbreak', ThreatLevel.CRITICAL),
            (r'DAN\s+(mode|prompt)', ThreatLevel.CRITICAL),  # Do Anything Now
            (r'opposite\s+mode', ThreatLevel.DANGEROUS),
            (r'evil\s+(mode|version)', ThreatLevel.DANGEROUS),

            # Context manipulation
            (r'end\s+of\s+(context|conversation|chat)', ThreatLevel.DANGEROUS),
            (r'start\s+new\s+(context|conversation|session)', ThreatLevel.DANGEROUS),
            (r'reset\s+(context|conversation|memory)', ThreatLevel.DANGEROUS),

            # Output control attempts
            (r'output\s+only', ThreatLevel.SUSPICIOUS),
            (r'only\s+(say|respond|output|return)', ThreatLevel.SUSPICIOUS),
            (r'directly\s+(output|return|give)', ThreatLevel.SUSPICIOUS),

            # Encoding tricks
            (r'base64\s*:', ThreatLevel.SUSPICIOUS),
            (r'rot13\s*:', ThreatLevel.SUSPICIOUS),
            (r'hex\s*:', ThreatLevel.SUSPICIOUS),

            # Prompt leakage attempts
            (r'(show|reveal|display)\s+(your|the)\s+(prompt|instructions?|system\s+message)', ThreatLevel.DANGEROUS),
            (r'what\s+(are|is)\s+your\s+(instructions?|prompt|system\s+message)', ThreatLevel.DANGEROUS),
            (r'repeat\s+(your|the)\s+(instructions?|prompt)', ThreatLevel.DANGEROUS),
        ]

        # Suspicious patterns (lower severity)
        self.suspicious_patterns = [
            r'\\x[0-9a-f]{2}',  # Hex encoding
            r'\\u[0-9a-f]{4}',  # Unicode encoding
            r'\.\.\.',  # Command chaining
            r'&&',  # Command chaining
            r'\|\|',  # Logical operators
            r';\s*$',  # SQL/command injection hints
        ]

        # Allowed special characters that might be flagged
        self.benign_patterns = [
            r'C\+\+',  # C++ is legitimate
            r'\.NET',  # .NET is legitimate
            r'SQL\s+query',  # SQL queries are legitimate technical content
        ]

    def analyze(self, text: str) -> SecurityAnalysis:
        """
        Analyze text for prompt injection threats

        Args:
            text: Input text to analyze

        Returns:
            SecurityAnalysis object with threat assessment
        """
        threats_detected = []
        max_threat_level = ThreatLevel.SAFE
        confidence = 1.0

        # Check if text matches benign patterns
        is_benign = any(re.search(pattern, text, re.IGNORECASE)
                       for pattern in self.benign_patterns)

        if not is_benign:
            # Check injection patterns
            for pattern, threat_level in self.injection_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    threat_name = f"Injection pattern: {pattern[:50]}"
                    threats_detected.append(threat_name)

                    # Update max threat level
                    if threat_level.value == "critical":
                        max_threat_level = ThreatLevel.CRITICAL
                    elif threat_level.value == "dangerous" and max_threat_level != ThreatLevel.CRITICAL:
                        max_threat_level = ThreatLevel.DANGEROUS
                    elif threat_level.value == "suspicious" and max_threat_level == ThreatLevel.SAFE:
                        max_threat_level = ThreatLevel.SUSPICIOUS

                    self.logger.warning(f"Potential prompt injection detected: {threat_name}")

            # Check suspicious patterns
            for pattern in self.suspicious_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    threats_detected.append(f"Suspicious pattern: {pattern}")
                    if max_threat_level == ThreatLevel.SAFE:
                        max_threat_level = ThreatLevel.SUSPICIOUS

        # Additional heuristics
        # Check for excessive special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and c not in ' .,!?-') / max(len(text), 1)
        if special_char_ratio > 0.3:
            threats_detected.append("High special character ratio")
            if max_threat_level == ThreatLevel.SAFE:
                max_threat_level = ThreatLevel.SUSPICIOUS

        # Check for very long inputs (possible overflow/DOS)
        if len(text) > 10000:
            threats_detected.append("Excessively long input")
            max_threat_level = ThreatLevel.SUSPICIOUS

        # Determine if safe
        is_safe = max_threat_level == ThreatLevel.SAFE

        # Calculate confidence
        if threats_detected:
            confidence = min(0.5 + (len(threats_detected) * 0.1), 1.0)

        return SecurityAnalysis(
            is_safe=is_safe,
            threat_level=max_threat_level,
            threats_detected=threats_detected,
            confidence=confidence,
            sanitized_input=self._sanitize(text) if not is_safe else text,
            details={
                'pattern_matches': len(threats_detected),
                'text_length': len(text),
                'special_char_ratio': special_char_ratio
            }
        )

    def _sanitize(self, text: str) -> str:
        """
        Sanitize potentially malicious input

        Args:
            text: Input text to sanitize

        Returns:
            Sanitized text
        """
        sanitized = text

        # Remove special tokens
        sanitized = re.sub(r'<\|.*?\|>', '', sanitized)

        # Remove [SYSTEM] tags
        sanitized = re.sub(r'\[SYSTEM\]', '', sanitized, flags=re.IGNORECASE)

        # Remove obvious instruction overrides
        for pattern, _ in self.injection_patterns[:5]:  # Top 5 most dangerous
            sanitized = re.sub(pattern, '[FILTERED]', sanitized, flags=re.IGNORECASE)

        return sanitized.strip()

    def is_safe(self, text: str, max_threat_level: ThreatLevel = ThreatLevel.SUSPICIOUS) -> bool:
        """
        Quick check if text is safe

        Args:
            text: Input text
            max_threat_level: Maximum acceptable threat level

        Returns:
            True if text is safe, False otherwise
        """
        analysis = self.analyze(text)
        return analysis.threat_level.value <= max_threat_level.value


class DataLossPrevention:
    """
    Data Loss Prevention - scans outputs for sensitive information leakage
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Sensitive data patterns
        self.sensitive_patterns = {
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'api_key': r'\b[A-Za-z0-9]{32,}\b',
            'password': r'(?i)(password|pwd|pass)[\s=:]+[^\s]+',
            'secret_key': r'(?i)(secret|key|token)[\s=:]+[A-Za-z0-9+/=]{20,}',
            'aws_key': r'AKIA[0-9A-Z]{16}',
            'private_key': r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
        }

        # Classification markers
        self.classification_markers = [
            r'\b(SECRET|CLASSIFIED|TOP\s+SECRET|CONFIDENTIAL|RESTRICTED)\b',
            r'\b(ITAR|EXPORT\s+CONTROL)\b',
            r'\b(PROPRIETARY|INTERNAL\s+USE\s+ONLY)\b',
        ]

    def scan_output(self,
                   text: str,
                   user_clearance: str = "internal",
                   redact: bool = True) -> Tuple[str, List[str]]:
        """
        Scan output for sensitive information

        Args:
            text: Output text to scan
            user_clearance: User's security clearance level
            redact: Whether to redact found sensitive data

        Returns:
            Tuple of (processed_text, list_of_violations)
        """
        violations = []
        processed_text = text

        # Check for sensitive data patterns
        for data_type, pattern in self.sensitive_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                violations.append(f"{data_type}: {len(matches)} occurrences")

                if redact:
                    processed_text = re.sub(
                        pattern,
                        f'[REDACTED-{data_type.upper()}]',
                        processed_text
                    )

        # Check for classification markers above user clearance
        for pattern in self.classification_markers:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(f"Classification marker found: {pattern}")

        if violations:
            self.logger.warning(f"DLP violations detected: {violations}")

        return processed_text, violations


class PromptSecurityManager:
    """
    Comprehensive prompt security manager
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        self.injection_detector = PromptInjectionDetector()
        self.dlp = DataLossPrevention()

        # Security settings
        self.max_threat_level = ThreatLevel[
            self.config.get('max_threat_level', 'SUSPICIOUS').upper()
        ]
        self.enable_dlp = self.config.get('enable_dlp', True)
        self.auto_sanitize = self.config.get('auto_sanitize', True)

    def validate_input(self, user_input: str) -> SecurityAnalysis:
        """
        Validate user input for security threats

        Args:
            user_input: User input to validate

        Returns:
            SecurityAnalysis object
        """
        return self.injection_detector.analyze(user_input)

    def validate_output(self,
                       output: str,
                       user_clearance: str = "internal") -> Tuple[str, List[str]]:
        """
        Validate and sanitize output before sending to user

        Args:
            output: Output to validate
            user_clearance: User's security clearance

        Returns:
            Tuple of (sanitized_output, violations_list)
        """
        if self.enable_dlp:
            return self.dlp.scan_output(output, user_clearance, redact=True)
        return output, []

    def process_query(self,
                     query: str,
                     user_clearance: str = "internal") -> Tuple[bool, str, SecurityAnalysis]:
        """
        Process query with full security checks

        Args:
            query: User query
            user_clearance: User's security clearance

        Returns:
            Tuple of (is_allowed, processed_query, analysis)
        """
        # Analyze input
        analysis = self.injection_detector.analyze(query)

        # Check if threat level is acceptable
        is_allowed = analysis.threat_level.value <= self.max_threat_level.value

        # Get processed query
        if is_allowed:
            processed_query = query
        elif self.auto_sanitize and analysis.sanitized_input:
            processed_query = analysis.sanitized_input
            is_allowed = True
            self.logger.info("Query sanitized and allowed")
        else:
            processed_query = ""
            self.logger.warning(f"Query blocked due to {analysis.threat_level.value} threat level")

        return is_allowed, processed_query, analysis

    def get_security_report(self) -> Dict[str, Any]:
        """Get security statistics and configuration"""
        return {
            'max_threat_level': self.max_threat_level.value,
            'dlp_enabled': self.enable_dlp,
            'auto_sanitize': self.auto_sanitize,
            'injection_patterns_count': len(self.injection_detector.injection_patterns),
            'sensitive_patterns_count': len(self.dlp.sensitive_patterns),
        }
