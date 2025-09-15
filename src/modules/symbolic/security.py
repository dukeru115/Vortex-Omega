"""
Security Module for Symbolic AI
================================

Provides input validation, sanitization, and security checks.

Author: Team Omega
License: CC BY-NC 4.0
"""

import re
import hashlib
import secrets
import time
from typing import Any, Optional, Dict, List
from dataclasses import dataclass
import ast
import logging

logger = logging.getLogger(__name__)

# Security constants
MAX_INPUT_LENGTH = 100000  # 100KB max
MAX_EXPRESSION_DEPTH = 10
MAX_CLAUSES = 1000
ALLOWED_CHARS_PATTERN = re.compile(r'^[\w\s\d\+\-\*\/\^\(\)\[\]\{\}\=\<\>\.\,\;\:\!\?\&\|\~\%]+$')
DANGEROUS_PATTERNS = [
    r'__[a-z]+__',  # Dunder methods
    r'eval\s*\(',
    r'exec\s*\(',
    r'compile\s*\(',
    r'open\s*\(',
    r'import\s+',
    r'from\s+\w+\s+import',
    r'os\.',
    r'sys\.',
    r'subprocess',
    r'pickle',
    r'marshal',
]


@dataclass
class SecurityContext:
    """Security context for processing"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    rate_limit_remaining: int = 100
    max_processing_time: float = 5.0
    allowed_operations: List[str] = None
    
    def __post_init__(self):
        if self.allowed_operations is None:
            self.allowed_operations = ['read', 'process', 'verify']


class InputSanitizer:
    """Sanitizes and validates input"""
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = MAX_INPUT_LENGTH) -> str:
        """
        Sanitize input text
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
            
        Raises:
            ValueError: If input is invalid or dangerous
        """
        # Check length
        if not text:
            raise ValueError("Empty input")
        
        if len(text) > max_length:
            raise ValueError(f"Input too large: {len(text)} > {max_length}")
        
        # Remove null bytes and control characters
        text = text.replace('\0', '')
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Check for dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                raise ValueError(f"Dangerous pattern detected: {pattern}")
        
        # HTML escape
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        
        return text
    
    @staticmethod
    def validate_expression(expr: str, max_depth: int = MAX_EXPRESSION_DEPTH) -> bool:
        """
        Validate mathematical expression safety
        
        Args:
            expr: Expression to validate
            max_depth: Maximum AST depth allowed
            
        Returns:
            True if safe
        """
        try:
            # Parse to AST
            tree = ast.parse(expr, mode='eval')
            
            # Check depth
            depth = InputSanitizer._get_ast_depth(tree)
            if depth > max_depth:
                logger.warning(f"Expression too deep: {depth} > {max_depth}")
                return False
            
            # Check for dangerous nodes
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)):
                    # Check if it's calling dangerous functions
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            if node.func.id in ['eval', 'exec', 'compile', '__import__']:
                                logger.warning(f"Dangerous function: {node.func.id}")
                                return False
            
            return True
            
        except (SyntaxError, ValueError) as e:
            logger.warning(f"Invalid expression: {e}")
            return False
    
    @staticmethod
    def _get_ast_depth(node: ast.AST, current_depth: int = 0) -> int:
        """Get maximum depth of AST"""
        if not isinstance(node, ast.AST):
            return current_depth
        
        max_depth = current_depth
        for child in ast.iter_child_nodes(node):
            depth = InputSanitizer._get_ast_depth(child, current_depth + 1)
            max_depth = max(max_depth, depth)
        
        return max_depth


class RateLimiter:
    """Rate limiting for API protection"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # user_id -> [(timestamp, count)]
    
    def check_rate_limit(self, user_id: str, current_time: float) -> bool:
        """
        Check if user is within rate limit
        
        Args:
            user_id: User identifier
            current_time: Current timestamp
            
        Returns:
            True if within limit
        """
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        # Clean old requests
        window_start = current_time - self.window_seconds
        self.requests[user_id] = [
            (t, c) for t, c in self.requests[user_id]
            if t > window_start
        ]
        
        # Count requests in window
        total_requests = sum(c for _, c in self.requests[user_id])
        
        if total_requests >= self.max_requests:
            logger.warning(f"Rate limit exceeded for user {user_id}")
            return False
        
        # Add current request
        self.requests[user_id].append((current_time, 1))
        return True
    
    def get_remaining(self, user_id: str, current_time: float) -> int:
        """Get remaining requests for user"""
        if user_id not in self.requests:
            return self.max_requests
        
        window_start = current_time - self.window_seconds
        valid_requests = [
            (t, c) for t, c in self.requests[user_id]
            if t > window_start
        ]
        
        total_requests = sum(c for _, c in valid_requests)
        return max(0, self.max_requests - total_requests)


class SecureCache:
    """Secure caching with encryption"""
    
    def __init__(self, secret_key: Optional[bytes] = None):
        """
        Initialize secure cache
        
        Args:
            secret_key: Encryption key (generated if not provided)
        """
        self.secret_key = secret_key or secrets.token_bytes(32)
        self.cache = {}
    
    def _hash_key(self, key: str) -> str:
        """Create secure hash of cache key"""
        h = hashlib.blake2b(digest_size=32)
        h.update(self.secret_key)
        h.update(key.encode('utf-8'))
        return h.hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        hashed_key = self._hash_key(key)
        return self.cache.get(hashed_key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        hashed_key = self._hash_key(key)
        self.cache[hashed_key] = value
        # In production, implement TTL with timestamps
    
    def delete(self, key: str):
        """Delete from cache"""
        hashed_key = self._hash_key(key)
        self.cache.pop(hashed_key, None)
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()


class SecurityValidator:
    """Main security validation interface"""
    
    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.rate_limiter = RateLimiter()
        self.cache = SecureCache()
    
    def validate_input(self, 
                      text: str,
                      context: SecurityContext) -> Dict[str, Any]:
        """
        Comprehensive input validation
        
        Args:
            text: Input text
            context: Security context
            
        Returns:
            Validation result
        """
        result = {
            'valid': False,
            'sanitized_text': None,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check rate limiting
            if context.user_id:
                if not self.rate_limiter.check_rate_limit(
                    context.user_id, 
                    time.time()
                ):
                    result['errors'].append("Rate limit exceeded")
                    return result
            
            # Sanitize input
            sanitized = self.sanitizer.sanitize_text(text)
            result['sanitized_text'] = sanitized
            
            # Additional validation
            if len(sanitized.split()) > 10000:
                result['warnings'].append("Very large input - may be slow")
            
            result['valid'] = True
            
        except ValueError as e:
            result['errors'].append(str(e))
        except Exception as e:
            logger.error(f"Validation error: {e}")
            result['errors'].append("Validation failed")
        
        return result


# Circuit breaker implementation
class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Failures before opening
            recovery_timeout: Time before retry
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        """
        Call function with circuit breaker protection
        
        Args:
            func: Function to call
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == 'open':
            if self._should_attempt_reset():
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset"""
        return (
            self.last_failure_time and 
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = 'closed'
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


# Export main components
__all__ = [
    'SecurityContext',
    'InputSanitizer',
    'RateLimiter',
    'SecureCache',
    'SecurityValidator',
    'CircuitBreaker'
]