"""
Logging and Monitoring Utilities
=================================

Production-grade logging setup for the RAG system.
"""

import sys
import os
from datetime import datetime
from pathlib import Path

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    json_format: bool = False
):
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        json_format: Use JSON format for structured logging
    """
    # Remove default handler
    logger.remove()
    
    # Console format
    if json_format:
        console_format = "{message}"
    else:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=log_level,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        
        logger.add(
            log_file,
            format=file_format,
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
    
    logger.info(f"Logging configured: level={log_level}, file={log_file}")


class RequestLogger:
    """
    Middleware-style logger for API requests.
    """
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_latency_ms = 0
    
    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        latency_ms: float,
        user_agent: str = None
    ):
        """Log an API request."""
        self.request_count += 1
        self.total_latency_ms += latency_ms
        
        if status_code >= 400:
            self.error_count += 1
            log_func = logger.warning if status_code < 500 else logger.error
        else:
            log_func = logger.info
        
        log_func(
            f"{method} {path} - {status_code} - {latency_ms:.0f}ms"
        )
    
    def get_stats(self) -> dict:
        """Get request statistics."""
        avg_latency = (
            self.total_latency_ms / self.request_count
            if self.request_count > 0 else 0
        )
        
        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "avg_latency_ms": avg_latency
        }


class RAGMetrics:
    """
    Metrics collector for RAG system monitoring.
    """
    
    def __init__(self):
        self.queries = []
        self.retrievals = []
        self.generations = []
    
    def log_query(
        self,
        query: str,
        mode: str,
        retrieval_count: int,
        total_latency_ms: float,
        retrieval_latency_ms: float,
        generation_latency_ms: float,
        confidence: float
    ):
        """Log a RAG query."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query_length": len(query),
            "mode": mode,
            "retrieval_count": retrieval_count,
            "total_latency_ms": total_latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "confidence": confidence
        }
        
        self.queries.append(entry)
        
        # Keep last 1000 queries
        if len(self.queries) > 1000:
            self.queries = self.queries[-1000:]
    
    def get_summary(self) -> dict:
        """Get metrics summary."""
        if not self.queries:
            return {"message": "No queries logged yet"}
        
        import numpy as np
        
        latencies = [q["total_latency_ms"] for q in self.queries]
        confidences = [q["confidence"] for q in self.queries]
        retrieval_counts = [q["retrieval_count"] for q in self.queries]
        
        return {
            "total_queries": len(self.queries),
            "latency": {
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99)
            },
            "confidence": {
                "mean": np.mean(confidences),
                "min": min(confidences),
                "max": max(confidences)
            },
            "avg_retrieval_count": np.mean(retrieval_counts),
            "mode_distribution": {
                mode: sum(1 for q in self.queries if q["mode"] == mode)
                for mode in ["simple", "agent"]
            }
        }


# Global instances
request_logger = RequestLogger()
rag_metrics = RAGMetrics()


# Initialize logging on import
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE")
)
