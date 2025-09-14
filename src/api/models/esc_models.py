"""
ESC (Echo-Semantic Converter) API Models
========================================

Pydantic models for ESC token processing, semantic analysis, and constitutional filtering.
Implements NFCS v2.4.3 ESC Module 2.1 specifications.

Author: Team Ω (Omega)
Date: September 13, 2025
Version: 2.4.3
"""

from pydantic import BaseModel, Field, ConfigDict, validator
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import numpy as np


class ProcessingMode(str, Enum):
    """ESC processing modes"""
    STANDARD = "standard"
    CONSTITUTIONAL = "constitutional"
    SEMANTIC_ONLY = "semantic_only"
    ATTENTION_ANALYSIS = "attention_analysis"
    FULL_PIPELINE = "full_pipeline"


class TokenType(str, Enum):
    """Token classification types"""
    WORD = "word"
    PUNCTUATION = "punctuation"
    NUMBER = "number"
    SPECIAL = "special"
    UNKNOWN = "unknown"


class SemanticField(BaseModel):
    """Semantic field analysis results"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dimensionality": 512,
                "coherence_score": 0.87,
                "stability_measure": 0.92,
                "semantic_density": 0.75
            }
        }
    )
    
    dimensionality: int = Field(..., gt=0, description="Semantic space dimensionality")
    coherence_score: float = Field(..., ge=0.0, le=1.0, description="Field coherence (0-1)")
    stability_measure: float = Field(..., ge=0.0, le=1.0, description="Temporal stability (0-1)")
    semantic_density: float = Field(..., ge=0.0, le=1.0, description="Information density (0-1)")
    field_topology: Optional[Dict[str, float]] = Field(
        default=None,
        description="Topological characteristics"
    )
    attention_weights: Optional[List[float]] = Field(
        default=None,
        description="Multi-scale attention weights"
    )


class TokenAnalysis(BaseModel):
    """Individual token analysis results"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "token": "intelligence",
                "token_type": "word",
                "semantic_embedding": "[0.12, -0.45, 0.78, ...]",
                "attention_score": 0.85,
                "constitutional_compliance": 0.98
            }
        }
    )
    
    token: str = Field(..., description="Original token text")
    token_type: TokenType = Field(..., description="Token classification")
    position: int = Field(..., ge=0, description="Position in sequence")
    semantic_embedding: Optional[List[float]] = Field(
        default=None,
        description="High-dimensional semantic embedding"
    )
    attention_score: float = Field(..., ge=0.0, le=1.0, description="Attention weight (0-1)")
    constitutional_compliance: float = Field(
        ..., ge=0.0, le=1.0,
        description="Constitutional safety score (0-1)"
    )
    semantic_anchors: Optional[List[str]] = Field(
        default=None,
        description="Identified semantic anchor points"
    )
    risk_factors: Optional[Dict[str, float]] = Field(
        default=None,
        description="Risk factor analysis"
    )


class ConstitutionalFilter(BaseModel):
    """Constitutional filtering results"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "overall_compliance": 0.96,
                "policy_violations": [],
                "filtered_tokens": ["inappropriate_term"],
                "safety_score": 0.94
            }
        }
    )
    
    overall_compliance: float = Field(
        ..., ge=0.0, le=1.0,
        description="Overall constitutional compliance (0-1)"
    )
    policy_violations: List[str] = Field(
        default_factory=list,
        description="List of violated policies"
    )
    filtered_tokens: List[str] = Field(
        default_factory=list,
        description="Tokens filtered by constitutional constraints"
    )
    safety_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Safety assessment score (0-1)"
    )
    enforcement_actions: Optional[List[str]] = Field(
        default=None,
        description="Applied enforcement actions"
    )


class ESCProcessRequest(BaseModel):
    """ESC token processing request"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tokens": ["The", "neural", "field", "exhibits", "coherent", "behavior"],
                "processing_mode": "full_pipeline",
                "context": "Scientific analysis of neural dynamics",
                "enable_constitutional_filtering": true,
                "return_embeddings": false
            }
        }
    )
    
    tokens: List[str] = Field(..., min_items=1, description="Input tokens for processing")
    processing_mode: ProcessingMode = Field(
        default=ProcessingMode.STANDARD,
        description="ESC processing mode"
    )
    context: Optional[str] = Field(default=None, description="Processing context information")
    enable_constitutional_filtering: bool = Field(
        default=True,
        description="Enable constitutional safety filtering"
    )
    return_embeddings: bool = Field(
        default=False,
        description="Include semantic embeddings in response"
    )
    attention_layers: Optional[List[int]] = Field(
        default=None,
        description="Specific attention layers to analyze"
    )
    custom_parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom processing parameters"
    )

    @validator('tokens')
    def validate_tokens(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one token is required")
        if len(v) > 1000:  # Reasonable limit
            raise ValueError("Maximum 1000 tokens per request")
        return v


class ESCProcessResponse(BaseModel):
    """ESC token processing response"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": true,
                "processing_time_ms": 45.2,
                "token_analyses": "...",
                "semantic_field": "...",
                "constitutional_filter": "...",
                "sequence_coherence": 0.89
            }
        }
    )
    
    success: bool = Field(..., description="Processing success status")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in milliseconds")
    token_analyses: List[TokenAnalysis] = Field(
        ...,
        description="Per-token analysis results"
    )
    semantic_field: SemanticField = Field(..., description="Overall semantic field analysis")
    constitutional_filter: ConstitutionalFilter = Field(
        ...,
        description="Constitutional filtering results"
    )
    sequence_coherence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Overall sequence coherence score (0-1)"
    )
    
    # Optional detailed results
    attention_maps: Optional[List[List[float]]] = Field(
        default=None,
        description="Multi-scale attention maps"
    )
    semantic_graph: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Semantic relationship graph"
    )
    performance_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Processing performance metrics"
    )
    warnings: Optional[List[str]] = Field(
        default=None,
        description="Processing warnings or notices"
    )


class ESCConfiguration(BaseModel):
    """ESC system configuration"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "semantic_dimensions": 512,
                "attention_heads": 8,
                "constitutional_threshold": 0.8,
                "enable_adaptive_vocabulary": true
            }
        }
    )
    
    semantic_dimensions: int = Field(
        default=512, gt=0, le=2048,
        description="Semantic embedding dimensions"
    )
    attention_heads: int = Field(
        default=8, gt=0, le=32,
        description="Number of attention heads"
    )
    constitutional_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Constitutional filtering threshold"
    )
    enable_adaptive_vocabulary: bool = Field(
        default=True,
        description="Enable adaptive vocabulary learning"
    )
    max_sequence_length: int = Field(
        default=512, gt=0, le=2048,
        description="Maximum sequence length"
    )
    cache_embeddings: bool = Field(
        default=True,
        description="Cache semantic embeddings"
    )


class ESCSystemStatus(BaseModel):
    """ESC subsystem status"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "running",
                "processed_sequences": 1547,
                "avg_processing_time_ms": 12.4,
                "vocabulary_size": 50000,
                "cache_hit_rate": 0.78
            }
        }
    )
    
    status: str = Field(..., description="ESC subsystem status")
    processed_sequences: int = Field(..., ge=0, description="Total processed sequences")
    avg_processing_time_ms: float = Field(..., ge=0.0, description="Average processing time")
    vocabulary_size: int = Field(..., ge=0, description="Current vocabulary size")
    cache_hit_rate: float = Field(..., ge=0.0, le=1.0, description="Embedding cache hit rate")
    constitutional_violations: int = Field(
        ..., ge=0,
        description="Total constitutional violations detected"
    )
    active_attention_layers: int = Field(..., ge=0, description="Active attention layers")
    memory_usage_mb: Optional[float] = Field(
        default=None, ge=0.0,
        description="Memory usage in megabytes"
    )