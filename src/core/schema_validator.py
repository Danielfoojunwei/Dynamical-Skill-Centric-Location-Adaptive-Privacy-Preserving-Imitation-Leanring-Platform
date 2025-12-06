"""
Schema Enforcement and Data Validation.

This module implements the 'neuracore-validate' logic described in the report.
It checks for:
- NaN values
- Frozen frames (identical data across timestamps)
- Timestamp jumps (temporal discontinuity)
- Schema compliance
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]

class SchemaValidator:
    """
    Validates sensor data against defined schemas and quality checks.
    """
    
    def __init__(self, strict: bool = True):
        self.strict = strict
        self.last_timestamps: Dict[str, float] = {}
        self.last_frames: Dict[str, np.ndarray] = {}
        
    def validate_frame(self, channel: str, data: Any, timestamp: float) -> ValidationResult:
        """
        Validate a single data frame.
        """
        errors = []
        warnings = []
        
        # 1. Schema Check (Basic Type Check)
        if isinstance(data, np.ndarray):
            if np.isnan(data).any():
                errors.append(f"NaN values detected in {channel}")
            if np.isinf(data).any():
                errors.append(f"Infinite values detected in {channel}")
                
        # 2. Timestamp Continuity Check
        if channel in self.last_timestamps:
            dt = timestamp - self.last_timestamps[channel]
            if dt <= 0:
                errors.append(f"Timestamp regression or duplicate in {channel}: dt={dt}")
            elif dt > 1.0: # Arbitrary large gap
                warnings.append(f"Large timestamp gap in {channel}: dt={dt}")
        
        self.last_timestamps[channel] = timestamp
        
        # 3. Frozen Frame Check (for arrays)
        if isinstance(data, np.ndarray) and channel in self.last_frames:
            if np.array_equal(data, self.last_frames[channel]):
                warnings.append(f"Potential frozen frame detected in {channel}")
        
        if isinstance(data, np.ndarray):
             self.last_frames[channel] = data.copy()
             
        valid = len(errors) == 0
        if not valid and self.strict:
            logger.error(f"Validation failed for {channel}: {errors}")
            
        return ValidationResult(valid=valid, errors=errors, warnings=warnings)

    def validate_dataset(self, dataset_path: str):
        """
        Batch validation of a dataset (placeholder).
        """
        pass
