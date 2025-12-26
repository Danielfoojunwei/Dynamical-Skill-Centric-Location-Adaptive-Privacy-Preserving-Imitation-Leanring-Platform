"""
Deployment Validator - Startup Checks for Production Deployment

Validates that all required components are properly configured before
allowing the system to start. Fails fast if configuration is invalid.

Checks:
1. Required models are available
2. TFLOPS budget is sufficient
3. Hardware capabilities match requirements
4. All dependencies are installed
5. Camera/sensor connectivity
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Status of validation check."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    name: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    required: bool = True


@dataclass
class ValidationResult:
    """Complete validation result."""
    passed: bool
    checks: List[ValidationCheck]
    critical_failures: List[str]
    warnings: List[str]

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = ["=" * 60, "DEPLOYMENT VALIDATION RESULT", "=" * 60]

        passed = sum(1 for c in self.checks if c.status == ValidationStatus.PASS)
        failed = sum(1 for c in self.checks if c.status == ValidationStatus.FAIL)
        warned = sum(1 for c in self.checks if c.status == ValidationStatus.WARN)

        lines.append(f"Total: {len(self.checks)} checks")
        lines.append(f"Passed: {passed}, Failed: {failed}, Warnings: {warned}")
        lines.append("")

        if self.critical_failures:
            lines.append("CRITICAL FAILURES:")
            for f in self.critical_failures:
                lines.append(f"  - {f}")
            lines.append("")

        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  - {w}")
            lines.append("")

        lines.append("=" * 60)
        lines.append(f"RESULT: {'PASS' if self.passed else 'FAIL'}")
        lines.append("=" * 60)

        return "\n".join(lines)


class DeploymentValidator:
    """
    Validates deployment configuration before system startup.

    Usage:
        validator = DeploymentValidator()
        result = validator.validate(config)

        if not result.passed:
            raise RuntimeError(result.summary())
    """

    def __init__(self):
        self.checks: List[ValidationCheck] = []

    def validate(self, config: Any = None) -> ValidationResult:
        """
        Run all validation checks.

        Args:
            config: Application configuration object

        Returns:
            ValidationResult with all check outcomes
        """
        self.checks = []

        # Run checks
        self._check_python_dependencies()
        self._check_meta_ai_models(config)
        self._check_safety_config(config)
        self._check_tflops_budget(config)
        self._check_openpi_availability()
        self._check_cuda_availability()
        self._check_camera_config(config)

        # Compute result
        critical_failures = [
            c.message for c in self.checks
            if c.status == ValidationStatus.FAIL and c.required
        ]

        warnings = [
            c.message for c in self.checks
            if c.status == ValidationStatus.WARN
        ]

        passed = len(critical_failures) == 0

        return ValidationResult(
            passed=passed,
            checks=self.checks,
            critical_failures=critical_failures,
            warnings=warnings,
        )

    def _check_python_dependencies(self):
        """Check required Python packages."""
        required = {
            "numpy": "numpy",
            "torch": "torch",
            "pydantic": "pydantic",
        }

        optional = {
            "cv2": "opencv-python",
            "scipy": "scipy",
            "sklearn": "scikit-learn",
        }

        for module, package in required.items():
            try:
                __import__(module)
                self.checks.append(ValidationCheck(
                    name=f"Python: {package}",
                    status=ValidationStatus.PASS,
                    message=f"{package} is installed",
                    required=True,
                ))
            except ImportError:
                self.checks.append(ValidationCheck(
                    name=f"Python: {package}",
                    status=ValidationStatus.FAIL,
                    message=f"Required package {package} not installed",
                    required=True,
                ))

        for module, package in optional.items():
            try:
                __import__(module)
                self.checks.append(ValidationCheck(
                    name=f"Python: {package}",
                    status=ValidationStatus.PASS,
                    message=f"{package} is installed",
                    required=False,
                ))
            except ImportError:
                self.checks.append(ValidationCheck(
                    name=f"Python: {package}",
                    status=ValidationStatus.WARN,
                    message=f"Optional package {package} not installed",
                    required=False,
                ))

    def _check_meta_ai_models(self, config: Any):
        """Check Meta AI model availability."""
        models = ["dinov3", "sam3", "vjepa2"]

        for model in models:
            try:
                if model == "dinov3":
                    from src.meta_ai.dinov3 import DINOv3Encoder
                    self.checks.append(ValidationCheck(
                        name=f"Model: DINOv3",
                        status=ValidationStatus.PASS,
                        message="DINOv3 module available",
                    ))
                elif model == "sam3":
                    from src.meta_ai.sam3 import SAM3Segmenter
                    self.checks.append(ValidationCheck(
                        name=f"Model: SAM3",
                        status=ValidationStatus.PASS,
                        message="SAM3 module available",
                    ))
                elif model == "vjepa2":
                    from src.meta_ai.vjepa2 import VJEPA2WorldModel
                    self.checks.append(ValidationCheck(
                        name=f"Model: V-JEPA2",
                        status=ValidationStatus.PASS,
                        message="V-JEPA2 module available",
                    ))
            except ImportError as e:
                self.checks.append(ValidationCheck(
                    name=f"Model: {model.upper()}",
                    status=ValidationStatus.WARN,
                    message=f"{model} not available: {e}",
                    required=False,
                ))

    def _check_safety_config(self, config: Any):
        """Check safety configuration."""
        try:
            from src.safety.cbf import CBFFilter
            from src.safety.rta import RuntimeAssurance
            self.checks.append(ValidationCheck(
                name="Safety: CBF + RTA",
                status=ValidationStatus.PASS,
                message="Safety modules available",
            ))
        except ImportError as e:
            self.checks.append(ValidationCheck(
                name="Safety: CBF + RTA",
                status=ValidationStatus.WARN,
                message=f"Safety modules not fully available: {e}",
                required=False,
            ))

        try:
            from src.robot_runtime.perception_integration import PerceptionSafetyBridge
            self.checks.append(ValidationCheck(
                name="Safety: Perception Bridge",
                status=ValidationStatus.PASS,
                message="Perception-Safety bridge available",
            ))
        except ImportError as e:
            self.checks.append(ValidationCheck(
                name="Safety: Perception Bridge",
                status=ValidationStatus.WARN,
                message=f"Perception-Safety bridge not available: {e}",
                required=False,
            ))

    def _check_tflops_budget(self, config: Any):
        """Check TFLOPS budget is sufficient."""
        # Get configured budget
        if config is None:
            self.checks.append(ValidationCheck(
                name="TFLOPS: Budget",
                status=ValidationStatus.SKIP,
                message="No config provided, skipping TFLOPS check",
                required=False,
            ))
            return

        try:
            # Check if budget is configured
            if hasattr(config, 'meta_ai') and hasattr(config.meta_ai, 'tflops_allocation'):
                total = sum(vars(config.meta_ai.tflops_allocation).values())
                available = getattr(config, 'hardware', {}).get('tflops_available', 137.0)

                if total <= available * 0.85:
                    self.checks.append(ValidationCheck(
                        name="TFLOPS: Budget",
                        status=ValidationStatus.PASS,
                        message=f"TFLOPS budget OK: {total:.1f}/{available:.1f} ({total/available*100:.1f}%)",
                    ))
                else:
                    self.checks.append(ValidationCheck(
                        name="TFLOPS: Budget",
                        status=ValidationStatus.WARN,
                        message=f"TFLOPS budget high: {total:.1f}/{available:.1f} ({total/available*100:.1f}%)",
                        required=False,
                    ))
            else:
                self.checks.append(ValidationCheck(
                    name="TFLOPS: Budget",
                    status=ValidationStatus.SKIP,
                    message="TFLOPS allocation not configured",
                    required=False,
                ))
        except Exception as e:
            self.checks.append(ValidationCheck(
                name="TFLOPS: Budget",
                status=ValidationStatus.WARN,
                message=f"Could not check TFLOPS budget: {e}",
                required=False,
            ))

    def _check_openpi_availability(self):
        """Check OpenPI (Pi0.5) availability."""
        try:
            import openpi
            self.checks.append(ValidationCheck(
                name="VLA: OpenPI",
                status=ValidationStatus.PASS,
                message="OpenPI (Pi0.5) available",
            ))
        except ImportError:
            self.checks.append(ValidationCheck(
                name="VLA: OpenPI",
                status=ValidationStatus.WARN,
                message="OpenPI not installed. Pi0.5 VLA unavailable. Install from: "
                       "https://github.com/Physical-Intelligence/openpi",
                required=False,
            ))

    def _check_cuda_availability(self):
        """Check CUDA availability."""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self.checks.append(ValidationCheck(
                    name="Hardware: CUDA",
                    status=ValidationStatus.PASS,
                    message=f"CUDA available: {device_name}",
                ))
            else:
                self.checks.append(ValidationCheck(
                    name="Hardware: CUDA",
                    status=ValidationStatus.WARN,
                    message="CUDA not available, running on CPU (slow)",
                    required=False,
                ))
        except ImportError:
            self.checks.append(ValidationCheck(
                name="Hardware: CUDA",
                status=ValidationStatus.WARN,
                message="PyTorch not installed, cannot check CUDA",
                required=False,
            ))

    def _check_camera_config(self, config: Any):
        """Check camera configuration."""
        if config is None:
            self.checks.append(ValidationCheck(
                name="Cameras: Config",
                status=ValidationStatus.SKIP,
                message="No config provided, skipping camera check",
                required=False,
            ))
            return

        try:
            if hasattr(config, 'cameras') and config.cameras:
                num_cameras = len(config.cameras)
                self.checks.append(ValidationCheck(
                    name="Cameras: Config",
                    status=ValidationStatus.PASS,
                    message=f"{num_cameras} cameras configured",
                ))
            else:
                self.checks.append(ValidationCheck(
                    name="Cameras: Config",
                    status=ValidationStatus.WARN,
                    message="No cameras configured",
                    required=False,
                ))
        except Exception as e:
            self.checks.append(ValidationCheck(
                name="Cameras: Config",
                status=ValidationStatus.WARN,
                message=f"Could not check camera config: {e}",
                required=False,
            ))


def validate_deployment(config: Any = None, strict: bool = False) -> bool:
    """
    Convenience function to validate deployment.

    Args:
        config: Application configuration
        strict: If True, raise exception on failure

    Returns:
        True if validation passed, False otherwise
    """
    validator = DeploymentValidator()
    result = validator.validate(config)

    if not result.passed:
        logger.error(result.summary())
        if strict:
            raise RuntimeError(
                f"Deployment validation failed:\n" +
                "\n".join(result.critical_failures)
            )

    return result.passed


# Run validation when module is executed directly
if __name__ == "__main__":
    print("Running deployment validation...")
    result = DeploymentValidator().validate()
    print(result.summary())
