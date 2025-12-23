"""Auto-tuner for finding the optimal batch size for model training."""

import gc
import logging
import time
from typing import Any, Dict, Optional

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Mock classes for type hinting if torch is missing
    class Dataset: pass
    class DataLoader: pass

# Assuming NeuracoreModel is not directly available, we'll use a generic protocol or base class if needed.
# For now, we'll assume the model passed has the required methods (train, configure_optimizers, training_step).
# If Dynamical Edge has a specific base class, we should use that.
# Based on file exploration, there isn't a single clear base class like NeuracoreModel in the root.
# We will adapt the type hinting to be more generic or rely on duck typing for now, 
# but we need to import MemoryMonitor.

try:
    from .memory_monitor import MemoryMonitor, OutOfMemoryError
except ImportError:
    from src.core.memory_monitor import MemoryMonitor, OutOfMemoryError

logger = logging.getLogger(__name__)


class BatchSizeAutotuner:
    """Auto-tuner for finding the optimal batch size for model training."""

    def __init__(
        self,
        dataset: Dataset,
        model: Any, # generic model
        model_kwargs: Dict[str, Any],
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        min_batch_size: int = 2,
        max_batch_size: int = 512,
        num_iterations: int = 3,
    ):
        """Initialize the batch size auto-tuner.

        Args:
            dataset: Dataset to use for testing
            model: Model to use for testing
            model_kwargs: Arguments to pass to model constructor
            dataloader_kwargs: Additional arguments for the DataLoader
            min_batch_size: Minimum batch size to try
            max_batch_size: Maximum batch size to try
            num_iterations: Number of iterations to run for each batch size
        """
        self.dataset = dataset
        self.model_kwargs = model_kwargs
        self.dataloader_kwargs = dataloader_kwargs or {}
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.num_iterations = num_iterations
        
        if not HAS_TORCH:
            logger.warning("Torch not available, autotuner disabled")
            return

        # Determine device from model if possible, else default to cuda
        if hasattr(model, 'device'):
            self.device = model.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not torch.cuda.is_available() or "cuda" not in str(self.device):
             logger.warning("Autotuning batch size is intended for GPUs, but CUDA is not available or device is CPU.")
             
        self.model = model

        # create optimizers
        # We assume the model has a configure_optimizers method, or we might need to adapt this.
        if hasattr(self.model, 'configure_optimizers'):
            self.optimizers = self.model.configure_optimizers()
        else:
             # Fallback: create a simple optimizer if none provided, just for the stress test
             self.optimizers = [torch.optim.Adam(self.model.parameters(), lr=1e-3)]

        # Validate batch size ranges
        if min_batch_size > max_batch_size:
            raise ValueError(
                f"min_batch_size ({min_batch_size}) must be "
                f"<= max_batch_size ({max_batch_size})"
            )

        # Validate dataset size
        if len(dataset) < min_batch_size:
            raise ValueError(
                f"Dataset size ({len(dataset)}) is smaller "
                f"than min_batch_size ({min_batch_size})"
            )

    def find_optimal_batch_size(self) -> int:
        """Find the optimal batch size using binary search.

        Returns:
            The optimal batch size
        """
        logger.info(
            "Finding optimal batch size between "
            f"{self.min_batch_size} and {self.max_batch_size}"
        )

        # Binary search approach
        low = self.min_batch_size
        high = self.max_batch_size
        optimal_batch_size = low  # Start conservative

        while low <= high:
            mid = (low + high) // 2
            success = self._test_batch_size(mid)

            if success:
                # This batch size works, try a larger one
                optimal_batch_size = mid
                low = mid + 1
            else:
                # This batch size failed, try a smaller one
                high = mid - 1

            # Clean up memory
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Reduce by 15% to be safe
        reduced_batch_size = int(optimal_batch_size * 0.85)
        logger.info(
            f"Optimal batch size found {optimal_batch_size}, "
            f"Reducing it by 15% to {reduced_batch_size}"
        )
        return max(reduced_batch_size, self.min_batch_size)

    def _test_batch_size(self, batch_size: int) -> bool:
        """Test if a specific batch size works.

        Args:
            batch_size: Batch size to test

        Returns:
            True if the batch size works, False if it causes OOM error
        """
        logger.info(f"Testing batch size: {batch_size}")

        try:
            memory_monitor = MemoryMonitor(
                max_ram_utilization=0.9, max_gpu_utilization=0.95
            )

            # Create dataloader
            dataloader_kwargs = {**self.dataloader_kwargs, "batch_size": batch_size}
            # Ensure we don't use more workers than reasonable for a test
            if 'num_workers' in dataloader_kwargs:
                dataloader_kwargs['num_workers'] = min(dataloader_kwargs['num_workers'], 4)
                
            data_loader = DataLoader(self.dataset, **dataloader_kwargs)

            # Get a batch that we can reuse
            try:
                batch = next(iter(data_loader))
            except StopIteration:
                logger.warning("Dataset empty or failed to load batch.")
                return False

            for i in range(self.num_iterations):

                memory_monitor.check_memory()

                # Handle different batch types (dict, list, tensor)
                # This part needs to be adapted to what Dynamical Edge models expect.
                # We'll assume the model handles the move to device or we do it here if it's a tensor/dict of tensors.
                
                # Simple recursive to_device
                def to_device(item, device):
                    if isinstance(item, torch.Tensor):
                        return item.to(device)
                    elif isinstance(item, dict):
                        return {k: to_device(v, device) for k, v in item.items()}
                    elif isinstance(item, list):
                        return [to_device(v, device) for v in item]
                    return item

                batch = to_device(batch, self.device)

                # Forward pass
                self.model.train()

                for optimizer in self.optimizers:
                    optimizer.zero_grad()

                start_time = time.time()
                
                # Adapt to model interface. 
                # If model has training_step, use it. Else call model(batch) and assume it returns loss or output.
                if hasattr(self.model, 'training_step'):
                    outputs = self.model.training_step(batch)
                    # Assume outputs has a 'loss' or is a dict with 'loss' or is the loss itself
                    if isinstance(outputs, dict) and 'loss' in outputs:
                        loss = outputs['loss']
                    elif hasattr(outputs, 'losses') and isinstance(outputs.losses, dict):
                         loss = sum(outputs.losses.values()).mean()
                    else:
                        loss = outputs # assume it's the loss
                else:
                    # Fallback for standard pytorch models
                    outputs = self.model(batch)
                    # We need a loss function if the model doesn't return it. 
                    # For autotuning, we just need to run the graph. 
                    # If outputs is a tensor, sum it to get a scalar to backward.
                    if isinstance(outputs, torch.Tensor):
                        loss = outputs.sum()
                    else:
                        # If tuple/list, sum all tensors
                        loss = sum([o.sum() for o in outputs if isinstance(o, torch.Tensor)])

                # Backward pass
                loss.backward()
                for optimizer in self.optimizers:
                    optimizer.step()

                end_time = time.time()
                logger.info(
                    f"  Iteration {i+1}/{self.num_iterations} - "
                    f"Time: {end_time - start_time:.4f}s"
                )

            logger.info(f"Batch size {batch_size} succeeded ✓")
            return True

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.info(f"Batch size {batch_size} failed due to OOM error ✗")
                # Clean up memory after OOM
                if HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False
            else:
                # Re-raise if it's not an OOM error
                # logger.error(f"Runtime error: {e}")
                # return False # Be robust
                raise

        except OutOfMemoryError:
            logger.info(f"Batch size {batch_size} failed due to RAM OOM error ✗")
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error during batch tuning: {e}")
            return False

        finally:
            # Clean up
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


def find_optimal_batch_size(
    dataset: Dataset,
    model: Any,
    model_kwargs: Dict[str, Any],
    dataloader_kwargs: Optional[Dict[str, Any]] = None,
    min_batch_size: int = 2,
    max_batch_size: int = 512,
) -> int:
    """Find the optimal batch size for a given model and dataset.

    Args:
        dataset: Dataset to use for testing
        model: Model to use for testing
        model_kwargs: Arguments to pass to model constructor
        dataloader_kwargs: Additional arguments for the DataLoader
        min_batch_size: Minimum batch size to try
        max_batch_size: Maximum batch size to try

    Returns:
        The optimal batch size
    """
    autotuner = BatchSizeAutotuner(
        dataset=dataset,
        model=model,
        model_kwargs=model_kwargs,
        dataloader_kwargs=dataloader_kwargs,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
    )

    return autotuner.find_optimal_batch_size()
