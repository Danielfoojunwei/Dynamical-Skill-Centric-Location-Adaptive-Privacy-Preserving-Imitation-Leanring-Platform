"""
Watchdog - Hardware Safety Watchdog

Tier 1 component running at 1kHz.
Triggers emergency stop if not pet regularly.
"""

import time
import logging
import threading
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class Watchdog:
    """
    Hardware watchdog for safety.

    Must be pet regularly or triggers emergency stop.
    Typically connected to hardware watchdog timer.
    """

    def __init__(self, timeout_ms: int = 5):
        self.timeout_ms = timeout_ms
        self.timeout_sec = timeout_ms / 1000.0

        self._last_pet_time = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._estop_callback: Optional[Callable] = None

        # Statistics
        self._pet_count = 0
        self._missed_count = 0

    def start(self, estop_callback: Optional[Callable] = None) -> None:
        """
        Start watchdog timer.

        Args:
            estop_callback: Function to call on timeout (triggers E-stop)
        """
        self._estop_callback = estop_callback
        self._running = True
        self._last_pet_time = time.time()

        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="watchdog",
            daemon=True
        )
        self._thread.start()

        logger.info(f"Watchdog started with {self.timeout_ms}ms timeout")

    def stop(self) -> None:
        """Stop watchdog timer."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("Watchdog stopped")

    def pet(self) -> None:
        """
        Pet the watchdog to prevent timeout.

        Must be called at least every timeout_ms milliseconds.
        """
        self._last_pet_time = time.time()
        self._pet_count += 1

    def get_time_remaining_ms(self) -> float:
        """Get time remaining until timeout in milliseconds."""
        elapsed = time.time() - self._last_pet_time
        remaining = self.timeout_sec - elapsed
        return max(0, remaining * 1000)

    def get_stats(self) -> dict:
        """Get watchdog statistics."""
        return {
            'timeout_ms': self.timeout_ms,
            'pet_count': self._pet_count,
            'missed_count': self._missed_count,
            'running': self._running,
            'time_remaining_ms': self.get_time_remaining_ms(),
        }

    def _monitor_loop(self) -> None:
        """Monitor thread that checks for timeouts."""
        check_interval = self.timeout_sec / 2  # Check at 2x timeout rate

        while self._running:
            time.sleep(check_interval)

            elapsed = time.time() - self._last_pet_time

            if elapsed > self.timeout_sec:
                self._missed_count += 1
                logger.critical(f"WATCHDOG TIMEOUT - {elapsed*1000:.1f}ms since last pet")

                if self._estop_callback:
                    try:
                        self._estop_callback()
                    except Exception as e:
                        logger.error(f"E-stop callback failed: {e}")

                # Reset timer to avoid repeated triggers
                self._last_pet_time = time.time()


class HardwareWatchdog(Watchdog):
    """
    Hardware watchdog integration.

    Connects to actual hardware watchdog timer (e.g., on Jetson).
    """

    def __init__(self, timeout_ms: int = 5, device: str = "/dev/watchdog"):
        super().__init__(timeout_ms)
        self.device = device
        self._hw_file = None

    def start(self, estop_callback: Optional[Callable] = None) -> None:
        """Start hardware watchdog."""
        try:
            # Open hardware watchdog device
            self._hw_file = open(self.device, 'w')
            logger.info(f"Hardware watchdog opened: {self.device}")
        except (IOError, PermissionError) as e:
            logger.warning(f"Could not open hardware watchdog: {e}")
            logger.warning("Falling back to software watchdog")

        super().start(estop_callback)

    def stop(self) -> None:
        """Stop hardware watchdog."""
        if self._hw_file:
            try:
                # Write magic close character to disable watchdog
                self._hw_file.write('V')
                self._hw_file.close()
            except Exception as e:
                logger.error(f"Error closing hardware watchdog: {e}")

        super().stop()

    def pet(self) -> None:
        """Pet hardware watchdog."""
        super().pet()

        if self._hw_file:
            try:
                self._hw_file.write('1')
                self._hw_file.flush()
            except Exception as e:
                logger.error(f"Error petting hardware watchdog: {e}")
