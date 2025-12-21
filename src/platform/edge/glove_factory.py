"""
Unified Glove Factory for Dynamical.ai

Provides a unified interface for creating glove drivers that abstracts
the choice between DYGlove and MANUS hardware.

Supported Glove Types:
- DYGlove: Dynamical.ai's 21-DOF haptic feedback glove
- MANUS: MANUS VR/Motion Capture gloves (Quantum, Prime, Metaglove)
- Simulator: Virtual glove for testing

Example:
    ```python
    from src.platform.edge.glove_factory import create_glove, GloveType

    # Auto-detect available gloves
    glove = create_glove()

    # Or specify a type
    glove = create_glove(glove_type=GloveType.MANUS, side=Hand.RIGHT)

    # Connect and use
    glove.connect()
    state = glove.get_state()
    print(f"21-DOF: {state.to_21dof_array()}")
    glove.disconnect()
    ```
"""

import os
from enum import Enum
from typing import Optional, List, Dict, Any, Union

from src.platform.logging_utils import get_logger
from src.platform.edge.dyglove_sdk import (
    Hand,
    GloveState,
    GloveInfo,
    CalibrationData,
    DYGloveDriverBase,
    DYGloveSDKClient,
    DYGloveSimulator,
    DYGloveWiFiDriver,
    DYGloveDiscovery,
    DYGloveAsyncReader,
    DYGloveQualityConfig,
    create_dyglove_driver,
    create_dyglove_reader,
)
from src.platform.edge.manus_sdk import (
    ManusGloveDriver,
    ManusGloveSimulator,
    ManusGloveDiscovery,
    create_manus_driver,
    create_manus_reader,
)

logger = get_logger(__name__)

__version__ = "1.0.0"


class GloveType(str, Enum):
    """Supported glove types."""
    DYGLOVE = "dyglove"
    MANUS = "manus"
    SIMULATOR = "simulator"
    AUTO = "auto"


class GloveConfig:
    """
    Configuration for glove initialization.

    Attributes:
        glove_type: Type of glove to use
        side: Left or right hand
        use_simulator: Force simulation mode
        wifi_ip: WiFi IP for DYGlove wireless connection
        serial_port: Serial port for DYGlove wired connection
        target_hz: Target update rate
        enable_haptics: Enable haptic feedback
        quality_filter: Enable quality filtering
    """

    def __init__(
        self,
        glove_type: GloveType = GloveType.AUTO,
        side: Hand = Hand.RIGHT,
        use_simulator: bool = False,
        wifi_ip: Optional[str] = None,
        serial_port: Optional[str] = None,
        target_hz: float = 120.0,
        enable_haptics: bool = True,
        quality_filter: bool = True,
    ):
        self.glove_type = glove_type
        self.side = side
        self.use_simulator = use_simulator
        self.wifi_ip = wifi_ip
        self.serial_port = serial_port
        self.target_hz = target_hz
        self.enable_haptics = enable_haptics
        self.quality_filter = quality_filter

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'GloveConfig':
        """Create config from dictionary."""
        glove_type = config.get('glove_type', 'auto')
        if isinstance(glove_type, str):
            glove_type = GloveType(glove_type.lower())

        side = config.get('side', 'right')
        if isinstance(side, str):
            side = Hand(side.lower())

        return cls(
            glove_type=glove_type,
            side=side,
            use_simulator=config.get('use_simulator', False),
            wifi_ip=config.get('wifi_ip'),
            serial_port=config.get('serial_port'),
            target_hz=config.get('target_hz', 120.0),
            enable_haptics=config.get('enable_haptics', True),
            quality_filter=config.get('quality_filter', True),
        )

    @classmethod
    def from_env(cls) -> 'GloveConfig':
        """Create config from environment variables."""
        glove_type_str = os.environ.get('GLOVE_TYPE', 'auto').lower()
        glove_type = GloveType(glove_type_str) if glove_type_str in [e.value for e in GloveType] else GloveType.AUTO

        side_str = os.environ.get('GLOVE_SIDE', 'right').lower()
        side = Hand(side_str) if side_str in ['left', 'right'] else Hand.RIGHT

        return cls(
            glove_type=glove_type,
            side=side,
            use_simulator=os.environ.get('GLOVE_SIMULATOR', '').lower() == 'true',
            wifi_ip=os.environ.get('GLOVE_WIFI_IP'),
            serial_port=os.environ.get('GLOVE_SERIAL_PORT'),
            target_hz=float(os.environ.get('GLOVE_TARGET_HZ', '120')),
            enable_haptics=os.environ.get('GLOVE_HAPTICS', 'true').lower() == 'true',
            quality_filter=os.environ.get('GLOVE_QUALITY_FILTER', 'true').lower() == 'true',
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'glove_type': self.glove_type.value,
            'side': self.side.value,
            'use_simulator': self.use_simulator,
            'wifi_ip': self.wifi_ip,
            'serial_port': self.serial_port,
            'target_hz': self.target_hz,
            'enable_haptics': self.enable_haptics,
            'quality_filter': self.quality_filter,
        }


class UnifiedGloveDiscovery:
    """
    Unified discovery for all supported glove types.

    Scans for both DYGlove and MANUS devices and returns a combined list.
    """

    def __init__(self):
        self._dyglove_discovery = DYGloveDiscovery()
        self._manus_discovery = ManusGloveDiscovery()

    def scan(self, timeout: float = 5.0) -> List[Dict[str, Any]]:
        """
        Scan for all available gloves.

        Args:
            timeout: Scan timeout in seconds

        Returns:
            List of discovered devices with their properties
        """
        devices = []

        # Scan for DYGlove devices
        try:
            dyglove_devices = self._dyglove_discovery.scan(timeout=timeout / 2)
            for dev in dyglove_devices:
                dev['glove_type'] = GloveType.DYGLOVE.value
                devices.append(dev)
        except Exception as e:
            logger.debug(f"DYGlove scan error: {e}")

        # Scan for MANUS devices
        try:
            manus_devices = self._manus_discovery.scan(timeout=timeout / 2)
            for dev in manus_devices:
                dev['glove_type'] = GloveType.MANUS.value
                devices.append(dev)
        except Exception as e:
            logger.debug(f"MANUS scan error: {e}")

        return devices

    def scan_by_type(self, glove_type: GloveType, timeout: float = 5.0) -> List[Dict[str, Any]]:
        """
        Scan for a specific glove type.

        Args:
            glove_type: Type of glove to scan for
            timeout: Scan timeout in seconds

        Returns:
            List of discovered devices
        """
        if glove_type == GloveType.DYGLOVE:
            devices = self._dyglove_discovery.scan(timeout=timeout)
            for dev in devices:
                dev['glove_type'] = GloveType.DYGLOVE.value
            return devices
        elif glove_type == GloveType.MANUS:
            devices = self._manus_discovery.scan(timeout=timeout)
            for dev in devices:
                dev['glove_type'] = GloveType.MANUS.value
            return devices
        else:
            return self.scan(timeout=timeout)


def detect_glove_type(timeout: float = 3.0) -> GloveType:
    """
    Auto-detect available glove type.

    Scans for connected gloves and returns the first available type.

    Args:
        timeout: Scan timeout in seconds

    Returns:
        Detected GloveType or SIMULATOR if none found
    """
    discovery = UnifiedGloveDiscovery()
    devices = discovery.scan(timeout=timeout)

    for dev in devices:
        if dev.get('verified', False):
            glove_type = dev.get('glove_type', '')
            if glove_type == GloveType.DYGLOVE.value:
                logger.info("Auto-detected DYGlove")
                return GloveType.DYGLOVE
            elif glove_type == GloveType.MANUS.value:
                logger.info("Auto-detected MANUS glove")
                return GloveType.MANUS

    logger.info("No glove detected - using simulator")
    return GloveType.SIMULATOR


def create_glove(
    glove_type: GloveType = GloveType.AUTO,
    side: Hand = Hand.RIGHT,
    use_simulator: bool = False,
    config: Optional[GloveConfig] = None,
    **kwargs
) -> DYGloveDriverBase:
    """
    Create a glove driver of the specified type.

    This is the main factory function for creating glove drivers.
    It abstracts the choice between DYGlove and MANUS hardware.

    Args:
        glove_type: Type of glove (AUTO, DYGLOVE, MANUS, SIMULATOR)
        side: Left or right hand
        use_simulator: Force simulation mode
        config: Optional GloveConfig for detailed configuration
        **kwargs: Additional arguments passed to the driver

    Returns:
        DYGloveDriverBase-compatible driver

    Example:
        >>> glove = create_glove(glove_type=GloveType.MANUS, side=Hand.RIGHT)
        >>> glove.connect()
        >>> state = glove.get_state()
        >>> print(state.to_21dof_array())
    """
    # Use config if provided
    if config is not None:
        glove_type = config.glove_type
        side = config.side
        use_simulator = config.use_simulator
        kwargs.update({
            'wifi_ip': config.wifi_ip,
            'port': config.serial_port,
        })

    # Force simulator if requested
    if use_simulator:
        glove_type = GloveType.SIMULATOR

    # Auto-detect if needed
    if glove_type == GloveType.AUTO:
        glove_type = detect_glove_type()

    # Create the appropriate driver
    if glove_type == GloveType.DYGLOVE:
        wifi_ip = kwargs.get('wifi_ip')
        port = kwargs.get('port')

        if wifi_ip:
            driver = DYGloveWiFiDriver(ip_address=wifi_ip, side=side)
        else:
            driver = DYGloveSDKClient()

        logger.info(f"Created DYGlove driver ({side.value})")
        return driver

    elif glove_type == GloveType.MANUS:
        driver = ManusGloveDriver(side=side)
        logger.info(f"Created MANUS driver ({side.value})")
        return driver

    else:  # SIMULATOR or fallback
        # Choose simulator based on available hardware preference
        # Default to DYGlove simulator for consistency
        driver = DYGloveSimulator(side=side)
        logger.info(f"Created glove simulator ({side.value})")
        return driver


def create_glove_reader(
    glove_type: GloveType = GloveType.AUTO,
    side: Hand = Hand.RIGHT,
    use_simulator: bool = False,
    quality_config: Optional[DYGloveQualityConfig] = None,
    target_hz: float = 120.0,
    config: Optional[GloveConfig] = None,
    **kwargs
) -> DYGloveAsyncReader:
    """
    Create a glove async reader with quality filtering.

    Args:
        glove_type: Type of glove (AUTO, DYGLOVE, MANUS, SIMULATOR)
        side: Left or right hand
        use_simulator: Force simulation mode
        quality_config: Quality filter configuration
        target_hz: Target reading rate
        config: Optional GloveConfig for detailed configuration
        **kwargs: Additional arguments

    Returns:
        DYGloveAsyncReader configured for the specified glove

    Example:
        >>> reader = create_glove_reader(glove_type=GloveType.MANUS)
        >>> reader.driver.connect()
        >>> with reader:
        ...     time.sleep(1)
        ...     state = reader.get_latest_state()
        ...     print(state.to_21dof_array())
    """
    # Use config if provided
    if config is not None:
        glove_type = config.glove_type
        side = config.side
        use_simulator = config.use_simulator
        target_hz = config.target_hz
        if config.quality_filter and quality_config is None:
            quality_config = DYGloveQualityConfig(target_hz=target_hz)
        kwargs.update({
            'wifi_ip': config.wifi_ip,
            'port': config.serial_port,
        })

    # Create driver
    driver = create_glove(
        glove_type=glove_type,
        side=side,
        use_simulator=use_simulator,
        **kwargs
    )

    # Wrap in async reader
    return DYGloveAsyncReader(driver, quality_config, target_hz)


def create_glove_pair(
    glove_type: GloveType = GloveType.AUTO,
    use_simulator: bool = False,
    config: Optional[GloveConfig] = None,
    **kwargs
) -> Dict[str, DYGloveDriverBase]:
    """
    Create a pair of gloves (left and right).

    Useful for bimanual teleoperation scenarios.

    Args:
        glove_type: Type of glove
        use_simulator: Force simulation mode
        config: Optional base configuration
        **kwargs: Additional arguments

    Returns:
        Dictionary with 'left' and 'right' drivers

    Example:
        >>> gloves = create_glove_pair(glove_type=GloveType.MANUS)
        >>> gloves['left'].connect()
        >>> gloves['right'].connect()
    """
    left_driver = create_glove(
        glove_type=glove_type,
        side=Hand.LEFT,
        use_simulator=use_simulator,
        config=config,
        **kwargs
    )

    right_driver = create_glove(
        glove_type=glove_type,
        side=Hand.RIGHT,
        use_simulator=use_simulator,
        config=config,
        **kwargs
    )

    return {
        'left': left_driver,
        'right': right_driver,
    }


# =============================================================================
# CLI for testing
# =============================================================================

def main():
    """Command-line interface for unified glove factory."""
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Unified Glove Factory")
    parser.add_argument(
        "command",
        choices=["scan", "detect", "connect", "stream", "test"],
        help="Command to execute"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["auto", "dyglove", "manus", "simulator"],
        default="auto",
        help="Glove type"
    )
    parser.add_argument("--side", "-s", default="right", choices=["left", "right"])
    parser.add_argument("--rate", type=float, default=120.0)
    parser.add_argument("--duration", type=float, default=10.0)

    args = parser.parse_args()

    glove_type = GloveType(args.type)
    side = Hand(args.side)

    if args.command == "scan":
        print("Scanning for all glove types...")
        discovery = UnifiedGloveDiscovery()
        devices = discovery.scan()
        print(f"Found {len(devices)} device(s):")
        for dev in devices:
            print(f"  [{dev.get('glove_type', 'unknown')}] {dev}")

    elif args.command == "detect":
        print("Auto-detecting glove type...")
        detected = detect_glove_type()
        print(f"Detected: {detected.value}")

    elif args.command == "connect":
        print(f"Creating {glove_type.value} glove driver ({side.value})...")
        glove = create_glove(glove_type=glove_type, side=side)
        if glove.connect():
            print(f"Connected: {glove.info}")
            state = glove.get_state()
            print(f"21-DOF: {state.to_21dof_array()}")
            glove.disconnect()
        else:
            print("Connection failed")

    elif args.command == "stream":
        print(f"Streaming from {glove_type.value} glove ({side.value})...")
        reader = create_glove_reader(
            glove_type=glove_type,
            side=side,
            target_hz=args.rate
        )

        if reader.driver.connect():
            reader.start()
            start_time = time.time()

            try:
                while time.time() - start_time < args.duration:
                    state = reader.get_latest_state()
                    if state:
                        angles = state.to_21dof_array()
                        print(f"T={angles[0]:6.1f} I={angles[5]:6.1f} "
                              f"M={angles[9]:6.1f} R={angles[13]:6.1f} "
                              f"P={angles[17]:6.1f}", end='\r')
                    time.sleep(0.02)
            except KeyboardInterrupt:
                pass

            print("\nStopping...")
            reader.stop()
            reader.driver.disconnect()
        else:
            print("Connection failed")

    elif args.command == "test":
        print("Testing unified glove factory...")

        # Test all glove types in simulator mode
        for gt in [GloveType.DYGLOVE, GloveType.MANUS, GloveType.SIMULATOR]:
            print(f"\n  Testing {gt.value}...")

            glove = create_glove(glove_type=gt, side=side, use_simulator=True)
            assert glove.connect(), f"{gt.value} connection failed"

            state = glove.get_state()
            assert len(state.to_21dof_array()) == 21, f"{gt.value} 21-DOF failed"
            print(f"    Connected, 21-DOF: OK")

            glove.disconnect()

        # Test pair creation
        print("\n  Testing glove pair creation...")
        pair = create_glove_pair(use_simulator=True)
        assert 'left' in pair and 'right' in pair
        assert pair['left'].connect() and pair['right'].connect()
        print(f"    Left: {pair['left'].info.hand.value}")
        print(f"    Right: {pair['right'].info.hand.value}")
        pair['left'].disconnect()
        pair['right'].disconnect()

        # Test config from environment
        print("\n  Testing config from environment...")
        os.environ['GLOVE_TYPE'] = 'manus'
        os.environ['GLOVE_SIDE'] = 'left'
        config = GloveConfig.from_env()
        assert config.glove_type == GloveType.MANUS
        assert config.side == Hand.LEFT
        print(f"    Config: type={config.glove_type.value}, side={config.side.value}")

        # Cleanup
        del os.environ['GLOVE_TYPE']
        del os.environ['GLOVE_SIDE']

        print("\nAll tests passed!")


if __name__ == "__main__":
    main()
