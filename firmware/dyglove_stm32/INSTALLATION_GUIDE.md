# DYGlove Installation, Calibration & User Guide

**DYGlove: 21-DOF Wireless Haptic Force Feedback Glove**

Based on DOGlove (arXiv:2502.07730) with Dynamical.ai WiFi 6E modifications.

---

## Table of Contents

1. [Hardware Overview](#hardware-overview)
2. [Bill of Materials (BOM)](#bill-of-materials-bom)
3. [Assembly Instructions](#assembly-instructions)
4. [Firmware Installation](#firmware-installation)
5. [Calibration Procedure](#calibration-procedure)
6. [Software Setup](#software-setup)
7. [Usage Guide](#usage-guide)
8. [Troubleshooting](#troubleshooting)
9. [Design Considerations](#design-considerations)

---

## Hardware Overview

### Specifications

| Parameter | Value |
|-----------|-------|
| Degrees of Freedom | 21 DOF (16 encoders + 5 derived) |
| Force Feedback | 5 DOF (Dynamixel servos) |
| Haptic Feedback | 5 DOF (LRA vibrotactile @ 240Hz) |
| Motion Capture Rate | 120 Hz max |
| Haptic Update Rate | 30 Hz |
| Encoder Resolution | 0.01° (with calibration) |
| Raw Encoder Accuracy | ±7.2° (±2% linearity) |
| Calibrated Accuracy | ±1° |
| ADC Resolution | 24-bit |
| MCU | STM32F042K6T6 @ 48MHz |
| Connectivity | USB CDC / WiFi 6E (ESP32-C3) |
| Weight | ~550g |
| Cost | ~$600 USD |
| Assembly Time | 6-8 hours |

### Joint Configuration

```
Thumb (5 DOF):
  ├── TM_flex  (Trapeziometacarpal flexion)
  ├── TM_abd   (Trapeziometacarpal abduction)
  ├── MCP      (Metacarpophalangeal flexion)
  ├── IP       (Interphalangeal flexion)
  └── Wrist_PS (Wrist pronation/supination, from IMU)

Index/Middle/Ring/Pinky (4 DOF each):
  ├── MCP_flex (MCP flexion, encoder)
  ├── MCP_abd  (MCP abduction, encoder)
  ├── PIP      (PIP flexion, encoder)
  └── DIP      (DIP flexion, derived: 0.67 × PIP)
```

---

## Bill of Materials (BOM)

### Electronics

| Part | Description | Qty | Unit Cost | Source | Notes |
|------|-------------|-----|-----------|--------|-------|
| STM32F042K6T6 | MCU LQFP32 48MHz 32KB Flash | 1 | $3.50 | DigiKey | ARM Cortex-M0 |
| ADS1256 | 24-bit ADC module | 1 | $15.00 | AliExpress | 30kSPS, 8-channel |
| Alps RDC506018A | Rotary position sensor | 16 | $8.00 | DigiKey | 10kΩ, ±2% linearity |
| Dynamixel XL330-M288-T | Smart servo motor | 5 | $24.00 | Robotis | 0.52 N·m stall |
| DRV2605L | Haptic driver IC | 5 | $2.50 | DigiKey | I2C, TouchSense |
| LRA 8mm | Linear resonant actuator | 5 | $3.00 | AliExpress | 240Hz resonance |
| MPU6050 | 6-axis IMU module | 1 | $3.00 | AliExpress | I2C, for wrist |
| ESP32-C3-MINI | WiFi module (optional) | 1 | $4.00 | AliExpress | WiFi 6 capable |
| TCA9548A | I2C multiplexer | 1 | $2.00 | DigiKey | For 5× DRV2605L |
| 3.3V LDO | AMS1117-3.3 | 2 | $0.50 | AliExpress | Power regulation |
| USB-C connector | SMD 16-pin | 1 | $1.00 | AliExpress | |
| Li-Po battery | 1000mAh 3.7V | 1 | $8.00 | AliExpress | Optional wireless |

**Electronics Subtotal: ~$200**

### Mechanical

| Part | Description | Qty | Unit Cost | Source | Notes |
|------|-------------|-----|-----------|--------|-------|
| Stainless cable | 0.6mm braided wire | 5m | $5.00 | Amazon | For force transmission |
| Teflon tubing | 1mm ID cable guide | 2m | $3.00 | Amazon | Low friction |
| Tension springs | 5mm × 10mm | 20 | $0.50 | AliExpress | Return mechanism |
| M2 screws | Various lengths | 50 | $5.00 | Amazon | Assembly |
| M2 heat inserts | Brass M2 | 30 | $3.00 | Amazon | 3D print reinforcement |
| Bearings | 3×8×4mm MR83ZZ | 20 | $0.40 | AliExpress | Pulley bearings |
| Silicone tubing | 3mm ID finger caps | 0.5m | $2.00 | Amazon | Comfort |
| Velcro straps | 20mm width | 0.5m | $3.00 | Amazon | Securing glove |

**Mechanical Subtotal: ~$50**

### 3D Printed Parts

| Part | Material | Print Time | Notes |
|------|----------|------------|-------|
| Palm shell (L/R) | PETG/Nylon | 8 hrs | Main structure |
| Finger links ×20 | PETG | 4 hrs | Encoder mounts |
| Thumb assembly | PETG | 3 hrs | Complex geometry |
| Servo mounts ×5 | PETG | 2 hrs | Cable anchor |
| Cable guides ×10 | TPU | 1 hr | Flexible routing |
| PCB enclosure | PETG | 2 hrs | Electronics housing |
| Vive tracker mount | PETG | 1 hr | Optional |

**3D Printing Subtotal: ~$30** (filament cost)

### Tools Required

- Soldering station with fine tip
- Heat gun (for heat inserts)
- Crimping tool (JST connectors)
- Multimeter
- ST-LINK V2 programmer
- 3D printer (FDM, 0.4mm nozzle)
- Allen key set (metric)
- Wire strippers
- Tweezers

**Total BOM Cost: ~$600 USD**

---

## Assembly Instructions

### Step 1: PCB Assembly (2 hours)

1. **Solder STM32F042K6T6**
   - Use solder paste and hot air
   - Check orientation (pin 1 dot)
   - Verify continuity after soldering

2. **Solder power section**
   - 2× AMS1117-3.3 LDO
   - 10µF input/output capacitors
   - USB-C connector

3. **Solder communication interfaces**
   - SPI header for ADS1256
   - I2C header for IMU/haptics
   - UART header for servos
   - USB D+/D- to STM32

4. **Test power**
   - Connect USB, verify 3.3V on VCC
   - Check current draw (<50mA idle)

### Step 2: Encoder Assembly (2 hours)

1. **Mount encoders on finger links**
   - Press-fit RDC506018A into 3D printed mounts
   - Ensure shaft rotates freely
   - Apply thread locker to screws

2. **Wire encoders**
   - Connect VCC (3.3V), GND, and signal
   - Use shielded cable for signal wires
   - Route wires through cable channels

3. **Connect to ADS1256**
   - Encoders 0-7 → ADS1256 CH0-CH7
   - Encoders 8-15 → Second ADS1256 (or mux)

### Step 3: Force Feedback Assembly (1.5 hours)

1. **Mount Dynamixel servos**
   - Attach to palm shell servo mounts
   - Secure with M2 screws

2. **Cable-driven mechanism**
   - Thread 0.6mm steel cable through Teflon tubing
   - Attach cable ends to servo horn and finger tip
   - Ensure 1:1 transmission ratio
   - Add return spring for bidirectional force

3. **Connect Dynamixel bus**
   - Daisy-chain all 5 servos
   - Set unique IDs (1-5) using Dynamixel Wizard
   - Connect to STM32 UART (PB10/PB11)

### Step 4: Haptic Feedback Assembly (1 hour)

1. **Mount LRA actuators**
   - Glue 8mm LRA to fingertip caps
   - Ensure good skin contact

2. **Wire DRV2605L drivers**
   - Connect via I2C through TCA9548A mux
   - Each DRV2605L drives one LRA
   - Set I2C addresses via mux channel

3. **Test haptics**
   - Send waveform command
   - Verify vibration on each finger

### Step 5: IMU Installation (30 minutes)

1. **Mount MPU6050**
   - Attach to palm shell (back of hand)
   - Align axes with hand coordinate frame

2. **Wire IMU**
   - Connect to I2C bus (shared with haptics)
   - Address: 0x68 (AD0 to GND)

### Step 6: Final Assembly (1 hour)

1. **Assemble finger linkages**
   - Connect proximal/middle/distal links
   - Install bearings at pivot points
   - Ensure smooth motion through full ROM

2. **Install PCB**
   - Mount in enclosure on back of hand
   - Connect all cables

3. **Add comfort features**
   - Velcro straps at wrist and fingers
   - Silicone fingertip caps
   - Padding on pressure points

4. **Final wiring check**
   - Verify all connections
   - Check for shorts
   - Test continuity

---

## Firmware Installation

### Prerequisites

- STM32CubeIDE or arm-none-eabi-gcc
- ST-LINK V2 programmer
- USB cable

### Method 1: STM32CubeIDE

1. **Create project**
   ```
   File → New → STM32 Project
   Select: STM32F042K6T6
   ```

2. **Configure peripherals (CubeMX)**
   - SPI1: Mode=Full-Duplex Master, Baud=1MHz
   - I2C1: Mode=I2C, Speed=400kHz
   - USART1: Mode=Async, Baud=115200 (USB CDC)
   - USART2: Mode=Async, Baud=1000000 (Dynamixel)
   - TIM2: PWM for status LED
   - DMA1_CH1: SPI RX for ADC

3. **Copy firmware source**
   - Copy `main.c` to `Core/Src/`
   - Add HAL drivers for peripherals

4. **Build and flash**
   ```
   Project → Build All
   Run → Debug (with ST-LINK connected)
   ```

### Method 2: Command Line (arm-none-eabi-gcc)

```bash
# Install toolchain
sudo apt install gcc-arm-none-eabi

# Clone firmware
cd /path/to/dyglove_firmware

# Build
make all

# Flash via ST-LINK
st-flash write build/dyglove.bin 0x08000000
```

### Method 3: USB DFU Bootloader

1. **Enter DFU mode**
   - Hold BOOT0 button while powering on
   - Or: connect BOOT0 pin to VCC

2. **Flash via dfu-util**
   ```bash
   dfu-util -a 0 -D build/dyglove.bin -s 0x08000000
   ```

### Verify Installation

1. Connect USB to computer
2. Check for serial device:
   ```bash
   ls /dev/ttyACM*  # Linux
   ls /dev/cu.usb*  # macOS
   ```
3. Open serial terminal at 115200 baud
4. Send `0xAA 0x05 0x55` (GET_INFO command)
5. Should receive device info response

---

## Calibration Procedure

### Why Calibration is Necessary

The Alps RDC506018A encoders have ±2% linearity, which translates to ±7.2° error. Calibration creates a correction table that reduces this to ±1° accuracy.

### Calibration Types

1. **Quick Calibration** (2 minutes)
   - Records zero and max positions
   - Linear interpolation
   - Accuracy: ±3°

2. **Full Calibration** (10 minutes)
   - Records multiple positions
   - Polynomial correction
   - Accuracy: ±1°

### Quick Calibration Procedure

1. **Launch calibration mode**
   ```bash
   python -m src.platform.edge.dyglove_sdk calibrate --port /dev/ttyACM0
   ```

2. **Step 1: Zero position**
   - Extend all fingers fully (flat hand)
   - Keep hand still for 3 seconds
   - LED blinks slowly

3. **Step 2: Max position**
   - Flex all fingers fully (closed fist)
   - Keep hand still for 3 seconds
   - LED blinks fast

4. **Calibration complete**
   - "CAL:DONE" message appears
   - Data saved to flash memory

### Full Calibration Procedure

For best accuracy, use an external reference encoder (e.g., rotary encoder on jig).

1. **Setup calibration jig**
   - Mount finger in jig with reference encoder
   - Reference encoder should have <0.1° accuracy

2. **Run calibration script**
   ```bash
   python scripts/dyglove_full_calibration.py --port /dev/ttyACM0
   ```

3. **Record positions**
   - Script prompts for 10 positions per joint
   - Move finger to each position
   - Record both DYGlove reading and reference

4. **Compute correction**
   - Script fits polynomial: θ = a₀ + a₁V + a₂V² + a₃V³
   - Coefficients saved to flash

### Verifying Calibration

```python
from src.platform.edge.dyglove_sdk import DYGloveSDKClient

glove = DYGloveSDKClient()
glove.connect(port="/dev/ttyACM0")

# Read state
state = glove.get_state()

# Check angles are reasonable
print(f"21-DOF angles: {state.to_21dof_array()}")

# All angles should be near 0 with hand flat
# All angles should be 60-90° with fist closed
```

---

## Software Setup

### Python SDK Installation

```bash
# Clone repository
git clone https://github.com/dynamical-ai/dyglove-sdk.git
cd dyglove-sdk

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install SDK
pip install -e .
```

### Quick Start

```python
from src.platform.edge.dyglove_sdk import (
    DYGloveSDKClient, 
    DYGloveSimulator,
    Hand, 
    HapticWaveform
)

# Option 1: Real hardware
glove = DYGloveSDKClient()
glove.connect(port="/dev/ttyACM0")

# Option 2: Simulator (no hardware)
glove = DYGloveSimulator(side=Hand.RIGHT)
glove.connect()

# Read hand state (21 DOF)
state = glove.get_state()
print(f"Joint angles: {state.to_21dof_array()}")
print(f"Finger closures: {state.get_finger_closure()}")

# Send force feedback
glove.set_force_feedback([0.5, 0.3, 0.0, 0.0, 0.8])

# Send haptic feedback
glove.set_haptic_feedback(
    waveform_id=HapticWaveform.PULSING_SHARP_1_100,
    fingers=[0, 1]  # Thumb and index
)

# Combined feedback (as per DOGlove paper Table I)
glove.set_combined_feedback(force_readings=[120, 80, 30, 0, 0])

glove.disconnect()
```

### Streaming Mode

```python
def on_state(state):
    closure = state.get_finger_closure()
    if closure['index'] > 0.8:
        print("Grasping detected!")

glove.start_streaming(callback=on_state, rate_hz=120)

# ... do other things ...

glove.stop_streaming()
```

### WiFi Connection (DYGlove Extension)

```python
# Connect via WiFi instead of USB
glove.connect(wifi_ip="192.168.1.100")
```

---

## Usage Guide

### Wearing the Glove

1. Loosen all Velcro straps
2. Insert hand, aligning fingers with links
3. Secure wrist strap first (snug, not tight)
4. Secure finger straps
5. Adjust until comfortable with full ROM

### Operating Modes

| Mode | LED Pattern | Description |
|------|-------------|-------------|
| Idle | Slow blink (0.5 Hz) | Connected, not streaming |
| Streaming | Fast blink (5 Hz) | Sending data continuously |
| Calibrating | Very fast (10 Hz) | Calibration in progress |
| Error | Fastest (20 Hz) | Error condition |

### Best Practices

1. **Warm up period**: Let glove stabilize for 30 seconds after wearing
2. **Calibrate daily**: For best accuracy, run quick calibration each session
3. **Avoid moisture**: Keep electronics dry
4. **Cable management**: Route cables to avoid snagging
5. **Force feedback**: Start with low intensities, increase gradually

### Haptic Feedback Modes (DOGlove Table I)

| Force Sensor | Haptic | Force | Sensation |
|--------------|--------|-------|-----------|
| < 10g | ✗ | ✗ | No feedback (no contact) |
| 10-50g | ✓ | ✗ | Light touch (texture) |
| 50-100g | ✓ | ✓ | Contact with resistance |
| > 100g | ✗ | ✓ | Strong grip (force only) |

---

## Troubleshooting

### Common Issues

| Problem | Possible Cause | Solution |
|---------|---------------|----------|
| No serial device | Driver missing | Install STM32 VCP driver |
| Erratic readings | Poor calibration | Re-run calibration |
| Servo not moving | Wrong ID | Use Dynamixel Wizard to set ID |
| No haptic feedback | I2C address conflict | Check TCA9548A mux |
| High latency | USB polling | Use streaming mode |
| WiFi drops | Interference | Change WiFi channel |

### Diagnostic Commands

```bash
# Check USB connection
dmesg | grep ttyACM

# Monitor serial traffic
screen /dev/ttyACM0 115200

# Test encoders (raw ADC values)
python -c "
from dyglove_sdk import DYGloveSDKClient
g = DYGloveSDKClient()
g.connect()
print(g.get_state().to_21dof_array())
"
```

### Factory Reset

1. Hold BOOT0 button
2. Power on while holding
3. Flash firmware again
4. Calibration data is erased

---

## Design Considerations

### Encoder Selection: Alps RDC506018A

**Why this encoder:**
- Compact form factor (6mm diameter)
- Analog output (easy ADC interface)
- Low cost (~$8)
- 360° rotation without dead zone

**Trade-offs:**
- ±2% linearity requires calibration
- Temperature sensitivity (±0.3%/°C)
- Wear over time (rated 1M cycles)

**Alternative:** Magnetic encoders (AS5600) offer better linearity but larger size.

### ADC Selection: TI ADS1256

**Why this ADC:**
- 24-bit resolution (theoretical 0.00002°)
- 30 kSPS sample rate
- 8 channels (need 2 for 16 encoders, or use mux)
- Low noise (17 nV RMS)

**Trade-offs:**
- External module adds size
- SPI interface uses 4 pins
- Higher cost than internal ADC

**Alternative:** STM32 internal 12-bit ADC (lower resolution but simpler).

### Servo Selection: Dynamixel XL330

**Why this servo:**
- Compact (20×16.5×14mm)
- Smart protocol (position, current, temp feedback)
- Current-based position control (compliance)
- Daisy-chain communication

**Trade-offs:**
- Higher cost (~$24) vs hobby servos
- Requires protocol library
- 1M baud serial limits chain length

**Alternative:** Hobby servos (cheaper but no feedback).

### Haptic Actuator: 8mm LRA

**Why LRA:**
- Fast response (resonance @ 240Hz)
- Low power consumption
- Compact size
- Wide waveform library via DRV2605L

**Trade-offs:**
- Narrowband (best at resonance)
- Lower force than ERM motors
- Requires matched driver

**Alternative:** ERM motors (broader frequency but slower).

### MCU Selection: STM32F042K6T6

**Why this MCU:**
- Small package (LQFP32)
- USB device support (CDC)
- DMA for efficient ADC reads
- Low cost (~$3.50)
- 48MHz sufficient for 120Hz loop

**Trade-offs:**
- Limited RAM (6KB)
- Limited Flash (32KB)
- No FPU (float operations slower)

**Alternative:** STM32F103 (more RAM/Flash, larger package).

### WiFi Module: ESP32-C3

**Why ESP32-C3:**
- WiFi + BLE in one module
- RISC-V core (fast)
- Low power modes
- Small footprint

**Integration:**
- UART bridge to STM32
- Can run secondary firmware for WiFi stack
- Supports WiFi 6 (802.11ax) with some SKUs

### Cable-Driven Force Feedback

**Design principles:**
- **0.6mm steel cable**: Strong, low stretch
- **Teflon tubing**: Low friction cable routing
- **1:1 ratio**: Direct force transmission
- **Bidirectional**: Springs for return force

**Considerations:**
- Cable tension affects sensitivity
- Pulley alignment critical
- Regular cable inspection needed
- Finger cap attachment must be secure

### Power Budget

| Component | Current (mA) | Notes |
|-----------|-------------|-------|
| STM32 | 20 | Active mode |
| ADS1256 | 5 | Per chip |
| Dynamixel ×5 | 200 (idle) / 800 (stall) | Total for 5 |
| DRV2605L ×5 | 5 (idle) / 50 (active) | Per chip |
| MPU6050 | 4 | |
| ESP32-C3 | 80 (WiFi active) | Optional |
| **Total** | ~350 (idle) / ~1000 (active) | |

**Battery sizing:**
- 1000mAh Li-Po provides ~3 hours active use
- USB power recommended for extended sessions

### Thermal Considerations

- Servos generate heat under load
- DRV2605L has thermal shutdown at 150°C
- STM32 rated to 85°C
- Add ventilation holes in enclosure
- Avoid continuous high-force operation

---

## Appendix: Pin Mapping

### STM32F042K6T6 Pinout

| Pin | Function | Connected To |
|-----|----------|--------------|
| PA0 | ADC_IN0 | (Reserved) |
| PA4 | SPI1_NSS | ADS1256 CS |
| PA5 | SPI1_SCK | ADS1256 CLK |
| PA6 | SPI1_MISO | ADS1256 DOUT |
| PA7 | SPI1_MOSI | ADS1256 DIN |
| PA8 | TIM1_CH1 | Haptic Enable |
| PA9 | USART1_TX | USB (via internal) |
| PA10 | USART1_RX | USB (via internal) |
| PA11 | USB_DM | USB D- |
| PA12 | USB_DP | USB D+ |
| PB6 | I2C1_SCL | IMU, Haptics |
| PB7 | I2C1_SDA | IMU, Haptics |
| PB10 | USART3_TX | Dynamixel TX |
| PB11 | USART3_RX | Dynamixel RX |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2025-01 | DOGlove paper compliance, 21-DOF |
| 1.0.0 | 2024-06 | Initial release |

---

**Support:** support@dynamical.ai  
**Documentation:** https://docs.dynamical.ai/dyglove  
**Source:** https://github.com/dynamical-ai/dyglove
