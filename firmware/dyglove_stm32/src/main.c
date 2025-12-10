/**
 * DYGlove STM32 Firmware
 * 
 * Firmware for the DYGlove haptic glove based on DOGlove design.
 * Runs on STM32F042K6T6 @ 48MHz with DMA-accelerated ADC.
 * 
 * Reference: DOGlove (arXiv:2502.07730)
 * 
 * Hardware Configuration:
 * - MCU: STM32F042K6T6 (32KB Flash, 6KB SRAM, 48MHz)
 * - ADC: TI ADS1256 24-bit external ADC @ 30kHz
 * - Encoders: 16x Alps RDC506018A rotary encoders
 * - Servos: 5x Dynamixel XC330/XL330 for force feedback
 * - Haptics: 5x 8mm LRA @ 240Hz with DRV2605L drivers
 * - IMU: MPU6050 for wrist orientation (optional)
 * - WiFi: ESP32-C3 module for wireless (DYGlove extension)
 * 
 * Communication Protocol:
 * - USB CDC @ 115200 baud (primary)
 * - WiFi UDP @ 120Hz for streaming
 * - WiFi TCP for control commands
 * 
 * Build Instructions:
 *   1. Install STM32CubeIDE or arm-none-eabi-gcc
 *   2. Configure project for STM32F042K6T6
 *   3. Flash via ST-LINK or USB DFU bootloader
 * 
 * @author Dynamical.ai
 * @version 2.0.0
 * @date 2025
 */

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Configuration
 * ============================================================================ */

#define FIRMWARE_VERSION_MAJOR  2
#define FIRMWARE_VERSION_MINOR  0
#define FIRMWARE_VERSION_PATCH  0

// Hardware pins (STM32F042K6T6 LQFP32)
#define PIN_LED_STATUS      PA5
#define PIN_LED_ERROR       PA6
#define PIN_SPI_SCK         PA5
#define PIN_SPI_MISO        PA6
#define PIN_SPI_MOSI        PA7
#define PIN_SPI_CS_ADC      PA4
#define PIN_I2C_SDA         PB7
#define PIN_I2C_SCL         PB6
#define PIN_UART_TX         PA9
#define PIN_UART_RX         PA10
#define PIN_SERVO_TX        PB10
#define PIN_SERVO_RX        PB11
#define PIN_HAPTIC_EN       PA8

// Timing
#define MOCAP_RATE_HZ       120     // Motion capture sampling rate
#define HAPTIC_RATE_HZ      30      // Haptic feedback update rate
#define ADC_SAMPLE_RATE     30000   // ADS1256 sample rate
#define SERVO_BAUDRATE      1000000 // Dynamixel baudrate

// Protocol constants
#define PACKET_HEADER       0xAA
#define PACKET_FOOTER       0x55
#define STATE_PACKET_SIZE   64
#define CMD_GET_STATE       0x01
#define CMD_SET_FORCE       0x02
#define CMD_SET_HAPTIC      0x03
#define CMD_CALIBRATE       0x04
#define CMD_GET_INFO        0x05
#define CMD_STREAM_START    0x10
#define CMD_STREAM_STOP     0x11

// Encoder configuration
#define NUM_ENCODERS        16      // Physical rotary encoders
#define NUM_JOINTS          21      // Total DOF
#define ADC_RESOLUTION      24      // Bits
#define ADC_VREF            3.3f    // Reference voltage

// Joint indices (matches DYGloveJoint enum in SDK)
#define THUMB_TM_FLEX       0
#define THUMB_TM_ABD        1
#define THUMB_MCP           2
#define THUMB_IP            3
#define THUMB_WRIST_PS      4
#define INDEX_MCP_FLEX      5
#define INDEX_MCP_ABD       6
#define INDEX_PIP           7
#define INDEX_DIP           8
#define MIDDLE_MCP_FLEX     9
#define MIDDLE_MCP_ABD      10
#define MIDDLE_PIP          11
#define MIDDLE_DIP          12
#define RING_MCP_FLEX       13
#define RING_MCP_ABD        14
#define RING_PIP            15
#define RING_DIP            16
#define PINKY_MCP_FLEX      17
#define PINKY_MCP_ABD       18
#define PINKY_PIP           19
#define PINKY_DIP           20

// Servo IDs (Dynamixel protocol)
#define SERVO_THUMB         1
#define SERVO_INDEX         2
#define SERVO_MIDDLE        3
#define SERVO_RING          4
#define SERVO_PINKY         5

// DRV2605L haptic driver addresses
#define DRV2605_ADDR_BASE   0x5A    // I2C address (all same, use mux)


/* ============================================================================
 * Type Definitions
 * ============================================================================ */

typedef struct {
    float angle;                // Calibrated angle in degrees
    float raw_voltage;          // Raw ADC voltage (0-3.3V)
    uint32_t raw_adc;           // 24-bit ADC count
    bool valid;                 // Encoder reading valid
} EncoderReading;

typedef struct {
    float joint_angles[NUM_JOINTS];     // Calibrated angles (degrees)
    int16_t joint_angles_int[NUM_JOINTS]; // Scaled for transmission (×100)
    uint16_t servo_positions[5];        // Current servo positions
    float quaternion[4];                // Wrist IMU quaternion [w,x,y,z]
    uint8_t battery_percent;            // Battery level 0-100
    uint8_t temperature;                // Board temperature °C
    uint32_t timestamp_ms;              // Millisecond timestamp
    uint8_t sequence;                   // Packet sequence number
} GloveState;

typedef struct {
    uint16_t pwm[5];            // Servo PWM values (0-1023)
    uint8_t enable_mask;        // Which servos are active
} ForceCommand;

typedef struct {
    uint8_t waveform_id;        // DRV2605L waveform (0-123)
    uint8_t finger_mask;        // Which fingers (5 bits)
    uint16_t duration_ms;       // Duration
    uint8_t intensity;          // 0-255 intensity
} HapticCommand;

typedef struct {
    // Per-encoder calibration
    float voltage_offset[NUM_ENCODERS];     // Zero offset voltage
    float voltage_scale[NUM_ENCODERS];      // Scale factor
    float angle_offset[NUM_ENCODERS];       // Angle offset after scaling
    
    // Polynomial correction coefficients (for linearity)
    // angle = a0 + a1*V + a2*V² + a3*V³
    float poly_coeff[NUM_ENCODERS][4];
    
    // Joint limits
    float joint_min[NUM_JOINTS];
    float joint_max[NUM_JOINTS];
    
    // Calibration valid flag
    bool valid;
    uint32_t timestamp;
} CalibrationData;

typedef enum {
    STATE_IDLE,
    STATE_STREAMING,
    STATE_CALIBRATING,
    STATE_ERROR
} FirmwareState;


/* ============================================================================
 * Global Variables
 * ============================================================================ */

static GloveState g_state;
static CalibrationData g_calibration;
static FirmwareState g_firmware_state = STATE_IDLE;

static uint8_t g_tx_buffer[STATE_PACKET_SIZE];
static uint8_t g_rx_buffer[64];
static volatile bool g_streaming_enabled = false;
static volatile uint32_t g_last_stream_time = 0;

// DMA buffers for ADC
static uint32_t g_adc_dma_buffer[NUM_ENCODERS];
static volatile bool g_adc_ready = false;

// Encoder to joint mapping
// Some joints share encoders, some are derived (e.g., DIP ≈ 0.67 × PIP)
static const int8_t ENCODER_TO_JOINT[NUM_ENCODERS] = {
    THUMB_TM_FLEX,      // Encoder 0
    THUMB_TM_ABD,       // Encoder 1
    THUMB_MCP,          // Encoder 2
    THUMB_IP,           // Encoder 3
    INDEX_MCP_FLEX,     // Encoder 4
    INDEX_MCP_ABD,      // Encoder 5
    INDEX_PIP,          // Encoder 6
    MIDDLE_MCP_FLEX,    // Encoder 7
    MIDDLE_MCP_ABD,     // Encoder 8
    MIDDLE_PIP,         // Encoder 9
    RING_MCP_FLEX,      // Encoder 10
    RING_MCP_ABD,       // Encoder 11
    RING_PIP,           // Encoder 12
    PINKY_MCP_FLEX,     // Encoder 13
    PINKY_MCP_ABD,      // Encoder 14
    PINKY_PIP,          // Encoder 15
};

// Coupled joint ratios (for joints without direct encoders)
#define DIP_PIP_RATIO       0.67f   // DIP ≈ 0.67 × PIP (anatomical coupling)
#define WRIST_PS_IMU        true    // Wrist PS from IMU, not encoder


/* ============================================================================
 * Hardware Abstraction (Platform-specific implementations)
 * ============================================================================ */

// These would be implemented with STM32 HAL/LL drivers
extern void HAL_Init(void);
extern void SystemClock_Config(void);
extern void GPIO_Init(void);
extern void SPI_Init(void);
extern void I2C_Init(void);
extern void UART_Init(void);
extern void TIM_Init(void);
extern void DMA_Init(void);

extern uint32_t HAL_GetTick(void);
extern void HAL_Delay(uint32_t ms);

// ADC functions
extern void ADS1256_Init(void);
extern uint32_t ADS1256_ReadChannel(uint8_t channel);
extern void ADS1256_StartDMA(uint32_t* buffer, uint8_t num_channels);
extern bool ADS1256_DMAComplete(void);

// Servo functions (Dynamixel protocol)
extern void Dynamixel_Init(uint32_t baudrate);
extern bool Dynamixel_SetPosition(uint8_t id, uint16_t position);
extern bool Dynamixel_SetTorque(uint8_t id, uint16_t torque);
extern uint16_t Dynamixel_GetPosition(uint8_t id);
extern bool Dynamixel_SetCurrentBasedPosition(uint8_t id, uint16_t position, uint16_t current_limit);

// Haptic driver functions (DRV2605L)
extern void DRV2605_Init(uint8_t index);
extern void DRV2605_SetWaveform(uint8_t index, uint8_t waveform_id);
extern void DRV2605_Go(uint8_t index);
extern void DRV2605_Stop(uint8_t index);

// IMU functions (MPU6050)
extern void MPU6050_Init(void);
extern void MPU6050_GetQuaternion(float* quat);

// USB CDC functions
extern void USB_CDC_Init(void);
extern int USB_CDC_Transmit(uint8_t* data, uint16_t len);
extern int USB_CDC_Receive(uint8_t* data, uint16_t max_len);
extern bool USB_CDC_IsConnected(void);

// WiFi functions (ESP32-C3)
extern void WiFi_Init(const char* ssid, const char* password);
extern bool WiFi_IsConnected(void);
extern int WiFi_UDP_Send(uint8_t* data, uint16_t len, const char* ip, uint16_t port);
extern int WiFi_TCP_Send(uint8_t* data, uint16_t len);
extern int WiFi_TCP_Receive(uint8_t* data, uint16_t max_len);

// Flash storage
extern void Flash_Write(uint32_t address, uint8_t* data, uint32_t len);
extern void Flash_Read(uint32_t address, uint8_t* data, uint32_t len);


/* ============================================================================
 * Calibration Functions
 * ============================================================================ */

/**
 * Initialize calibration with default values.
 */
void Calibration_Init(void) {
    memset(&g_calibration, 0, sizeof(CalibrationData));
    
    // Default linear conversion: angle = (V / 3.3) * 360
    for (int i = 0; i < NUM_ENCODERS; i++) {
        g_calibration.voltage_offset[i] = 0.0f;
        g_calibration.voltage_scale[i] = 1.0f;
        g_calibration.angle_offset[i] = 0.0f;
        
        // Default polynomial: linear
        g_calibration.poly_coeff[i][0] = 0.0f;
        g_calibration.poly_coeff[i][1] = 360.0f / 3.3f;  // ~109.09 deg/V
        g_calibration.poly_coeff[i][2] = 0.0f;
        g_calibration.poly_coeff[i][3] = 0.0f;
    }
    
    // Default joint limits (degrees)
    // Thumb
    g_calibration.joint_min[THUMB_TM_FLEX] = -20.0f;
    g_calibration.joint_max[THUMB_TM_FLEX] = 70.0f;
    g_calibration.joint_min[THUMB_TM_ABD] = -30.0f;
    g_calibration.joint_max[THUMB_TM_ABD] = 30.0f;
    g_calibration.joint_min[THUMB_MCP] = -15.0f;
    g_calibration.joint_max[THUMB_MCP] = 90.0f;
    g_calibration.joint_min[THUMB_IP] = -10.0f;
    g_calibration.joint_max[THUMB_IP] = 80.0f;
    g_calibration.joint_min[THUMB_WRIST_PS] = -90.0f;
    g_calibration.joint_max[THUMB_WRIST_PS] = 90.0f;
    
    // Fingers (same limits for index, middle, ring, pinky)
    for (int f = 0; f < 4; f++) {
        int base = 5 + f * 4;  // INDEX_MCP_FLEX starts at 5
        g_calibration.joint_min[base + 0] = -20.0f;   // MCP flex
        g_calibration.joint_max[base + 0] = 90.0f;
        g_calibration.joint_min[base + 1] = -20.0f;   // MCP abd
        g_calibration.joint_max[base + 1] = 20.0f;
        g_calibration.joint_min[base + 2] = 0.0f;     // PIP
        g_calibration.joint_max[base + 2] = 100.0f;
        g_calibration.joint_min[base + 3] = 0.0f;     // DIP
        g_calibration.joint_max[base + 3] = 70.0f;
    }
    
    g_calibration.valid = false;
}

/**
 * Convert raw ADC voltage to calibrated angle.
 * 
 * Uses polynomial correction for encoder non-linearity:
 *   angle = a0 + a1*V + a2*V² + a3*V³
 * 
 * @param encoder_idx Encoder index (0-15)
 * @param voltage Raw ADC voltage (0-3.3V)
 * @return Calibrated angle in degrees
 */
float Calibration_VoltageToAngle(uint8_t encoder_idx, float voltage) {
    if (encoder_idx >= NUM_ENCODERS) return 0.0f;
    
    // Apply offset and scale
    float v = (voltage - g_calibration.voltage_offset[encoder_idx]) 
              * g_calibration.voltage_scale[encoder_idx];
    
    // Polynomial evaluation (Horner's method)
    float* c = g_calibration.poly_coeff[encoder_idx];
    float angle = c[0] + v * (c[1] + v * (c[2] + v * c[3]));
    
    // Apply angle offset
    angle += g_calibration.angle_offset[encoder_idx];
    
    return angle;
}

/**
 * Run interactive calibration procedure.
 * 
 * Procedure:
 *   1. User extends all fingers fully (zero position)
 *   2. User flexes all fingers fully (max position)
 *   3. Firmware computes offset and scale
 */
void Calibration_Run(void) {
    g_firmware_state = STATE_CALIBRATING;
    
    // Step 1: Record zero position
    float zero_voltages[NUM_ENCODERS];
    USB_CDC_Transmit((uint8_t*)"CAL:EXTEND_ALL\n", 15);
    HAL_Delay(3000);  // Wait for user
    
    for (int i = 0; i < NUM_ENCODERS; i++) {
        uint32_t raw = ADS1256_ReadChannel(i);
        zero_voltages[i] = (float)raw * ADC_VREF / (1 << ADC_RESOLUTION);
    }
    
    // Step 2: Record max position
    float max_voltages[NUM_ENCODERS];
    USB_CDC_Transmit((uint8_t*)"CAL:FLEX_ALL\n", 13);
    HAL_Delay(3000);
    
    for (int i = 0; i < NUM_ENCODERS; i++) {
        uint32_t raw = ADS1256_ReadChannel(i);
        max_voltages[i] = (float)raw * ADC_VREF / (1 << ADC_RESOLUTION);
    }
    
    // Step 3: Compute calibration
    for (int i = 0; i < NUM_ENCODERS; i++) {
        int joint = ENCODER_TO_JOINT[i];
        float range = max_voltages[i] - zero_voltages[i];
        float joint_range = g_calibration.joint_max[joint] - g_calibration.joint_min[joint];
        
        if (fabsf(range) > 0.1f) {  // Valid range
            g_calibration.voltage_offset[i] = zero_voltages[i];
            g_calibration.poly_coeff[i][1] = joint_range / range;
            g_calibration.angle_offset[i] = g_calibration.joint_min[joint];
        }
    }
    
    g_calibration.valid = true;
    g_calibration.timestamp = HAL_GetTick();
    
    // Save to flash
    Flash_Write(0x08007000, (uint8_t*)&g_calibration, sizeof(CalibrationData));
    
    USB_CDC_Transmit((uint8_t*)"CAL:DONE\n", 9);
    g_firmware_state = STATE_IDLE;
}

/**
 * Load calibration from flash.
 */
void Calibration_Load(void) {
    Flash_Read(0x08007000, (uint8_t*)&g_calibration, sizeof(CalibrationData));
    
    // Validate
    if (g_calibration.timestamp == 0 || g_calibration.timestamp == 0xFFFFFFFF) {
        Calibration_Init();  // Use defaults
    }
}


/* ============================================================================
 * Encoder Reading Functions
 * ============================================================================ */

/**
 * Read all encoders and update state.
 */
void Encoders_Read(void) {
    // Start DMA transfer
    ADS1256_StartDMA(g_adc_dma_buffer, NUM_ENCODERS);
    
    // Wait for completion (with timeout)
    uint32_t start = HAL_GetTick();
    while (!ADS1256_DMAComplete()) {
        if (HAL_GetTick() - start > 10) {
            return;  // Timeout
        }
    }
    
    // Convert to angles
    for (int i = 0; i < NUM_ENCODERS; i++) {
        float voltage = (float)g_adc_dma_buffer[i] * ADC_VREF / (1 << ADC_RESOLUTION);
        int joint = ENCODER_TO_JOINT[i];
        
        float angle = Calibration_VoltageToAngle(i, voltage);
        
        // Clamp to limits
        if (angle < g_calibration.joint_min[joint]) {
            angle = g_calibration.joint_min[joint];
        } else if (angle > g_calibration.joint_max[joint]) {
            angle = g_calibration.joint_max[joint];
        }
        
        g_state.joint_angles[joint] = angle;
    }
    
    // Compute coupled joints (DIP = 0.67 * PIP)
    g_state.joint_angles[INDEX_DIP] = g_state.joint_angles[INDEX_PIP] * DIP_PIP_RATIO;
    g_state.joint_angles[MIDDLE_DIP] = g_state.joint_angles[MIDDLE_PIP] * DIP_PIP_RATIO;
    g_state.joint_angles[RING_DIP] = g_state.joint_angles[RING_PIP] * DIP_PIP_RATIO;
    g_state.joint_angles[PINKY_DIP] = g_state.joint_angles[PINKY_PIP] * DIP_PIP_RATIO;
    
    // Get wrist PS from IMU if available
    if (WRIST_PS_IMU) {
        float quat[4];
        MPU6050_GetQuaternion(quat);
        memcpy(g_state.quaternion, quat, sizeof(quat));
        
        // Extract yaw as wrist pronation/supination
        // yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        float yaw = atan2f(2.0f * (quat[0]*quat[3] + quat[1]*quat[2]),
                          1.0f - 2.0f * (quat[2]*quat[2] + quat[3]*quat[3]));
        g_state.joint_angles[THUMB_WRIST_PS] = yaw * 180.0f / 3.14159f;
    }
    
    // Scale for transmission (×100 for 0.01° resolution)
    for (int i = 0; i < NUM_JOINTS; i++) {
        g_state.joint_angles_int[i] = (int16_t)(g_state.joint_angles[i] * 100.0f);
    }
}


/* ============================================================================
 * Force Feedback Functions
 * ============================================================================ */

/**
 * Set force feedback on servos.
 * 
 * Uses current-based position control for compliance.
 */
void Force_SetFeedback(ForceCommand* cmd) {
    for (int i = 0; i < 5; i++) {
        if (cmd->enable_mask & (1 << i)) {
            uint8_t servo_id = SERVO_THUMB + i;
            
            // Convert PWM to Dynamixel current limit
            // DOGlove maps force [0g, 3000g] linearly to PWM
            uint16_t current_limit = cmd->pwm[i];  // Already 0-1023
            
            // Set current-based position control
            Dynamixel_SetCurrentBasedPosition(servo_id, 512, current_limit);
        }
    }
}

/**
 * Read current servo positions.
 */
void Force_ReadPositions(void) {
    for (int i = 0; i < 5; i++) {
        g_state.servo_positions[i] = Dynamixel_GetPosition(SERVO_THUMB + i);
    }
}


/* ============================================================================
 * Haptic Feedback Functions
 * ============================================================================ */

/**
 * Trigger haptic feedback.
 * 
 * Uses DRV2605L with Immersion TouchSense waveform library.
 */
void Haptic_Trigger(HapticCommand* cmd) {
    for (int i = 0; i < 5; i++) {
        if (cmd->finger_mask & (1 << i)) {
            DRV2605_SetWaveform(i, cmd->waveform_id);
            DRV2605_Go(i);
        } else {
            DRV2605_Stop(i);
        }
    }
}


/* ============================================================================
 * Communication Protocol
 * ============================================================================ */

/**
 * Build state packet for transmission.
 * 
 * Format (64 bytes):
 *   [0]: Header (0xAA)
 *   [1]: Sequence number
 *   [2-43]: 21 joint angles (int16 × 21 = 42 bytes, scaled ×100)
 *   [44-53]: 5 servo positions (int16 × 5 = 10 bytes)
 *   [54-61]: Quaternion (float16 × 4 = 8 bytes)
 *   [62]: Battery level (uint8)
 *   [63]: Footer (0x55)
 */
void Protocol_BuildStatePacket(uint8_t* buffer) {
    buffer[0] = PACKET_HEADER;
    buffer[1] = g_state.sequence++;
    
    // Joint angles (21 × int16)
    for (int i = 0; i < NUM_JOINTS; i++) {
        int16_t angle = g_state.joint_angles_int[i];
        buffer[2 + i*2] = angle & 0xFF;
        buffer[2 + i*2 + 1] = (angle >> 8) & 0xFF;
    }
    
    // Servo positions (5 × int16)
    for (int i = 0; i < 5; i++) {
        int16_t pos = g_state.servo_positions[i];
        buffer[44 + i*2] = pos & 0xFF;
        buffer[44 + i*2 + 1] = (pos >> 8) & 0xFF;
    }
    
    // Quaternion (4 × float16) - simplified to int16
    for (int i = 0; i < 4; i++) {
        int16_t q = (int16_t)(g_state.quaternion[i] * 10000.0f);
        buffer[54 + i*2] = q & 0xFF;
        buffer[54 + i*2 + 1] = (q >> 8) & 0xFF;
    }
    
    // Battery
    buffer[62] = g_state.battery_percent;
    
    // Footer
    buffer[63] = PACKET_FOOTER;
}

/**
 * Parse received command.
 */
void Protocol_ParseCommand(uint8_t* buffer, uint16_t len) {
    if (len < 3) return;
    if (buffer[0] != PACKET_HEADER) return;
    
    uint8_t cmd = buffer[1];
    
    switch (cmd) {
        case CMD_GET_STATE:
            // Single state request
            Protocol_BuildStatePacket(g_tx_buffer);
            USB_CDC_Transmit(g_tx_buffer, STATE_PACKET_SIZE);
            break;
            
        case CMD_SET_FORCE: {
            ForceCommand force_cmd = {0};
            force_cmd.enable_mask = 0x1F;  // All enabled
            for (int i = 0; i < 5; i++) {
                force_cmd.pwm[i] = buffer[2 + i*2] | (buffer[3 + i*2] << 8);
            }
            Force_SetFeedback(&force_cmd);
            break;
        }
        
        case CMD_SET_HAPTIC: {
            HapticCommand haptic_cmd = {0};
            haptic_cmd.waveform_id = buffer[2] & 0x7F;
            haptic_cmd.finger_mask = buffer[3] & 0x1F;
            haptic_cmd.duration_ms = buffer[4] | (buffer[5] << 8);
            haptic_cmd.intensity = buffer[6];
            Haptic_Trigger(&haptic_cmd);
            break;
        }
        
        case CMD_CALIBRATE:
            Calibration_Run();
            break;
            
        case CMD_GET_INFO: {
            // Send device info
            uint8_t info[32] = {0};
            info[0] = PACKET_HEADER;
            info[1] = 0x85;  // RSP_INFO
            // Serial number (16 bytes)
            strcpy((char*)&info[2], "DYGLOVE-001");
            // Firmware version
            info[18] = FIRMWARE_VERSION_MAJOR;
            info[19] = FIRMWARE_VERSION_MINOR;
            info[20] = FIRMWARE_VERSION_PATCH;
            // Hand (0=left, 1=right)
            info[21] = 1;  // Right hand
            info[31] = PACKET_FOOTER;
            USB_CDC_Transmit(info, 32);
            break;
        }
        
        case CMD_STREAM_START:
            g_streaming_enabled = true;
            g_firmware_state = STATE_STREAMING;
            break;
            
        case CMD_STREAM_STOP:
            g_streaming_enabled = false;
            g_firmware_state = STATE_IDLE;
            break;
    }
}


/* ============================================================================
 * Main Loop
 * ============================================================================ */

/**
 * Update task - called at MOCAP_RATE_HZ (120 Hz).
 */
void Update_Task(void) {
    static uint32_t last_update = 0;
    uint32_t now = HAL_GetTick();
    
    if (now - last_update < (1000 / MOCAP_RATE_HZ)) {
        return;  // Not time yet
    }
    last_update = now;
    
    // Read encoders
    Encoders_Read();
    
    // Read servo positions
    Force_ReadPositions();
    
    // Update timestamp
    g_state.timestamp_ms = now;
    
    // Stream if enabled
    if (g_streaming_enabled) {
        Protocol_BuildStatePacket(g_tx_buffer);
        
        // Send via USB
        if (USB_CDC_IsConnected()) {
            USB_CDC_Transmit(g_tx_buffer, STATE_PACKET_SIZE);
        }
        
        // Send via WiFi UDP if connected
        if (WiFi_IsConnected()) {
            WiFi_UDP_Send(g_tx_buffer, STATE_PACKET_SIZE, "255.255.255.255", 9876);
        }
    }
}

/**
 * Communication task - check for incoming commands.
 */
void Communication_Task(void) {
    // Check USB
    int len = USB_CDC_Receive(g_rx_buffer, sizeof(g_rx_buffer));
    if (len > 0) {
        Protocol_ParseCommand(g_rx_buffer, len);
    }
    
    // Check WiFi TCP
    if (WiFi_IsConnected()) {
        len = WiFi_TCP_Receive(g_rx_buffer, sizeof(g_rx_buffer));
        if (len > 0) {
            Protocol_ParseCommand(g_rx_buffer, len);
        }
    }
}

/**
 * Status LED task.
 */
void LED_Task(void) {
    static uint32_t last_blink = 0;
    static bool led_on = false;
    uint32_t now = HAL_GetTick();
    
    uint32_t blink_rate = 1000;  // Default 1 Hz
    
    switch (g_firmware_state) {
        case STATE_IDLE:
            blink_rate = 2000;  // 0.5 Hz
            break;
        case STATE_STREAMING:
            blink_rate = 200;   // 5 Hz
            break;
        case STATE_CALIBRATING:
            blink_rate = 100;   // 10 Hz
            break;
        case STATE_ERROR:
            blink_rate = 50;    // 20 Hz
            break;
    }
    
    if (now - last_blink >= blink_rate) {
        last_blink = now;
        led_on = !led_on;
        // GPIO_Write(PIN_LED_STATUS, led_on);
    }
}

/**
 * Main entry point.
 */
int main(void) {
    // Initialize hardware
    HAL_Init();
    SystemClock_Config();
    GPIO_Init();
    SPI_Init();
    I2C_Init();
    UART_Init();
    TIM_Init();
    DMA_Init();
    
    // Initialize peripherals
    ADS1256_Init();
    Dynamixel_Init(SERVO_BAUDRATE);
    for (int i = 0; i < 5; i++) {
        DRV2605_Init(i);
    }
    MPU6050_Init();
    USB_CDC_Init();
    
    // WiFi (optional - comment out for USB-only version)
    // WiFi_Init("DYGlove_Network", "password");
    
    // Load calibration
    Calibration_Init();
    Calibration_Load();
    
    // Initialize state
    memset(&g_state, 0, sizeof(GloveState));
    g_state.battery_percent = 100;
    g_state.quaternion[0] = 1.0f;  // Identity quaternion
    
    g_firmware_state = STATE_IDLE;
    
    // Main loop
    while (1) {
        Update_Task();
        Communication_Task();
        LED_Task();
    }
    
    return 0;
}


/* ============================================================================
 * Interrupt Handlers
 * ============================================================================ */

/**
 * ADC DMA complete interrupt.
 */
void DMA1_Channel1_IRQHandler(void) {
    g_adc_ready = true;
    // Clear interrupt flag
}

/**
 * USB interrupt.
 */
void USB_IRQHandler(void) {
    // Handle USB events
}

/**
 * SysTick interrupt (1ms).
 */
void SysTick_Handler(void) {
    // HAL_IncTick();
}
