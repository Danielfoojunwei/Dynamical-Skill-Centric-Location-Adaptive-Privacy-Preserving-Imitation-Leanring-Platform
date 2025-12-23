# Individual safety violation details

# Violation types
uint8 TYPE_POSITION_LIMIT = 0
uint8 TYPE_VELOCITY_LIMIT = 1
uint8 TYPE_TORQUE_LIMIT = 2
uint8 TYPE_OBSTACLE_PROXIMITY = 3
uint8 TYPE_HUMAN_PROXIMITY = 4
uint8 TYPE_WATCHDOG_TIMEOUT = 5
uint8 TYPE_COMMUNICATION_LOSS = 6

# Severity levels
uint8 SEVERITY_WARNING = 0
uint8 SEVERITY_VIOLATION = 1
uint8 SEVERITY_CRITICAL = 2

uint8 type
uint8 severity
int32 joint_index           # -1 if not joint-specific
float64 current_value
float64 limit_value
string message
