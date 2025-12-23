# Safety system status
# Published at 1kHz from safety shield

std_msgs/Header header

# Status levels
uint8 STATUS_OK = 0
uint8 STATUS_WARNING = 1
uint8 STATUS_VIOLATION = 2
uint8 STATUS_ESTOP = 3

uint8 status

# Violations
SafetyViolation[] violations

# Flags
bool estop_active
bool human_detected
bool obstacle_detected

# Distances
float64 min_obstacle_distance  # meters
float64 min_human_distance     # meters

# Timing
float64 last_check_time_us
uint32 missed_deadlines_count
