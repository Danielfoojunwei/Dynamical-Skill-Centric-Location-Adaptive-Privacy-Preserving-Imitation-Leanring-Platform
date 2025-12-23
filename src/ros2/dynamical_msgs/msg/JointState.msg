# Extended joint state with limits and status
# Used internally for safety checking

std_msgs/Header header

string[] joint_names
float64[] positions            # radians
float64[] velocities           # rad/s
float64[] efforts              # Nm
float64[] temperatures         # Celsius

# Limits (for runtime checking)
float64[] position_limits_lower
float64[] position_limits_upper
float64[] velocity_limits
float64[] effort_limits

# Status per joint
bool[] enabled
bool[] has_error
uint8[] error_codes
