/**
 * @file safety_node.cpp
 * @brief Safety Shield Node Implementation - 1kHz real-time safety monitoring
 */

#include "dynamical_runtime/safety_node.hpp"

#include <sched.h>
#include <sys/mman.h>
#include <chrono>
#include <cmath>

namespace dynamical
{

using namespace std::chrono_literals;
using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

SafetyNode::SafetyNode(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("safety_node", options)
{
  // Declare parameters
  this->declare_parameter("rate_hz", 1000);
  this->declare_parameter("num_joints", 7);
  this->declare_parameter("joint_limit_margin_deg", 5.0);
  this->declare_parameter("velocity_limit_margin_percent", 10.0);
  this->declare_parameter("torque_limit_margin_percent", 20.0);
  this->declare_parameter("min_obstacle_distance_m", 0.10);
  this->declare_parameter("human_safety_distance_m", 0.50);
  this->declare_parameter("watchdog_timeout_ms", 10);
  this->declare_parameter("use_realtime_scheduling", true);

  // Joint limits (default for 7-DOF arm)
  this->declare_parameter("joint_limits_lower",
    std::vector<double>{-3.14, -2.09, -3.14, -2.09, -3.14, -2.09, -3.14});
  this->declare_parameter("joint_limits_upper",
    std::vector<double>{3.14, 2.09, 3.14, 2.09, 3.14, 2.09, 3.14});
  this->declare_parameter("velocity_limits",
    std::vector<double>{3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14});
  this->declare_parameter("torque_limits",
    std::vector<double>{100.0, 100.0, 50.0, 50.0, 25.0, 25.0, 10.0});

  RCLCPP_INFO(this->get_logger(), "SafetyNode created");
}

SafetyNode::~SafetyNode()
{
  RCLCPP_INFO(this->get_logger(), "SafetyNode destroyed");
}

CallbackReturn SafetyNode::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Configuring SafetyNode...");

  // Load parameters
  num_joints_ = this->get_parameter("num_joints").as_int();
  config_.joint_limit_margin = this->get_parameter("joint_limit_margin_deg").as_double() * M_PI / 180.0;
  config_.velocity_limit_margin = this->get_parameter("velocity_limit_margin_percent").as_double() / 100.0;
  config_.torque_limit_margin = this->get_parameter("torque_limit_margin_percent").as_double() / 100.0;
  config_.min_obstacle_distance = this->get_parameter("min_obstacle_distance_m").as_double();
  config_.human_safety_distance = this->get_parameter("human_safety_distance_m").as_double();
  config_.watchdog_timeout_ms = this->get_parameter("watchdog_timeout_ms").as_int();

  joint_limits_lower_ = this->get_parameter("joint_limits_lower").as_double_array();
  joint_limits_upper_ = this->get_parameter("joint_limits_upper").as_double_array();
  velocity_limits_ = this->get_parameter("velocity_limits").as_double_array();
  torque_limits_ = this->get_parameter("torque_limits").as_double_array();

  // Create subscribers
  robot_state_sub_ = this->create_subscription<dynamical_msgs::msg::RobotState>(
    "robot_state", rclcpp::SensorDataQoS(),
    std::bind(&SafetyNode::robot_state_callback, this, std::placeholders::_1));

  obstacles_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
    "obstacles", rclcpp::SensorDataQoS(),
    std::bind(&SafetyNode::obstacles_callback, this, std::placeholders::_1));

  humans_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
    "humans", rclcpp::SensorDataQoS(),
    std::bind(&SafetyNode::humans_callback, this, std::placeholders::_1));

  heartbeat_sub_ = this->create_subscription<std_msgs::msg::Bool>(
    "heartbeat", rclcpp::SensorDataQoS(),
    std::bind(&SafetyNode::heartbeat_callback, this, std::placeholders::_1));

  // Create publishers (real-time safe)
  auto safety_status_pub = this->create_publisher<dynamical_msgs::msg::SafetyStatus>(
    "safety_status", rclcpp::SensorDataQoS());
  safety_status_pub_ = std::make_shared<
    realtime_tools::RealtimePublisher<dynamical_msgs::msg::SafetyStatus>>(safety_status_pub);

  auto estop_pub = this->create_publisher<std_msgs::msg::Bool>(
    "estop", rclcpp::QoS(1).reliable());
  estop_pub_ = std::make_shared<
    realtime_tools::RealtimePublisher<std_msgs::msg::Bool>>(estop_pub);

  // Create services
  trigger_estop_srv_ = this->create_service<dynamical_msgs::srv::TriggerEstop>(
    "trigger_estop",
    std::bind(&SafetyNode::handle_trigger_estop, this,
      std::placeholders::_1, std::placeholders::_2));

  reset_estop_srv_ = this->create_service<dynamical_msgs::srv::ResetEstop>(
    "reset_estop",
    std::bind(&SafetyNode::handle_reset_estop, this,
      std::placeholders::_1, std::placeholders::_2));

  RCLCPP_INFO(this->get_logger(), "SafetyNode configured with %zu joints", num_joints_);
  return CallbackReturn::SUCCESS;
}

CallbackReturn SafetyNode::on_activate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Activating SafetyNode...");

  // Set real-time scheduling if requested
  bool use_rt = this->get_parameter("use_realtime_scheduling").as_bool();
  if (use_rt) {
    struct sched_param param;
    param.sched_priority = 99;  // Highest priority
    if (sched_setscheduler(0, SCHED_FIFO, &param) == 0) {
      RCLCPP_INFO(this->get_logger(), "Real-time scheduling enabled (SCHED_FIFO, priority 99)");
    } else {
      RCLCPP_WARN(this->get_logger(),
        "Failed to set real-time scheduling (requires elevated permissions)");
    }

    // Lock memory to prevent page faults
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) {
      RCLCPP_INFO(this->get_logger(), "Memory locked");
    }
  }

  // Initialize timing
  last_loop_time_ = std::chrono::steady_clock::now();
  last_heartbeat_time_ = std::chrono::steady_clock::now().time_since_epoch().count();

  // Create timer for main loop
  int rate_hz = this->get_parameter("rate_hz").as_int();
  auto period = std::chrono::duration<double>(1.0 / rate_hz);
  timer_ = this->create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(period),
    std::bind(&SafetyNode::safety_loop, this));

  RCLCPP_INFO(this->get_logger(), "SafetyNode activated at %d Hz", rate_hz);
  return CallbackReturn::SUCCESS;
}

CallbackReturn SafetyNode::on_deactivate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Deactivating SafetyNode...");

  // Trigger E-stop on deactivation for safety
  trigger_estop("Node deactivated");

  // Stop timer
  timer_.reset();

  return CallbackReturn::SUCCESS;
}

CallbackReturn SafetyNode::on_cleanup(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Cleaning up SafetyNode...");

  // Reset all resources
  robot_state_sub_.reset();
  obstacles_sub_.reset();
  humans_sub_.reset();
  heartbeat_sub_.reset();
  safety_status_pub_.reset();
  estop_pub_.reset();
  trigger_estop_srv_.reset();
  reset_estop_srv_.reset();

  return CallbackReturn::SUCCESS;
}

CallbackReturn SafetyNode::on_shutdown(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(this->get_logger(), "Shutting down SafetyNode...");
  return CallbackReturn::SUCCESS;
}

void SafetyNode::safety_loop()
{
  auto loop_start = std::chrono::steady_clock::now();

  // Clear previous violations
  {
    std::lock_guard<std::mutex> lock(violations_mutex_);
    current_violations_.clear();
  }

  // Run all safety checks
  bool all_safe = true;
  all_safe &= check_joint_limits();
  all_safe &= check_velocity_limits();
  all_safe &= check_torque_limits();
  all_safe &= check_obstacle_proximity();
  all_safe &= check_human_proximity();
  all_safe &= check_watchdog();

  // Determine status
  uint8_t status = dynamical_msgs::msg::SafetyStatus::STATUS_OK;
  if (estop_active_) {
    status = dynamical_msgs::msg::SafetyStatus::STATUS_ESTOP;
  } else if (!all_safe) {
    std::lock_guard<std::mutex> lock(violations_mutex_);
    bool has_critical = false;
    for (const auto & v : current_violations_) {
      if (v.severity == dynamical_msgs::msg::SafetyViolation::SEVERITY_CRITICAL) {
        has_critical = true;
        break;
      }
    }
    if (has_critical) {
      status = dynamical_msgs::msg::SafetyStatus::STATUS_ESTOP;
      trigger_estop("Critical safety violation");
    } else {
      status = dynamical_msgs::msg::SafetyStatus::STATUS_VIOLATION;
    }
  }

  // Publish safety status (real-time safe)
  if (safety_status_pub_->trylock()) {
    auto & msg = safety_status_pub_->msg_;
    msg.header.stamp = this->now();
    msg.status = status;
    msg.estop_active = estop_active_;

    {
      std::lock_guard<std::mutex> lock(violations_mutex_);
      msg.violations = current_violations_;
    }

    msg.last_check_time_us = loop_time_us_;
    msg.missed_deadlines_count = missed_deadlines_;

    safety_status_pub_->unlockAndPublish();
  }

  // Calculate timing
  auto loop_end = std::chrono::steady_clock::now();
  loop_time_us_ = std::chrono::duration<double, std::micro>(loop_end - loop_start).count();

  // Check for missed deadline
  auto elapsed = std::chrono::duration<double, std::micro>(loop_end - last_loop_time_).count();
  if (elapsed > 1100.0) {  // More than 10% over 1ms
    missed_deadlines_++;
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "Safety loop missed deadline: %.2f us", elapsed);
  }
  last_loop_time_ = loop_start;
}

bool SafetyNode::check_joint_limits()
{
  auto state_ptr = robot_state_buffer_.readFromRT();
  if (!state_ptr || state_ptr->joint_positions.empty()) {
    return true;  // No data, assume safe
  }

  bool safe = true;
  for (size_t i = 0; i < std::min(state_ptr->joint_positions.size(), num_joints_); ++i) {
    double pos = state_ptr->joint_positions[i];
    double lower = joint_limits_lower_[i] + config_.joint_limit_margin;
    double upper = joint_limits_upper_[i] - config_.joint_limit_margin;

    if (pos < lower || pos > upper) {
      safe = false;
      dynamical_msgs::msg::SafetyViolation violation;
      violation.type = dynamical_msgs::msg::SafetyViolation::TYPE_POSITION_LIMIT;
      violation.severity = (pos < joint_limits_lower_[i] || pos > joint_limits_upper_[i])
        ? dynamical_msgs::msg::SafetyViolation::SEVERITY_CRITICAL
        : dynamical_msgs::msg::SafetyViolation::SEVERITY_WARNING;
      violation.joint_index = static_cast<int32_t>(i);
      violation.current_value = pos;
      violation.limit_value = (pos < lower) ? lower : upper;

      std::lock_guard<std::mutex> lock(violations_mutex_);
      current_violations_.push_back(violation);
    }
  }
  return safe;
}

bool SafetyNode::check_velocity_limits()
{
  auto state_ptr = robot_state_buffer_.readFromRT();
  if (!state_ptr || state_ptr->joint_velocities.empty()) {
    return true;
  }

  bool safe = true;
  double margin_factor = 1.0 - config_.velocity_limit_margin;

  for (size_t i = 0; i < std::min(state_ptr->joint_velocities.size(), num_joints_); ++i) {
    double vel = std::abs(state_ptr->joint_velocities[i]);
    double limit = velocity_limits_[i] * margin_factor;

    if (vel > limit) {
      safe = false;
      dynamical_msgs::msg::SafetyViolation violation;
      violation.type = dynamical_msgs::msg::SafetyViolation::TYPE_VELOCITY_LIMIT;
      violation.severity = (vel > velocity_limits_[i])
        ? dynamical_msgs::msg::SafetyViolation::SEVERITY_VIOLATION
        : dynamical_msgs::msg::SafetyViolation::SEVERITY_WARNING;
      violation.joint_index = static_cast<int32_t>(i);
      violation.current_value = vel;
      violation.limit_value = velocity_limits_[i];

      std::lock_guard<std::mutex> lock(violations_mutex_);
      current_violations_.push_back(violation);
    }
  }
  return safe;
}

bool SafetyNode::check_torque_limits()
{
  auto state_ptr = robot_state_buffer_.readFromRT();
  if (!state_ptr || state_ptr->joint_torques.empty()) {
    return true;
  }

  bool safe = true;
  double margin_factor = 1.0 - config_.torque_limit_margin;

  for (size_t i = 0; i < std::min(state_ptr->joint_torques.size(), num_joints_); ++i) {
    double torque = std::abs(state_ptr->joint_torques[i]);
    double limit = torque_limits_[i] * margin_factor;

    if (torque > limit) {
      safe = false;
      dynamical_msgs::msg::SafetyViolation violation;
      violation.type = dynamical_msgs::msg::SafetyViolation::TYPE_TORQUE_LIMIT;
      violation.severity = (torque > torque_limits_[i])
        ? dynamical_msgs::msg::SafetyViolation::SEVERITY_CRITICAL
        : dynamical_msgs::msg::SafetyViolation::SEVERITY_WARNING;
      violation.joint_index = static_cast<int32_t>(i);
      violation.current_value = torque;
      violation.limit_value = torque_limits_[i];

      std::lock_guard<std::mutex> lock(violations_mutex_);
      current_violations_.push_back(violation);
    }
  }
  return safe;
}

bool SafetyNode::check_obstacle_proximity()
{
  auto obstacles_ptr = obstacles_buffer_.readFromRT();
  if (!obstacles_ptr || obstacles_ptr->empty()) {
    return true;
  }

  bool safe = true;
  for (const auto & obstacle : *obstacles_ptr) {
    double distance = std::sqrt(
      obstacle.position.x * obstacle.position.x +
      obstacle.position.y * obstacle.position.y +
      obstacle.position.z * obstacle.position.z);

    if (distance < config_.min_obstacle_distance) {
      safe = false;
      dynamical_msgs::msg::SafetyViolation violation;
      violation.type = dynamical_msgs::msg::SafetyViolation::TYPE_OBSTACLE_PROXIMITY;
      violation.severity = dynamical_msgs::msg::SafetyViolation::SEVERITY_VIOLATION;
      violation.joint_index = -1;
      violation.current_value = distance;
      violation.limit_value = config_.min_obstacle_distance;

      std::lock_guard<std::mutex> lock(violations_mutex_);
      current_violations_.push_back(violation);
    }
  }
  return safe;
}

bool SafetyNode::check_human_proximity()
{
  auto humans_ptr = humans_buffer_.readFromRT();
  if (!humans_ptr || humans_ptr->empty()) {
    return true;
  }

  bool safe = true;
  for (const auto & human : *humans_ptr) {
    double distance = std::sqrt(
      human.position.x * human.position.x +
      human.position.y * human.position.y +
      human.position.z * human.position.z);

    if (distance < config_.human_safety_distance) {
      safe = false;
      dynamical_msgs::msg::SafetyViolation violation;
      violation.type = dynamical_msgs::msg::SafetyViolation::TYPE_HUMAN_PROXIMITY;
      violation.severity = dynamical_msgs::msg::SafetyViolation::SEVERITY_CRITICAL;
      violation.joint_index = -1;
      violation.current_value = distance;
      violation.limit_value = config_.human_safety_distance;

      std::lock_guard<std::mutex> lock(violations_mutex_);
      current_violations_.push_back(violation);
    }
  }
  return safe;
}

bool SafetyNode::check_watchdog()
{
  auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  auto last = last_heartbeat_time_.load();
  double elapsed_ms = static_cast<double>(now - last) / 1e6;

  if (elapsed_ms > config_.watchdog_timeout_ms) {
    dynamical_msgs::msg::SafetyViolation violation;
    violation.type = dynamical_msgs::msg::SafetyViolation::TYPE_WATCHDOG_TIMEOUT;
    violation.severity = dynamical_msgs::msg::SafetyViolation::SEVERITY_CRITICAL;
    violation.current_value = elapsed_ms;
    violation.limit_value = config_.watchdog_timeout_ms;

    std::lock_guard<std::mutex> lock(violations_mutex_);
    current_violations_.push_back(violation);
    return false;
  }
  return true;
}

void SafetyNode::trigger_estop(const std::string & reason)
{
  if (!estop_active_) {
    RCLCPP_ERROR(this->get_logger(), "E-STOP TRIGGERED: %s", reason.c_str());
    estop_active_ = true;

    // Publish E-stop
    if (estop_pub_->trylock()) {
      estop_pub_->msg_.data = true;
      estop_pub_->unlockAndPublish();
    }
  }
}

bool SafetyNode::reset_estop()
{
  std::lock_guard<std::mutex> lock(violations_mutex_);
  if (!current_violations_.empty()) {
    RCLCPP_WARN(this->get_logger(), "Cannot reset E-stop: %zu violations present",
      current_violations_.size());
    return false;
  }

  RCLCPP_INFO(this->get_logger(), "E-STOP RESET");
  estop_active_ = false;

  // Publish E-stop cleared
  if (estop_pub_->trylock()) {
    estop_pub_->msg_.data = false;
    estop_pub_->unlockAndPublish();
  }
  return true;
}

void SafetyNode::robot_state_callback(const dynamical_msgs::msg::RobotState::SharedPtr msg)
{
  robot_state_buffer_.writeFromNonRT(*msg);
}

void SafetyNode::obstacles_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
{
  obstacles_buffer_.writeFromNonRT(msg->poses);
}

void SafetyNode::humans_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
{
  humans_buffer_.writeFromNonRT(msg->poses);
}

void SafetyNode::heartbeat_callback(const std_msgs::msg::Bool::SharedPtr)
{
  last_heartbeat_time_ = std::chrono::steady_clock::now().time_since_epoch().count();
}

void SafetyNode::handle_trigger_estop(
  const std::shared_ptr<dynamical_msgs::srv::TriggerEstop::Request> request,
  std::shared_ptr<dynamical_msgs::srv::TriggerEstop::Response> response)
{
  trigger_estop(request->reason);
  response->success = true;
  response->message = "E-stop triggered";
}

void SafetyNode::handle_reset_estop(
  const std::shared_ptr<dynamical_msgs::srv::ResetEstop::Request>,
  std::shared_ptr<dynamical_msgs::srv::ResetEstop::Response> response)
{
  if (reset_estop()) {
    response->success = true;
    response->message = "E-stop reset successfully";
  } else {
    response->success = false;
    response->message = "Cannot reset E-stop due to active violations";

    std::lock_guard<std::mutex> lock(violations_mutex_);
    for (const auto & v : current_violations_) {
      response->blocking_conditions.push_back(v.message);
    }
  }
}

}  // namespace dynamical

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(dynamical::SafetyNode)
