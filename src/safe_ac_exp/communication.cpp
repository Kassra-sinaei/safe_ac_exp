#ifndef ROBOT_CONTROL_KF_ROBOT_CONTROLLER_HPP
#define ROBOT_CONTROL_KF_ROBOT_CONTROLLER_HPP

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "ros2_unitree_legged_msgs/msg/high_cmd.hpp"
#include "ros2_unitree_legged_msgs/msg/high_state.hpp"
#include "unitree_legged_sdk/unitree_legged_sdk.h"

class RobotController : public rclcpp::Node
{
public:
    RobotController() : Node("robot_controller") {
        subscription_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, std::bind(&RobotController::twistCallback, this, std::placeholders::_1));
        high_command_publisher_ = this->create_publisher<ros2_unitree_legged_msgs::msg::HighCmd>(
            "/high_cmd", 10);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(5), std::bind(&RobotController::timerCallback, this));
        
        // preallocate high_cmd_ros_ message
        high_cmd_ros_ = ros2_unitree_legged_msgs::msg::HighCmd();
        high_cmd_ros_.head[0] = 0xFE;
        high_cmd_ros_.head[1] = 0xEF;
        high_cmd_ros_.level_flag = UNITREE_LEGGED_SDK::HIGHLEVEL;
        high_cmd_ros_.mode = 2;
        high_cmd_ros_.gait_type = 1; 
        high_cmd_ros_.speed_level = 1; 
        high_cmd_ros_.foot_raise_height = 0.08;
        high_cmd_ros_.body_height = 0.30;
        high_cmd_ros_.velocity[0] = 0.0;
        high_cmd_ros_.velocity[1] = 0.0;
        high_cmd_ros_.yaw_speed = 0.0;
        high_cmd_ros_.reserve = 0;


        RCLCPP_INFO(this->get_logger(), "Robot Controller initialized with high-level command publisher");
    }

private:
    void twistCallback(const geometry_msgs::msg::Twist::SharedPtr msg){
        // Omni-directional robot command mapping
        high_cmd_ros_.velocity[0] = msg->linear.x;   // Forward/backward velocity
        high_cmd_ros_.velocity[1] = msg->linear.y;   // Left/right velocity 
        high_cmd_ros_.yaw_speed = msg->angular.z;    // Yaw rate (turning)
    }
    void timerCallback(){
        // Publish the most recent commands
        high_command_publisher_->publish(high_cmd_ros_);
    }

    ros2_unitree_legged_msgs::msg::HighCmd high_cmd_ros_;
    sensor_msgs::msg::Imu imu_msg_;

    rclcpp::TimerBase::SharedPtr timer_;

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr subscription_;
    rclcpp::Publisher<ros2_unitree_legged_msgs::msg::HighCmd>::SharedPtr high_command_publisher_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RobotController>());
    rclcpp::shutdown();
    return 0;
}

#endif // ROBOT_CONTROL_KF_ROBOT_CONTROLLER_HPP