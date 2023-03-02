
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>

void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& twistMsg)
{
  std::string command;

  // Check if angular velocity around the z-axis is positive or negative
  if (twistMsg->angular.z > 0)
    command = "D"; // Turn right in place
  else if (twistMsg->angular.z < 0)
    command = "A"; // Turn left in place
  else
  {
    // Find the maximum linear velocity in each direction
    double maxLinearX = std::max(fabs(twistMsg->linear.x), fabs(twistMsg->linear.y));
    
    // Convert the maximum linear velocity to a movement command
    if (maxLinearX > 0)
      command = (twistMsg->linear.x > 0) ? "W" : "S"; // Move forward or backward
    else if (twistMsg->linear.y > 0)
      command = "Q"; // Move left while maintaining the same orientation
    else if (twistMsg->linear.y < 0)
      command = "E"; // Move right while maintaining the same orientation
    else
      command = "B"; // Stop
  }

  // Output the movement command to std::cout
  std::cout << command << std::endl;
}

int main(int argc, char** argv)
{
  // Initialize ROS node
  ros::init(argc, argv, "cmd_vel_listener");

  // Create a ROS node handle
  ros::NodeHandle nodeHandle;

  // Subscribe to the cmd_vel topic with the cmdVelCallback function
  ros::Subscriber cmdVelSubscriber = nodeHandle.subscribe<geometry_msgs::Twist>("cmd_vel", 1, cmdVelCallback);

  // Spin the ROS node
  ros::spin();

  return 0;
}