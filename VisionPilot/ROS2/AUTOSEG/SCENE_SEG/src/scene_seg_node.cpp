
#include "scene_seg_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>

namespace autoware_pov::AutoSeg::SceneSeg
{

SceneSegNode::SceneSegNode(const rclcpp::NodeOptions & node_options)
: Node("scene_seg_node", node_options)
{
  // Declare and get parameters
  const std::string model_path = this->declare_parameter<std::string>("model_path");
  const std::string precision = this->declare_parameter<std::string>("precision", "cpu");
  const int gpu_id = this->declare_parameter<int>("gpu_id", 0);

  // Initialize the segmentation engine
  scene_seg_ = std::make_unique<SceneSeg>(model_path, precision, gpu_id);
  
  // Setup publishers
  mask_pub_ = image_transport::create_publisher(this, "~/out/mask");
  color_mask_pub_ = image_transport::create_publisher(this, "~/out/color_mask");
  
  // Use a timer to defer subscriber creation until there's a connection
  using std::chrono_literals::operator""ms;
  timer_ = rclcpp::create_timer(this, get_clock(), 100ms, std::bind(&SceneSegNode::onConnect, this));
}

void SceneSegNode::onConnect()
{
  if (mask_pub_.getNumSubscribers() == 0 && color_mask_pub_.getNumSubscribers() == 0) {
    image_sub_.shutdown();
  } else if (!image_sub_) {
    // Define a QoS profile that prioritizes freshness over reliability
    rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
    qos_profile.reliability = RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;
    qos_profile.history = RMW_QOS_POLICY_HISTORY_KEEP_LAST;
    qos_profile.depth = 1;

    image_sub_ = image_transport::create_subscription(
      this, "~/in/image", std::bind(&SceneSegNode::onImage, this, std::placeholders::_1), "raw",
      qos_profile);
  }
}

void SceneSegNode::onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  cv_bridge::CvImagePtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (const cv_bridge::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  
  // Ultra-lightweight latency monitoring - only measure occasionally
  bool measure_latency = (++frame_count_ % LATENCY_SAMPLE_INTERVAL == 0);
  if (measure_latency) {
    inference_start_time_ = std::chrono::steady_clock::now();
  }
  
  // Run inference
  if (!scene_seg_->doInference(in_image_ptr->image)) {
    RCLCPP_WARN(this->get_logger(), "Failed to run inference");
    return;
  }
  
  // Report latency only occasionally (minimal overhead)
  if (measure_latency) {
    auto inference_end_time = std::chrono::steady_clock::now();
    auto latency_ms = std::chrono::duration<double, std::milli>(inference_end_time - inference_start_time_).count();
    RCLCPP_INFO(this->get_logger(), "Frame %zu: Inference=%.1fms (%.1f FPS)", 
                frame_count_, latency_ms, 1000.0/latency_ms);
  }
  
  // Decide whether we need to generate any masks
  const bool need_raw_mask = mask_pub_.getNumSubscribers() > 0;
  const bool need_color_mask = color_mask_pub_.getNumSubscribers() > 0;

  if (need_raw_mask || need_color_mask) {
    cv::Mat raw_mask;
    scene_seg_->getRawMask(raw_mask, in_image_ptr->image.size());

    if (need_raw_mask) {
      sensor_msgs::msg::Image::SharedPtr out_mask_msg =
        cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::MONO8, raw_mask).toImageMsg();
      mask_pub_.publish(out_mask_msg);
    }
    
    if (need_color_mask) {
      cv::Mat blended_image;
      scene_seg_->colorizeMask(raw_mask, in_image_ptr->image, blended_image);
      sensor_msgs::msg::Image::SharedPtr out_color_mask_msg =
        cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, blended_image).toImageMsg();
      color_mask_pub_.publish(out_color_mask_msg);
    }
  }
}

}  // namespace autoware_pov::AutoSeg::SceneSeg

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoware_pov::AutoSeg::SceneSeg::SceneSegNode) 