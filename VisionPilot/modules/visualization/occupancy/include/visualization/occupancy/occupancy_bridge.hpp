#ifndef VISIONPILOT_OCCUPANCY_BRIDGE_HPP
#define VISIONPILOT_OCCUPANCY_BRIDGE_HPP

#include <common/types.hpp>
#include <models/inference.hpp>
#include <opencv2/core.hpp>

#include <visualization/occupancy/occupancy_view.hpp>

class VisualInterface;

namespace visualization {
namespace occupancy {

// Convert stock VisionPilot pipeline outputs → Occupancy Scene.
// This is the only place that knows about InferenceFrameResult / Plan.
Scene make_scene(const visionpilot::models::InferenceFrameResult& result,
                 const Plan& plan,
                 const cv::Mat& H_resized);

// make_scene() + render().
cv::Mat build_frame(const visionpilot::models::InferenceFrameResult& result,
                    const Plan& plan,
                    const cv::Mat& H_resized);

// One-line upstream hook: build panel and push to VisualInterface::set_aux_frame.
void publish(VisualInterface* visual_interface,
             const visionpilot::models::InferenceFrameResult& result,
             const Plan& plan,
             const cv::Mat& H_resized);

}  // namespace occupancy
}  // namespace visualization

#endif  // VISIONPILOT_OCCUPANCY_BRIDGE_HPP
