# Merging Occupancy into upstream VisionPilot

Copy this entire `occupancy/` directory into:

```text
VisionPilot/modules/visualization/occupancy/
```

Then apply these small patches to stock files.

## 1. `modules/visualization/CMakeLists.txt`

```cmake
option(VISIONPILOT_ENABLE_OCCUPANCY "Build heuristic Occupancy BEV module" OFF)

# after add_library(visualization ...):
if (VISIONPILOT_ENABLE_OCCUPANCY)
    add_subdirectory(occupancy)
    target_link_libraries(visualization PUBLIC vp_occupancy)
    target_compile_definitions(visualization PUBLIC VISIONPILOT_ENABLE_OCCUPANCY=1)
endif ()
```

Do **not** add `occupancy_view.cpp` to the `visualization` target sources — it builds as `vp_occupancy`.

## 2. `visual_interface.hpp` (generic, 2 lines)

```cpp
virtual bool render_frame(const cv::Mat& display_frame) = 0;
// Optional second view (e.g. occupancy BEV). Default: ignore.
virtual void set_aux_frame(const cv::Mat& /*aux*/) {}
virtual bool stop() = 0;
```

## 3. `visualization.cpp` — single hook

```cpp
#if defined(VISIONPILOT_ENABLE_OCCUPANCY)
#include <visualization/occupancy/occupancy_bridge.hpp>
#endif

// inside Visualization::build_frame(...), after HUD render:
#if defined(VISIONPILOT_ENABLE_OCCUPANCY)
    occupancy::publish(visual_interface.get(), result, plan, H_resized);
#endif
```

## 4. `LocalDisplay` — show aux window

Implement `set_aux_frame()` and, when aux is non-empty, `imshow("Occupancy", ...)`.
When `VISIONPILOT_ENABLE_OCCUPANCY` is defined, wire mouse/key to
`occupancy::on_mouse` / `occupancy::on_key`.

## What NOT to change for Occupancy

- `onnx_engine.cpp` (GPU tuning — separate)
- `file_interface.cpp` (loop wrap — separate)
- `image_preprocessor.cpp` (parallel warp — separate)
- `ProductionView` fields (Occupancy uses its own `Scene`)

## Verify

```bash
cmake -DVISIONPILOT_ENABLE_OCCUPANCY=ON  ...
cmake -DVISIONPILOT_ENABLE_OCCUPANCY=OFF ...   # stock HUD only
```
