# Occupancy view (heuristic 3D BEV)

Self-contained optional module for VisionPilot. Designed so it can be copied into
[autowarefoundation/vision_pilot](https://github.com/autowarefoundation/vision_pilot)
with a **small glue patch**.

## What it does
- Separate **Occupancy** window beside the main HUD
- Foreshortened / orbitable 3D scene:
  - light-gray ground + grid (0–150 m)
  - green fused-path corridor (to 150 m)
  - AutoSpeed objects as extruded boxes (blue traffic, red CIPO, white ego)
  - distance label on the red in-lane vehicle
- Mouse: L-drag orbit, R-drag pan, wheel zoom, `R` reset

## Layout
```
occupancy/
  CMakeLists.txt                 → library vp_occupancy
  include/visualization/occupancy/
    occupancy_view.hpp           → Scene + render + mouse/key (no ProductionView)
    occupancy_bridge.hpp         → make_scene / build_frame / publish
  src/
    occupancy_view.cpp
    occupancy_bridge.cpp
  README.md
```

## Build switch
In `modules/visualization/CMakeLists.txt`:

```cmake
option(VISIONPILOT_ENABLE_OCCUPANCY "Build heuristic Occupancy BEV module" ON)
```

Disable with: `-DVISIONPILOT_ENABLE_OCCUPANCY=OFF`

## Upstream glue (when enabled)
Only one call site in stock HUD code:

```cpp
#if defined(VISIONPILOT_ENABLE_OCCUPANCY)
#include <visualization/occupancy/occupancy_bridge.hpp>
// inside Visualization::build_frame(...):
occupancy::publish(visual_interface.get(), result, plan, H_resized);
#endif
```

Plus generic display support (reusable, not Occupancy-specific):
- `VisualInterface::set_aux_frame()` default no-op
- `LocalDisplay` shows a second window when aux is non-empty

## Notes
- Heuristic geometry from detections / path / lane — **not** a neural occupancy network
- Module depends on `models` + `common` + OpenCV only (not on HUD internals)
