## CoarseSeg
The CoarseSeg Neural Expert performs Semantic Scene Segmentation of Stuff Categories. It is aims to learn scene level feature representations that generalize across object types. For example, rather than explicitly learning features to recognise cars from buses, CoarseSeg is able to recognise high level features that can distinguish any movalbe foreground object from the static background, road and sky. This helps provide an autonomous vehicle with a core safety layer since CoarseSeg can comprehend strange presentations of known objects and previously unseen object types, helping to address 'long-tail' edge cases which plauge object-level detectors.

Semantic Classes

- All Movable Foreground Objects
- All Static Background Elements
- Drivable Road Surface
- Sky

![CoarseSeg Network Diagram](../Diagrams/CoarseSeg.jpg)