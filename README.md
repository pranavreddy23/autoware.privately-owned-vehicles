
# AutoSeg

AutoSeg is an AI Foundation Model which provides real-time visual scene perception for autonomous vehicles. It utilizes a single neural network backbone to extract diverse image features, a set of context blocks which focus the network's attention on key visual elements within input images, and multiple segmentation and classification heads which provide useful perceptual outputs for autonomous decision making.

AutoSeg has been hardened by training on a diverse set of real-world image data collected from various countries, across different road types and weather conditions.

By following an ensemble-of-experts approach, AutoSeg is able to learn generalizble features that are adaptable to out-of-domain scenarios and can facilitate multiple downstream perceptual tasks such as semantic object segmentation, lane perception, and even end-to-end autonomous driving. Furthermore, each neural expert can be independently refined and fine-tuned with additional data allowing for a richer representation of edge-case scenarios which are challenging to capture in a single predictor model.

The current AutoSeg release comprises 5 perceptual tasks performed by different sub-network experts, these include: CoarseSeg, ObjectSeg, RoadWorksSeg, LanesSeg, and PathSeg.

![Autoseg Network Diagram](Diagrams/AutoSeg.jpg)
