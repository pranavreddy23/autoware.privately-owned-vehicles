Contains scripts to process open datasets and create semantic masks in a unified labelling scheme according to the DriveSeg neural task specification. To process each open dataset, the file structure and heirarchy of published open datasets was simplified manually such that each open dataset had a two top level folders, a first folder with all images and a second folder with ground truth labels.

Open semantic segmentation datasets contain various labelling methodologies and semantic classes. The scripts in create_masks parse data and create semantic colormaps in a single unified semantic format.

Colormap values for unified semantic classes created from training data are as follows:

| DriveSeg Semantic Class             | DriveSeg RGB Label                             |
| ----------------- | ------------------------------------------------------------------ |
| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
| Safety Critical Objects | ![#FF7D00](https://via.placeholder.com/10/FF7D00?text=+) rgb(255, 125, 0) |
| Road | ![#00FFDC](https://via.placeholder.com/10/00FFDC?text=+) rgb(0, 255, 220) |

#### The open datasets used in DriveSeg include:
- [ROADWork](https://www.cs.cmu.edu/~ILIM/roadwork_dataset/)
- [nuImages](https://www.nuscenes.org/nuimages)
