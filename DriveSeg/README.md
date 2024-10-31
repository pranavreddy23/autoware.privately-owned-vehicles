## DriveSeg
Self-driving cars require an understanding of important edge case scenarios that can occur in driving scenes. DriveSeg is a neural network which addresses this challenge by processesing raw images and segmenting all safety critical objects in the driving scene including traffic cones, construction barriers, emergency vehicles, construction crew, police officers and other such safety critical objects.

During training, DriveSeg estimates three semantic classes

- `Safety Critical Objects`
- `Background Elements`
- `Drivable Road Surface`

However, during inference, we only use the outputs from the **`Safety Critical Objects`** class.

