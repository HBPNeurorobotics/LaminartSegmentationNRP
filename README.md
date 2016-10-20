# LaminartSegmentationNRP

Implementation of a cortical model for visual grouping and segmentation
in PyNN. Meant to be implemented in the Neurobotics Platform of the
Human Brain Project.

Change any parameter you want in RunSimulation.py and run it as a
python script. It will call LaminartWithSegmentationPyNN.py to
create the network and the connections, and then run the simulation,
updating the input and regularly sending segmentation signals.