# NEST implementation of the LAMINART model of visual cortex.
# NEST simulation time is in milliseconds

import sys, numpy, random
import setInput
import pyNN.nest as sim
# from pyNN.utility import get_simulator
# sim, options = get_simulator()
from LaminartWithSegmentationPyNN import buildNetworkAndConnections


####################################
### Define parameters down here! ###
####################################

# How time goes
dt = 1.0                    # (ms) time step for network updates
stepDuration = 50.0         # (ms) time step for visual input and segmentation signal updates
synDelay = 2.0              # (ms) ??? check, car avec toutes les layers ca peut faire de la merde
simTime = 1000.0            # (ms)
nTimeSteps = numpy.int(simTime/stepDuration)
sim.setup(timestep=dt)

# General parameters
fileName = "Test"          # would be removed if it is in the NRP
input, ImageNumPixelRows, ImageNumPixelColumns = setInput.readAndCropBMP(fileName, onlyZerosAndOnes=0)
weightScale = 1.0		   # general weight for all connections between neurons
normalCellParams = {
    'i_offset'   : 0.0,    # (nA)
    'tau_m'      : 20.0,   # (ms)
    'tau_syn_E'  : 2.0,    # (ms)
    'tau_syn_I'  : 4.0,    # (ms)
    'tau_refrac' : 2.0,    # (ms)
    'v_rest'     : -60.0,  # (mV)
    'v_reset'    : -70.0,  # (mV)
    'v_thresh'   : -50.0,  # (mV)
    'cm'         : 0.5}    # (nF)
tonicCellParams = {
    'i_offset'   : 1.0,    # (nA)
    'tau_m'      : 20.0,   # (ms)
    'tau_syn_E'  : 2.0,    # (ms)
    'tau_syn_I'  : 4.0,    # (ms)
    'tau_refrac' : 2.0,    # (ms)
    'v_rest'     : -60.0,  # (mV)
    'v_reset'    : -70.0,  # (mV)
    'v_thresh'   : -50.0,  # (mV)
    'cm'         : 0.5}    # (nF)
normalCellType = sim.IF_curr_alpha(**normalCellParams)  # any neuron in the network
tonicCellType = sim.IF_curr_alpha(**tonicCellParams)    # tonically active interneurons (inter3 in segmentation network)

# Orientation filters parameters
numOrientations = 2		   # number of orientations (2, 4 or 8 ; 8 is experimental)
oriFilterSize = 4          # better as an even number (filtering from LGN to V1) ; 4 is original
V1PoolSize = 3             # better as an odd number (pooling in V1) ; 3 is original
V2PoolSize = 7             # better as an odd number (pooling in V2) ; 7 is original

# Segmentation parameters
numSegmentationLayers = 3        # number of segmentation layers (usual is 3, minimum is 1)
useSurfaceSegmentation = 0       # use segmentation that flows across closed shapes
useBoundarySegmentation = 1      # use segmentation that flows along connected boundaries
useSDPropToDist = 0			     # if 1, segmentationTargetLocationSD ~ dist(segmentationTargetLocation;fix.point)
minSD = 3                        # minimum segmentationTargetLocationSD, (e.g. sending segmentation signals around fovea)
rateSD = 0.1                     # how much segmentationTargetLocationSD grows with excentricity (pixels SD per pixel excentricity)
segmentationTargetLocationSD = 8 # standard deviation of where location actually ends up (originally 8) ; in pixels
segmentationSignalSize = 20	     # even number ; circle diameter where a segmentation signal is triggered ; in pixels
if numSegmentationLayers == 1:
    useSurfaceSegmentation = 0
    useBoundarySegmentation = 0

# Build the whole network and take the layers that have to be updated during the simulation
network = buildNetworkAndConnections(sim, ImageNumPixelRows, ImageNumPixelColumns, numOrientations, oriFilterSize, V1PoolSize, V2PoolSize,
                                     normalCellType, tonicCellType, weightScale, numSegmentationLayers, useBoundarySegmentation, useSurfaceSegmentation)
LGNBrightInput = network.get_population("LGNBrightInput")
LGNDarkInput = network.get_population("LGNDarkInput")
if useSurfaceSegmentation:
    SurfaceSegmentationOffSignal = network.get_population("SurfaceSegmentationOffSignal")
    SurfaceSegmentationOnSignal = network.get_population("SurfaceSegmentationOnSignal")
if useBoundarySegmentation:
    BoundarySegmentationOffSignal = network.get_population("BoundarySegmentationOffSignal")
    BoundarySegmentationOnSignal  = network.get_population("BoundarySegmentationOnSignal")


########################################################################################
### Network is defined, now set up stimulus, segmentation signal and run everything! ###
########################################################################################

for timeStep in range(nTimeSteps):

    # Set LGN input, using the image input
    brightPulse = sim.DCSource(start=timeStep*stepDuration, stop=(timeStep+1)*stepDuration) # bright input
    darkPulse   = sim.DCSource(start=timeStep*stepDuration, stop=(timeStep+1)*stepDuration) # dark input
    # input = getInputFromNRPRetinaModel(timeStep) # something like this will happen in the NRP and input would be a 2D numpy array
    for i in range(ImageNumPixelRows):
        for j in range(ImageNumPixelColumns):
            brightPulse.set_parameters(amplitude=input[i,j])
            darkPulse.set_parameters(amplitude=(254.0-input[i,j]))
            LGNBrightInput[i*ImageNumPixelColumns + j].inject(brightPulse)
            LGNDarkInput[i*ImageNumPixelColumns + j].inject(darkPulse)

    # Segmentation signals
    segmentationInput = sim.DCSource(amplitude=1000.0, start=timeStep*stepDuration, stop=(timeStep+1)*stepDuration)
    for n in range(numSegmentationLayers-1):

        # Pick a segmentation signal location
        segmentationTargetLocationX, segmentationTargetLocationY = [-5,6] # TO BE CHOSEN ACCORDING TO SALIENCY MAP AND/OR DEFINED BY USER
        if useSDPropToDist:
            segmentationTargetLocationSD = minSD + rateSD*numpy.sqrt((segmentationTargetLocationX)**2 + segmentationTargetLocationY**2)
        segmentLocationX = int(round(random.gauss(ImageNumPixelColumns/2 - segmentationTargetLocationX, segmentationTargetLocationSD)))
        segmentLocationY = int(round(random.gauss(ImageNumPixelRows/2    - segmentationTargetLocationY, segmentationTargetLocationSD)))

        # Define surface segmentation signals (gives local DC inputs to surface and boundary segmentation networks)
        if useSurfaceSegmentation==1:
            for i in range(0, ImageNumPixelRows):           # Rows
                for j in range(0, ImageNumPixelColumns):    # Columns
                    # Off signals are at borders of image (will flow in to other locations unless stopped by boundaries)
                    if i==0 or i==(ImageNumPixelRows-1) or j==0 or j==(ImageNumPixelColumns-1):
                        for h in range(0, numSegmentationLayers-1):  # Segmentation layers (not including baseline)
                            SurfaceSegmentationOffSignal[h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j].inject(segmentationInput)
                    distance = numpy.sqrt(numpy.power(segmentLocationX-j, 2.0) + numpy.power(segmentLocationY-i, 2.0))
                    if distance < segmentationSignalSize:   # [0][i][j]
                        SurfaceSegmentationOnSignal[0*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j].inject(segmentationInput)

        # Define boundary segmentation signals
        if useBoundarySegmentation==1:
            for i in range(0, ImageNumPixelRows):           # Rows
                for j in range(0, ImageNumPixelColumns):    # Columns
                    distance = numpy.sqrt(numpy.power(segmentLocationX-j, 2.0) + numpy.power(segmentLocationY-i, 2.0))
                    if distance < segmentationSignalSize:   # [0][i][j]
                        BoundarySegmentationOnSignal[0*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j].inject(segmentationInput)

    # Actual run of the network, using the input and the segmentation signals
    sim.run(stepDuration)

# Close the simulation
sim.end()