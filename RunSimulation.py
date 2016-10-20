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
dt = 1.0                    # (ms)
synDelay = 2.0              # (ms) ??? check, car avec toutes les layers ca peut faire de la merde
stepDuration = 50.0         # (ms) time step for stimulus and segmentation signal updates
simTime = 1000.0            # (ms)
nTimeSteps = numpy.int(simTime/stepDuration)
sim.setup(timestep=dt, min_delay=max(dt,0.1), max_delay=synDelay)

# General parameters
fileName = "Test"           # would be removed if it is in the NRP
input, ImageNumPixelRows, ImageNumPixelColumns = setInput.readAndCropBMP(fileName, onlyZerosAndOnes=0)
numOrientations = 2			# number of orientations (2, 4 or 8 ; 8 is experimental)
weightScale = 1.0			# general weight for all connections between neurons
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

# Segmentation parameters
useSDPropToDist = 0					# if 1, segmentationTargetLocationSD ~ dist(segmentationTargetLocation;fix.point)
distanceToTarget = 70				# distance between the target and the fixation point ; in pixels (TO REFINE)
startSegmentationSignal = 100		# in milliseconds (originally 100)
segmentationTargetLocationSD = 8	# standard deviation of where location actually ends up (originally 8) ; in pixels
segmentationSignalSize = 20			# even number ; circle diameter, where a segmentation signal is triggered ; in pixels

# Orientation filters parameters
oriFilterSize = 4  # better as an even number (filtering from LGN to V1) ; 4 is original
V1PoolSize = 3     # better as an odd number (pooling in V1) ; 3 is original
V2PoolSize = 7     # better as an odd number (pooling in V2) ; 7 is original

# Segmentation layers parameters ; one of these is the baseline layer (commonly use 3) minimum is 1
numSegmentationLayers = 3
useSurfaceSegmentation = 0
useBoundarySegmentation = 1
if numSegmentationLayers == 1:
    useSurfaceSegmentation = 0
    useBoundarySegmentation = 0

# Build the whole network
network = buildNetworkAndConnections(sim, ImageNumPixelRows, ImageNumPixelColumns, numOrientations, oriFilterSize, V1PoolSize, V2PoolSize,
                                     normalCellType, tonicCellType, weightScale, numSegmentationLayers, useBoundarySegmentation, useSurfaceSegmentation)


########################################################################################
### Network is defined, now set up stimulus, segmentation signal and run everything! ###
########################################################################################

for timeStep in range(nTimeSteps):

    # Set LGN input, using the image input
    brightPulse = sim.DCSource(start=timeStep*stepDuration, stop=(timeStep+1)*stepDuration) # bright input
    darkPulse   = sim.DCSource(start=timeStep*stepDuration, stop=(timeStep+1)*stepDuration) # dark input
    # input = getInputFromNRPRetinaModel(timeStep) # something like this will happen in the NRP and input would be a 2D numpy array
    LGNBrightInput = network.get_population("LGNBrightInput")
    LGNDarkInput   = network.get_population("LGNDarkInput")
    for i in range(ImageNumPixelRows):
        for j in range(ImageNumPixelColumns):
            brightPulse.set_parameters(amplitude=input[i,j])
            darkPulse.set_parameters(amplitude=(254.0-input[i,j]))
            LGNBrightInput[i*ImageNumPixelColumns + j].inject(brightPulse)
            LGNDarkInput[i*ImageNumPixelColumns + j].inject(darkPulse)

    # Pick central locations of segmentation signals from a gaussian distribution
    segmentationTargetLocationX, segmentationTargetLocationY = [20,0] # TO BE CHOSEN ACCORDING TO SALIENCY MAP AND/OR DEFINED BY USER
    segmentationInput = sim.DCSource(amplitude=1000.0, start=timeStep*stepDuration, stop=(timeStep+1)*stepDuration)
    if useSDPropToDist:
        segmentationTargetLocationSDRight = segmentationTargetLocationSD * numpy.sqrt((distanceToTarget + segmentationTargetLocationX)**2 + segmentationTargetLocationY**2)/distanceToTarget
        segmentationTargetLocationSDLeft  = segmentationTargetLocationSD * numpy.sqrt((distanceToTarget - segmentationTargetLocationX)**2 + segmentationTargetLocationY**2)/distanceToTarget
    else:
        segmentationTargetLocationSDRight = segmentationTargetLocationSDLeft = segmentationTargetLocationSD
    adjustLocationXLeft  = int(round(random.gauss(segmentationTargetLocationX, segmentationTargetLocationSDLeft)))
    adjustLocationYLeft  = int(round(random.gauss(segmentationTargetLocationY, segmentationTargetLocationSDLeft)))
    adjustLocationXRight = int(round(random.gauss(segmentationTargetLocationX, segmentationTargetLocationSDRight)))
    adjustLocationYRight = int(round(random.gauss(segmentationTargetLocationY, segmentationTargetLocationSDRight)))

    # Define surface segmentation signals (gives local DC inputs to surface and boundary segmentation networks)
    if useSurfaceSegmentation==1:
        # Left side
        segmentLocationX = ImageNumPixelColumns/2 - (adjustLocationXLeft-1)
        segmentLocationY = ImageNumPixelRows/2 - adjustLocationYLeft
        sys.stdout.write('\Left center surface segment signal = '+str(segmentLocationX)+", "+str(segmentLocationY))
        SurfaceSegmentationOffSignal = network.get_population("SurfaceSegmentationOffSignal")
        SurfaceSegmentationOnSignal  = network.get_population("SurfaceSegmentationOnSignal")
        target = []
        target2 = []
        for i in range(0, ImageNumPixelRows):           # Rows
            for j in range(0, ImageNumPixelColumns):    # Columns
                # Off signals are at borders of image (will flow in to other locations unless stopped by boundaries)
                if i==0 or i==(ImageNumPixelRows-1) or j==0 or j==(ImageNumPixelColumns-1):
                    for h in range(0, numSegmentationLayers-1):  # Segmentation layers (not including baseline)
                        SurfaceSegmentationOffSignal[h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j].inject(segmentationInput)
                distance = numpy.sqrt(numpy.power(segmentLocationX-j, 2.0) + numpy.power(segmentLocationY-i, 2.0))
                if distance < segmentationSignalSize:   # [0][i][j]
                    SurfaceSegmentationOnSignal[0*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j].inject(segmentationInput)

        if numSegmentationLayers>2:
            # Right side
            segmentLocationX = ImageNumPixelColumns/2 + adjustLocationXRight
            segmentLocationY = ImageNumPixelRows/2 - adjustLocationYRight
            sys.stdout.write('\Right center surface segment signal = '+str(segmentLocationX)+", "+str(segmentLocationY))
            for i in range(0, ImageNumPixelRows):           # Rows
                for j in range(0, ImageNumPixelColumns):    # Columns
                    distance = numpy.sqrt(numpy.power(segmentLocationX-j, 2.0) + numpy.power(segmentLocationY-i, 2.0))
                    if distance < segmentationSignalSize:   # [1][i][j]
                        SurfaceSegmentationOnSignal[1*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j].inject(segmentationInput)

    # Define boundary segmentation signals
    if useBoundarySegmentation==1:
        # Left side
        segmentLocationX = ImageNumPixelColumns/2 - (adjustLocationXLeft-1)
        segmentLocationY = ImageNumPixelRows/2 - adjustLocationYLeft
        sys.stdout.write('\nLeft center boundary segment signal = '+str(segmentLocationX)+', '+str(segmentLocationY))
        BoundarySegmentationOffSignal = network.get_population("BoundarySegmentationOffSignal")
        BoundarySegmentationOnSignal  = network.get_population("BoundarySegmentationOnSignal")
        for i in range(0, ImageNumPixelRows):           # Rows
            for j in range(0, ImageNumPixelColumns):    # Columns
                distance = numpy.sqrt(numpy.power(segmentLocationX-j, 2.0) + numpy.power(segmentLocationY-i, 2.0))
                if distance < segmentationSignalSize:   # [0][i][j]
                    BoundarySegmentationOnSignal[0*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j].inject(segmentationInput)

        if numSegmentationLayers>2:
            # Right side
            segmentLocationX = ImageNumPixelColumns/2 + adjustLocationXRight
            segmentLocationY = ImageNumPixelRows/2 - adjustLocationYRight
            sys.stdout.write('\nRight center boundary segment signal = '+str(segmentLocationX)+', '+str(segmentLocationY))
            target = []
            for i in range(0, ImageNumPixelRows):           # Rows
                for j in range(0, ImageNumPixelColumns):    # Columns
                    distance = numpy.sqrt(numpy.power(segmentLocationX - j, 2.0) + numpy.power(segmentLocationY - i, 2.0))
                    if distance <segmentationSignalSize:    # [1][i][j]
                        BoundarySegmentationOnSignal[1*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j].inject(segmentationInput)

    # Actual run of the network, using the input and the segmentation signals
    sim.run(stepDuration)

# Close the simulation
sim.end()