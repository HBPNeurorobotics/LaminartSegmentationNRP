# NEST implementation of the LAMINART model of visual cortex.
# NEST simulation time is in milliseconds

import sys, numpy, random, matplotlib.pyplot as plt, setInput, pyNN.nest as sim
from LaminartWithSegmentationPyNN import buildNetworkAndConnections
from images2gif import writeGif
# from pyNN.utility import get_simulator
# sim, options = get_simulator()

####################################
### Define parameters down here! ###
####################################

# How time goes
dt = 1.0                   # (ms) time step for network updates
stepDuration = 50.0        # (ms) time step for visual input and segmentation signal updates
synDelay = 2.0             # (ms) ??? check, car avec toutes les layers ca peut faire de la merde
simTime = 1000.0           # (ms)
nTimeSteps = numpy.int(simTime/stepDuration)
sim.setup(timestep=dt, min_delay=1.0, max_delay=10.0)

# General parameters
fileName = "Test"          # would be removed if it is in the NRP
input, ImageNumPixelRows, ImageNumPixelColumns = setInput.readAndCropBMP(fileName, onlyZerosAndOnes=0)
weightScale = 10.0		   # general weight for all connections between neurons
normalCellParams = {
    'i_offset'   : 0.0,    # (nA)
    'tau_m'      : 10.0,   # (ms)
    'tau_syn_E'  : 2.0,    # (ms)
    'tau_syn_I'  : 2.0,    # (ms)
    'tau_refrac' : 2.0,    # (ms)
    'v_rest'     : -70.0,  # (mV)
    'v_reset'    : -70.0,  # (mV)
    'v_thresh'   : -55.0,  # (mV)
    'cm'         : 0.25}   # (nF)
tonicCellParams  = {
    'i_offset'   : 1.0,    # (nA)
    'tau_m'      : 10.0,   # (ms)
    'tau_syn_E'  : 2.0,    # (ms)
    'tau_syn_I'  : 2.0,    # (ms)
    'tau_refrac' : 2.0,    # (ms)
    'v_rest'     : -70.0,  # (mV)
    'v_reset'    : -70.0,  # (mV)
    'v_thresh'   : -55.0,  # (mV)
    'cm'         : 0.25}   # (nF)
normalCellType = sim.IF_curr_alpha(**normalCellParams)  # any neuron in the network
tonicCellType = sim.IF_curr_alpha(**tonicCellParams)    # tonically active interneurons (inter3 in segmentation network)
connections = {

    # Input and LGN
    'brightInputToLGN'        :    700.0,
    'darkInputToLGN'          :    700.0,
    'LGN_ToV1Excite'          :    400.0,
    'LGN_ToV4Excite'          :    280.0,

    # V1 layers
    'V1_6To4Excite'           :      1.0,
    'V1_6To4Inhib'            :     -1.0,
    'V1_23To6Excite'          :    100.0,
    'V1_ComplexExcite'        :    500.0,
    'V1_ComplexInhib'         :   -500.0,
    'V1_FeedbackExcite'       :    500.0,
    'V1_NegFeedbackInhib'     :  -1500.0,
    'V1_InterInhib'           :  -1500.0,
    'V1_CrossInhib'           :  -1000.0,
    'V1_EndCutExcite'         :   1500.0,
    'V1_ToV2Excite'           :  10000.0,

    # V2 layers
    'V2_6To4Excite'           :      1.0,
    'V2_6To4Inhib'            :    -20.0,
    'V2_23To6Excite'          :    100.0,
    'V2_ComplexExcite'        :    500.0,
    'V2_ComplexInhib'         :  -1000.0,
    'V2_ComplexInhib2'        :   -100.0,
    'V2_OrientInhib'          :  -1200.0,
    'V2_FeedbackExcite'       :    500.0,
    'V2_NegFeedbackInhib'     :   -800.0,
    'V2_InterInhib'           :  -1500.0,
    'V2_BoundaryInhib'        :  -5000.0,
    'V2_SegmentInhib'         : -20000.0,

    # V4 layers
    'V4_BrightnessExcite'     :   2000.0,
    'V4_BrightnessInhib'      :  -2000.0,
    'V4_BetweenColorsInhib'   :  -5000.0,

    # Surface segmentation layers
    'S_SegmentSignalExcite'   :      1.0,
    'S_SegmentSpreadExcite'   :   1000.0,
    'S_SegmentInterInhib'     :   -200.0,
    'S_SegmentOnOffInhib'     :  -5000.0,

    # Boundary segmentation layers
    'B_SegmentSignalExcite'   :      0.5,
    'B_SegmentInterInhib'     :   -200.0,
    'B_SegmentOnOffInhib'     :  -5000.0,
    'B_SegmentSpreadExcite'   :   2000.0,
    'B_SegmentTonicInhib'     : -20000.0,
    'B_SegmentOpenFlowInhib' : -150.0}

# Orientation filters parameters
numOrientations = 2                       # number of orientations   (2, 4 or 8 ; 8 is experimental)
oriFilterSize = 4                         # better as an even number (how V1 pools from LGN) ; 4 is original
V1PoolSize = 3                            # better as an odd  number (pooling in V1)         ; 3 is original
V2PoolSize = 7                            # better as an odd  number (pooling in V2)         ; 7 is original
numPixelRows    = ImageNumPixelRows+1     # number of rows for the oriented grid (in between un-oriented pixels)
numPixelColumns = ImageNumPixelColumns+1  # same for columns

# Segmentation parameters
numSegmentationLayers = 3                 # number of segmentation layers (usual is 3, minimum is 1)
useSurfaceSegmentation = 0                # use segmentation that flows across closed shapes
useBoundarySegmentation = 1               # use segmentation that flows along connected boundaries
useSDPropToDist = 0                       # if 1, segmentationTargetLocationSD ~ dist(segmentationTargetLocation;fix.point)
minSD = 3                                 # minimum segmentationTargetLocationSD, (e.g. sending segmentation signals around fovea)
rateSD = 0.1                              # how much segmentationTargetLocationSD grows with excentricity (pixels SD per pixel excentricity)
segmentationTargetLocationSD = 8          # standard deviation of where location actually ends up (originally 8) ; in pixels
segmentationSignalSize = 20               # even number ; circle diameter where a segmentation signal is triggered ; in pixels
if numSegmentationLayers == 1 or fileName in ["Test", "Test2"]:
    numSegmentationLayers = 1             # for test cases
    useSurfaceSegmentation = 0
    useBoundarySegmentation = 0

# Scale the weights, and then sends everything to build the whole network and take the layers that have to be updated during the simulation
for key, value in connections.items():
    connections[key] = value*weightScale
sim, network = buildNetworkAndConnections(sim, ImageNumPixelRows, ImageNumPixelColumns, numOrientations, oriFilterSize, V1PoolSize, V2PoolSize, connections,
                                          normalCellType, tonicCellType, numSegmentationLayers, useBoundarySegmentation, useSurfaceSegmentation)

# Take the neuron layers that need to be updated online or that we want to take records of
LGNBrightInput = network.get_population("LGNBrightInput")
LGNDarkInput   = network.get_population("LGNDarkInput")
LGNBright      = network.get_population("LGNBright")
V1             = network.get_population("V1Layer6P1")
V2             = network.get_population("V2Layer23")
LGNBright.record("spikes")
V1.record("spikes")
V2.record("spikes")
if useSurfaceSegmentation:
    SurfaceSegmentationOffSignal  = network.get_population("SurfaceSegmentationOffSignal")
    SurfaceSegmentationOnSignal   = network.get_population("SurfaceSegmentationOnSignal")
if useBoundarySegmentation:
    BoundarySegmentationOffSignal = network.get_population("BoundarySegmentationOffSignal")
    BoundarySegmentationOnSignal  = network.get_population("BoundarySegmentationOnSignal")

########################################################################################
### Network is defined, now set up stimulus, segmentation signal and run everything! ###
########################################################################################

# Stuff to plot activity of LGN, V1, V2, etc. layers
cumplotDensityLGNBright = [[0 for j in range(ImageNumPixelColumns)] for i in range(ImageNumPixelRows)]
cumplotDensityOrientationV2 = [[[[0 for j in range(numPixelColumns)] for i in range(numPixelRows)] for h in range(numSegmentationLayers)] for k in range(numOrientations)] # TO DEFINE OUTSIDE TRIALS (si jamais)
cumplotDensityOrientationV1 = [[[0 for j in range(numPixelColumns)] for i in range(numPixelRows)] for k in range(numOrientations)]
newMaxV1 = 0
newMaxV2 = 0
newMaxLGNBright = 0
outImagesV1 = []
outImagesV2 = [[] for i in range(numSegmentationLayers)]
outImagesLGNBright = []

# Set input, segmentation signals, run simulation and collect network activities
for timeStep in range(nTimeSteps):

    sys.stdout.write('Current time step: ' + str(timeStep*stepDuration) + ' ms\n')

    # Set LGN input, using the image input
    for i in range(ImageNumPixelRows):
        for j in range(ImageNumPixelColumns):
            LGNBrightInput[i*ImageNumPixelColumns + j].rate = 500*max((input[i][j]/127.0-1.0), 0)
            LGNDarkInput[i*ImageNumPixelColumns + j].rate = 500*max(1.0-(input[i][j]/127.0), 0.0)

    # Segmentation signals
    if numSegmentationLayers > 1:
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

    # Store results for later plotting
    plotDensityLGNBright = [[0 for j in range(ImageNumPixelColumns)] for i in range(ImageNumPixelRows)]
    plotDensityOrientationV1 = [[[0 for j in range(numPixelColumns)] for i in range(numPixelRows)] for k in range(numOrientations)]
    plotDensityOrientationV2 = [[[[0 for j in range(numPixelColumns)] for i in range(numPixelRows)] for h in range(numSegmentationLayers)] for k in range(numOrientations)]
    LGNBrightSpikeCountUpToNow = LGNBright.get_spike_counts().values()
    V1SpikeCountUpToNow = V1.get_spike_counts().values()
    V2SpikeCountUpToNow = V2.get_spike_counts().values()

    # V1 and V2 sampling
    for k in range(0, numOrientations):                    # Orientations
        for i in range(0, numPixelRows):                   # Rows
            for j in range(0, numPixelColumns):            # Columns
                plotDensityOrientationV1[k][i][j] += V1SpikeCountUpToNow[k*numPixelRows*numPixelColumns + i*numPixelColumns + j] - cumplotDensityOrientationV1[k][i][j]  # spike count for the current step
                cumplotDensityOrientationV1[k][i][j] += plotDensityOrientationV1[k][i][j]  # update cumulative spikes
                for h in range(0, numSegmentationLayers):  # Segmentation layers
                    plotDensityOrientationV2[k][h][i][j] += V2SpikeCountUpToNow[h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j] - cumplotDensityOrientationV2[k][h][i][j]  # spike count for the current step
                    cumplotDensityOrientationV2[k][h][i][j] += plotDensityOrientationV2[k][h][i][j]  # update cumulative spikes

    # LGN sampling
    for i in range(0, ImageNumPixelRows):                  # Rows
        for j in range(0, ImageNumPixelColumns):           # Columns
            plotDensityLGNBright[i][j] += LGNBrightSpikeCountUpToNow[i*ImageNumPixelColumns + j] - cumplotDensityLGNBright[i][j]
            cumplotDensityLGNBright[i][j] += plotDensityLGNBright[i][j]  # update cumulative spikes

    # Set up images for boundaries
    maxDV1 = numpy.max(plotDensityOrientationV1)
    maxDV2 = numpy.max(plotDensityOrientationV2)
    newMaxV1 = max(maxDV1, newMaxV1)
    newMaxV2 = max(maxDV2, newMaxV2)
    if numOrientations == 8:
        rgbMap = numpy.array([[0.,.5,.5], [0.,0.,1.], [.5,0.,.5], [1.,0.,0.], [.5,0.,.5], [0.,0.,1.], [0.,.5,.5], [0.,1.,0.]])
    dataV1 = numpy.zeros((numPixelRows, numPixelColumns,3), dtype=numpy.uint8)
    for h in range(0, numSegmentationLayers):
        dataV2 = numpy.zeros((numPixelRows, numPixelColumns,3), dtype=numpy.uint8)
        if maxDV2 > 0:
            for i in range(0, numPixelRows):         # Rows
                for j in range(0, numPixelColumns):  # Columns
                    if numOrientations==2:           # Vertical and horizontal
                        dataV1[i][j] = [plotDensityOrientationV1[0][i][j], plotDensityOrientationV1[1][i][j], 0]
                        dataV2[i][j] = [plotDensityOrientationV2[0][h][i][j], plotDensityOrientationV2[1][h][i][j], 0]
                    if numOrientations==4:           # Vertical, horizontal, either diagonal
                        diagV1 = max(plotDensityOrientationV1[0][i][j], plotDensityOrientationV1[2][i][j])
                        diagV2 = max(plotDensityOrientationV2[0][h][i][j], plotDensityOrientationV2[2][h][i][j])
                        dataV1[i][j] = [plotDensityOrientationV1[1][i][j], plotDensityOrientationV1[3][i][j], diagV1]
                        dataV2[i][j] = [plotDensityOrientationV2[1][h][i][j], plotDensityOrientationV2[3][h][i][j], diagV2]
                    if numOrientations==8:
                        tempV1 = [plotDensityOrientationV1[k][i][j] for k in numOrientations]
                        tempV2 = [plotDensityOrientationV2[k][h][i][j] for k in numOrientations]
                        dataV1[i][j] = rgbMap[numpy.argmax(tempV1)]*numpy.max(tempV1)
                        dataV2[i][j] = rgbMap[numpy.argmax(tempV2)]*numpy.max(tempV2)
        outImagesV2[h].append(dataV2)
    outImagesV1.append(dataV1)

    # Set up images for LGN
    maxDLGNBright = numpy.max(plotDensityLGNBright)
    newMaxLGNBright = max(maxDLGNBright, newMaxLGNBright)
    if maxDLGNBright > 0:
        outImagesLGNBright.append(numpy.array(plotDensityLGNBright, dtype=numpy.uint8))

# End of time steps: close the simulation
sim.end()

# Create animated gifs for the recorded neuron layers ; rescale firing rates to max value
if newMaxV1 == 0:
    newMaxV1 = 1
writeGif("V1Orientations.GIF", [255/newMaxV1*data for data in outImagesV1], duration=0.2)
for h in range(0,numSegmentationLayers):
    if newMaxV2==0:
        newMaxV2 = 1
    writeGif("V2OrientationsSeg"+str(h)+".GIF", [255/newMaxV2*data for data in outImagesV2[h]], duration=0.2)
if newMaxLGNBright == 0:
    newMaxLGNBright = 1
writeGif("LGNBright.GIF", [255/newMaxLGNBright*data for data in outImagesLGNBright], duration=0.2)

