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
    'brightInputToLGN'        :   3000.0,
    'darkInputToLGN'          :   3000.0,
    'inputToLGN'              :    700.0,

    # V1 or V2
    'excite6to4'              :      1.0,
    'inhibit6to4'             :     -1.0,
    'complexExcit'            :    500.0,
    'complexInhib'            :   -500.0,

    # V1 layers
    'V1Feedback'              :    500.0,
    'V1NegFeedback'           :  -1500.0,
    'endCutExcit'             :   1500.0,
    '23to6Excite'             :    100.0,
    'interInhibV1'            :  -1500.0,
    'crossInhib'              :  -1000.0,
    'V2toV1'                  :  10000.0,
    'V2inhibit6to4'           :    -20.0,

    # V2 layers
    'V2PoolInhib'             :  -1000.0,
    'V2PoolInhib2'            :   -100.0,
    'V2OrientInhib'           :  -1200.0,
    'V2Feedback'              :    500.0,
    'V2NegFeedback'           :   -800.0,
    'interInhibV2'            :  -1500.0,

    # V4 layers
    'LGNV4excit'              :    280.0,
    'V4betweenColorInhib'     :  -5000.0,
    'brightnessInhib'         :  -2000.0,
    'brightnessExcite'        :   2000.0,
    'boundaryInhib'           :  -5000.0,
    'V4inhib'                 :   -200.0,

    # Surface segmentation layers
    'SegmentInhib'            :  -5000.0,
    'SegmentInhib2'           : -20000.0,
    'SegmentExcite'           :      1.0,
    'BoundaryToSegmentExcite' :   2000.0,
    'brightnessExcite2'       :   1000.0,

    # Boundary segmentation layers
    'SegmentInhib3'           :   -150.0,
    'SegmentInhib4'           :  -5000.0,
    'SegmentInhib5'           : -20000.0,
    'SegmentExcite1'          :   2000.0,
    'SegmentExcite2'          :      0.5,
    'SegmentExcite4'          :    500.0}

# Orientation filters parameters
numOrientations = 2		                 # number of orientations (2, 4 or 8 ; 8 is experimental)
oriFilterSize = 4                        # better as an even number (filtering from LGN to V1) ; 4 is original
V1PoolSize = 3                           # better as an odd number (pooling in V1) ; 3 is original
V2PoolSize = 7                           # better as an odd number (pooling in V2) ; 7 is original
numPixelRows    = ImageNumPixelRows+1    # number of rows for the oriented grid (in between un-oriented pixels)
numPixelColumns = ImageNumPixelColumns+1 # same for columns

# Segmentation parameters
numSegmentationLayers = 3        # number of segmentation layers (usual is 3, minimum is 1)
useSurfaceSegmentation = 0       # use segmentation that flows across closed shapes
useBoundarySegmentation = 1      # use segmentation that flows along connected boundaries
useSDPropToDist = 0			     # if 1, segmentationTargetLocationSD ~ dist(segmentationTargetLocation;fix.point)
minSD = 3                        # minimum segmentationTargetLocationSD, (e.g. sending segmentation signals around fovea)
rateSD = 0.1                     # how much segmentationTargetLocationSD grows with excentricity (pixels SD per pixel excentricity)
segmentationTargetLocationSD = 8 # standard deviation of where location actually ends up (originally 8) ; in pixels
segmentationSignalSize = 20	     # even number ; circle diameter where a segmentation signal is triggered ; in pixels
if numSegmentationLayers == 1 or fileName in ["Test", "Test2"]:
    numSegmentationLayers = 1    # for test cases
    useSurfaceSegmentation = 0
    useBoundarySegmentation = 0

# Build the whole network and take the layers that have to be updated during the simulation
sim, network = buildNetworkAndConnections(sim, ImageNumPixelRows, ImageNumPixelColumns, numOrientations, oriFilterSize, V1PoolSize, V2PoolSize, connections,
                                          normalCellType, tonicCellType, weightScale, numSegmentationLayers, useBoundarySegmentation, useSurfaceSegmentation)

LGNBrightInput = network.get_population("LGNBrightInput")
LGNDarkInput   = network.get_population("LGNDarkInput")
V2layer23      = network.get_population("V1layer6P1")
LGNBright      = network.get_population("LGNBright")
V2layer23.record("spikes")
LGNBright.record("spikes")
if useSurfaceSegmentation:
    SurfaceSegmentationOffSignal  = network.get_population("SurfaceSegmentationOffSignal")
    SurfaceSegmentationOnSignal   = network.get_population("SurfaceSegmentationOnSignal")
if useBoundarySegmentation:
    BoundarySegmentationOffSignal = network.get_population("BoundarySegmentationOffSignal")
    BoundarySegmentationOnSignal  = network.get_population("BoundarySegmentationOnSignal")

########################################################################################
### Network is defined, now set up stimulus, segmentation signal and run everything! ###
########################################################################################

cumplotDensityOrientation = [[[[0 for j in range(numPixelColumns)] for i in range(numPixelRows)] for h in range(numSegmentationLayers)] for k in range(numOrientations)] # TO DEFINE OUTSIDE TRIALS (si jamais)
cumplotDensityLGNBright = [[0 for j in range(ImageNumPixelColumns)] for i in range(ImageNumPixelRows)]
newMax = 0  # for plotting
newMaxLGNBright = 0
outImages = [[] for i in range(numSegmentationLayers)]  # to create animated gifs for V2 boundaries of different segmentation layers
outImagesLGNBright = []

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
    plotDensityOrientation = [[[[0 for j in range(numPixelColumns)] for i in range(numPixelRows)] for h in range(numSegmentationLayers)] for k in range(numOrientations)]
    plotDensityLGNBright   = [[0 for j in range(ImageNumPixelColumns)] for i in range(ImageNumPixelRows)]
    V2layer23SpikeCountUpToNow = V2layer23.get_spike_counts().values()
    LGNBrightSpikeCountUpToNow = LGNBright.get_spike_counts().values()
    for i in range(0, numPixelRows):                    # Rows
        for j in range(0, numPixelColumns):             # Columns
            for h in range(0, numSegmentationLayers):   # Segmentation layers
                for k in range(0, numOrientations):     # Orientations
                    plotDensityOrientation[k][h][i][j] += V2layer23SpikeCountUpToNow[h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j] - cumplotDensityOrientation[k][h][i][j]  # spike count for the current step
                    cumplotDensityOrientation[k][h][i][j] += plotDensityOrientation[k][h][i][j]  # update cumulative spikes

    for i in range(0, ImageNumPixelRows):               # Rows
        for j in range(0, ImageNumPixelColumns):        # Columns
            plotDensityLGNBright[i][j] += LGNBrightSpikeCountUpToNow[i*ImageNumPixelColumns + j] - cumplotDensityLGNBright[i][j]
            cumplotDensityLGNBright[i][j] += plotDensityLGNBright[i][j]  # update cumulative spikes

    # Set up images for boundaries
    maxD = numpy.max(plotDensityOrientation)
    newMax = max(maxD, newMax)
    if numOrientations == 8:
        rgbMap = numpy.array([[0.,.5,.5], [0.,0.,1.], [.5,0.,.5], [1.,0.,0.], [.5,0.,.5], [0.,0.,1.], [0.,.5,.5], [0.,1.,0.]])
    for h in range(0, numSegmentationLayers):
        data = numpy.zeros((numPixelRows, numPixelColumns,3), dtype=numpy.uint8)
        if maxD > 0:
            for i in range(0, numPixelRows):         # Rows
                for j in range(0, numPixelColumns):  # Columns
                    if numOrientations==2:           # Vertical and horizontal
                        data[i][j] = [plotDensityOrientation[0][h][i][j], plotDensityOrientation[1][h][i][j], 0]
                    if numOrientations==4:           # Vertical, horizontal, either diagonal
                        diag = max(plotDensityOrientation[0][h][i][j], plotDensityOrientation[2][h][i][j])
                        data[i][j] = [plotDensityOrientation[1][h][i][j], plotDensityOrientation[3][h][i][j], diag]
                    if numOrientations==8:
                        temp = [plotDensityOrientation[k][h][i][j] for k in numOrientations]
                        data[i][j] = rgbMap[numpy.argmax(temp)]*numpy.max(temp)
        outImages[h].append(data)

    maxDLGNBright = numpy.max(plotDensityLGNBright)
    newMaxLGNBright = max(maxDLGNBright, newMaxLGNBright)
    if maxDLGNBright > 0:
        outImagesLGNBright.append(numpy.array(plotDensityLGNBright, dtype=numpy.uint8))

# End of time steps: close the simulation
sim.end()

# Create animated gifs for several neuron layers ; rescale firing rates to max value
for h in range(0,numSegmentationLayers):
    if newMax==0:
        newMax = 1
    writeGif("V2OrientationsSeg"+str(h)+".GIF", [255/newMax*data for data in outImages[h]], duration=0.2)
if newMaxLGNBright == 0:
    newMaxLGNBright = 1
writeGif("LGNBright.GIF", [255/newMaxLGNBright*data for data in outImagesLGNBright], duration=0.2)

