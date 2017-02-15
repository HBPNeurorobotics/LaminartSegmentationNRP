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
dt           = 1.0     # (ms) time step for network updates
stepDuration = 50.0    # (ms) time step for visual input and segmentation signal updates
simTime      = 1000.0  # (ms)
nTimeSteps   = numpy.int(simTime/stepDuration)
sloMoRate    = 4.0     # how much the GIFs are slowed vs real time, default is 4.0
sim.setup(timestep=dt, min_delay=1.0, max_delay=10.0)  # delays in ms

# General parameters
fileName = "squares 1"     # this would be removed if it is in the NRP
input, ImageNumPixelRows, ImageNumPixelColumns = setInput.readAndCropBMP(fileName, onlyZerosAndOnes=0)
print "Input image dimensions of " + fileName + ": [Rows, Columns] = " + str([ImageNumPixelRows,ImageNumPixelColumns])
weightScale   = 1.0        # general weight for all connections between neurons
constantInput = 0.5        # input given to the tonic interneuron layer and to the top-down activated segmentation neurons
inputPoisson  = 0          # 1 for a poisson spike source input, 0 for a DC current input
cellParams = {             # parameters for any neuron in the network
    'i_offset'   :   0.0,  # (nA)
    'tau_m'      :  10.0,  # (ms)
    'tau_syn_E'  :   2.0,  # (ms)
    'tau_syn_I'  :   2.0,  # (ms)
    'tau_refrac' :   2.0,  # (ms)
    'v_rest'     : -70.0,  # (mV)
    'v_reset'    : -70.0,  # (mV)
    'v_thresh'   : -56.0,  # (mV) -55.0 is NEST standard, -56.0 good for the current setup
    'cm'         :   0.25} # (nF)
cellType = sim.IF_curr_alpha(**cellParams)
connections = {
    # Input and LGN
    'brightInputToLGN'        :  0.1,   # only useful if inputDC == 1
    'darkInputToLGN'          :  0.1,   # only useful if inputDC == 1
    'LGN_ToV1Excite'          :  0.4,   #   400.0
    'LGN_ToV4Excite'          :  0.28,  #   280.0,

    # V1 layers
    'V1_6To4Excite'           :  0.001, #     1.0,
    'V1_6To4Inhib'            : -0.001, #    -1.0,
    'V1_4To23Excite'          :  0.5,   #   500.0,
    'V1_23To6Excite'          :  0.1,   #   100.0,
    'V1_ComplexExcite'        :  0.5,   #   500.0,
    'V1_ComplexInhib'         : -0.5,   #  -500.0,
    'V1_FeedbackExcite'       :  0.5,   #   500.0,
    'V1_NegFeedbackInhib'     : -1.5,   # -1500.0,
    'V1_InterInhib'           : -1.5,   # -1500.0,
    'V1_CrossOriInhib'        : -1.0,   # -1000.0,
    'V1_EndCutExcite'         :  1.5,   #  1500.0,
    'V1_ToV2Excite'           :  10.0,  # 10000.0,

    # V2 layers
    'V2_6To4Excite'           :  0.001, #     1.0,
    'V2_6To4Inhib'            : -0.02,  #   -20.0,
    'V2_4To23Excite'          :  0.5,   #   500.0,
    'V2_23To6Excite'          :  0.1,   #   100.0,
    'V2_ToV1FeedbackExcite'   :  1.0,   #   (not used at the moment)
    'V2_ComplexExcite'        :  0.5,   #   500.0,
    'V2_ComplexInhib'         : -1.0,   # -1000.0,
    'V2_ComplexInhib2'        : -0.1,   #  -100.0,
    'V2_CrossOriInhib'        : -1.2,   # -1200.0,
    'V2_FeedbackExcite'       :  0.5,   #   500.0,
    'V2_NegFeedbackInhib'     : -0.8,   #  -800.0,
    'V2_InterInhib'           : -1.5,   # -1500.0,
    'V2_BoundaryInhib'        : -5.0,   # -5000.0,
    'V2_SegmentInhib'         :-20.0,   #-20000.0, TRY TO ADAPT THIS TO THE NEURNO THRESHOLD POTENTIAL

    # V4 layers
    'V4_BrightnessExcite'     :  2.0,   #  2000.0,
    'V4_BrightnessInhib'      : -2.0,   # -2000.0,
    'V4_BetweenColorsInhib'   : -5.0,   # -5000.0,

    # Surface segmentation layers
    'S_SegmentSpreadExcite'   :  1.0,   #  1000.0,
    'S_SegmentOnOffInhib'     : -5.0,   # -5000.0,
    'S_SegmentGatingInhib'    : -5.0,   # -5000.0,

    # Boundary segmentation layers
    'B_SegmentOnOffInhib'     : -5.0,   # -5000.0,
    'B_SegmentGatingInhib'    : -5.0,   # -5000.0,
    'B_SegmentSpreadExcite'   :  2.0,   #  2000.0,
    'B_SegmentTonicInhib'     :-20.0,   #-20000.0,
    'B_SegmentOpenFlowInhib'  : -0.15}  #  -150.0}
for key, value in connections.items():
    connections[key] = value*weightScale

# Orientation filters parameters
numOrientations = 4                       # number of orientations   (2, 4 or 8 ; 8 is experimental)
oriFilterSize   = 4                       # better as an even number (how V1 pools from LGN) ; 4 is original
V1PoolSize      = 3                       # better as an odd  number (pooling in V1)         ; 3 is original
V2PoolSize      = 7                       # better as an odd  number (pooling in V2)         ; 7 is original
phi             = 0*numpy.pi/2            # filter phase (sometimes convenient to shift all filters)
numPixelRows    = ImageNumPixelRows+1     # number of rows for the oriented grid (in between un-oriented pixels)
numPixelColumns = ImageNumPixelColumns+1  # same for columns

# Segmentation parameters
numSegmentationLayers        = 3          # number of segmentation layers (usual is 3, minimum is 1)
useSurfaceSegmentation       = 0          # use segmentation that flows across closed shapes
useBoundarySegmentation      = 1          # use segmentation that flows along connected boundaries
useSDPropToDist              = 0          # if 1, segmentationTargetLocationSD ~ dist(segmentationTargetLocation;fix.point)
minSD                        = 3          # minimum segmentationTargetLocationSD, (e.g. sending segmentation signals around fovea)
rateSD                       = 0.1        # how much segmentationTargetLocationSD grows with excentricity (pixels SD per pixel excentricity)
segmentationTargetLocationSD = 0          # standard deviation of where location actually ends up (originally 8) ; in pixels
segmentationSignalSize       = 20         # even number ; circle diameter where a segmentation signal is triggered ; in pixels
segmentationSignalStart      = 100.0      # in ms, start of the segmentation signal (too soon and the segmentation acts on transient boundaries)
segLocX, segLocY             = setInput.chooseSegmentationSignal(fileName, segmentationSignalSize)
if numSegmentationLayers == 1 or fileName in ["Test", "Test2", "Test3"]:
    numSegmentationLayers    = 1          # for small tests (no segmentation)
    useSurfaceSegmentation   = 0
    useBoundarySegmentation  = 0

# Build the whole network and take the layers that have to be updated during the simulation (will modify sim)
network = buildNetworkAndConnections(sim, ImageNumPixelRows, ImageNumPixelColumns, numOrientations, oriFilterSize, V1PoolSize, V2PoolSize, phi,
                                     connections, cellType, numSegmentationLayers, useBoundarySegmentation, useSurfaceSegmentation)

# Take the neuron layers that need to be updated online or that we want to take records of
LGNBright      = network.get_population("LGNBright")
LGNDark        = network.get_population("LGNDark")
V1             = network.get_population("V1LayerToPlot")
V2             = network.get_population("V2LayerToPlot")
V4Bright       = network.get_population("V4Brightness")
V4Dark         = network.get_population("V4Darkness")
LGNBright.record("spikes")
LGNDark  .record("spikes")
V1       .record("spikes")
V2       .record("spikes")
V4Bright .record("spikes")
V4Dark   .record("spikes")
if inputPoisson:
    LGNBrightInput = sim.Population(ImageNumPixelRows*ImageNumPixelColumns, sim.SpikeSourcePoisson())
    LGNDarkInput   = sim.Population(ImageNumPixelRows*ImageNumPixelColumns, sim.SpikeSourcePoisson())
    sim.Projection(LGNBrightInput, LGNBright, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['brightInputToLGN']))
    sim.Projection(LGNDarkInput,   LGNDark  , sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['darkInputToLGN']))
if numSegmentationLayers > 1:
    if useSurfaceSegmentation:
        SurfaceSegmentationOff  = network.get_population("SurfaceSegmentationOff")
        SurfaceSegmentationOn   = network.get_population("SurfaceSegmentationOn")
    if useBoundarySegmentation:
        BoundarySegmentationOff = network.get_population("BoundarySegmentationOff")
        BoundarySegmentationOn  = network.get_population("BoundarySegmentationOn")
        network.get_population("BoundarySegmentationOnInter3").inject(sim.DCSource(amplitude=constantInput, start=0.0, stop=simTime))

########################################################################################
### Network is defined, now set up stimulus, segmentation signal and run everything! ###
########################################################################################

# Stuff to plot activity of LGN, V1, V2, etc. layers
cumplotDensityLGNBright     =   [[0 for j in range(ImageNumPixelColumns)] for i in range(ImageNumPixelRows)]
cumplotDensityLGNDark       =   [[0 for j in range(ImageNumPixelColumns)] for i in range(ImageNumPixelRows)]
cumplotDensityOrientationV2 = [[[[0 for j in range(numPixelColumns)] for i in range(numPixelRows)] for h in range(numSegmentationLayers)] for k in range(numOrientations)]
cumplotDensityOrientationV1 =  [[[0 for j in range(numPixelColumns)] for i in range(numPixelRows)] for k in range(numOrientations)]
cumplotDensityBrightnessV4  =  [[[0 for j in range(ImageNumPixelColumns)] for i in range(ImageNumPixelRows)] for h in range(numSegmentationLayers)]
cumplotDensityDarknessV4    =  [[[0 for j in range(ImageNumPixelColumns)] for i in range(ImageNumPixelRows)] for h in range(numSegmentationLayers)]
newMaxInput      = 0
newMaxV1         = 0
newMaxV2         = 0
newMaxBrightness = 0
newMaxDarkness   = 0
newMaxLGNBright  = 0
newMaxLGNDark    = 0
outImagesInput      = []
outImagesV1         = []
outImagesV2         = [[] for i in range(numSegmentationLayers)]
outImagesV4         = [[] for i in range(numSegmentationLayers)]
outImagesLGNBright  = []
outImagesLGNDark    = []

# Set input, segmentation signals, run simulation and collect network activities
for timeStep in range(nTimeSteps):

    sys.stdout.write('Current time step: ' + str(timeStep*stepDuration) + ' ms\n')

    # Set LGN input, using the image input
    inputToPlot = input.copy()            # Track the input (when it will not be constant)
    for i in range(ImageNumPixelRows):         # Rows
        for j in range(ImageNumPixelColumns):  # Columns

            if inputPoisson:
                LGNBrightInput[i*ImageNumPixelColumns + j].rate     = 500.0*max(0.0, (input[i][j]/127.0-1.0))
                LGNDarkInput  [i*ImageNumPixelColumns + j].rate     = 500.0*max(0.0, 1.0-(input[i][j]/127.0))
                LGNBrightInput[i*ImageNumPixelColumns + j].start    = timeStep*stepDuration # not sure if these 4 lines are necessary
                LGNDarkInput  [i*ImageNumPixelColumns + j].start    = timeStep*stepDuration
                LGNBrightInput[i*ImageNumPixelColumns + j].duration = stepDuration
                LGNDarkInput  [i*ImageNumPixelColumns + j].duration = stepDuration
            else:
                bright = sim.DCSource(amplitude = numpy.random.poisson(10.0*max(0.0, (input[i][j]/127.0-1.0))), start=timeStep*stepDuration, stop=(timeStep+1)*stepDuration)
                dark   = sim.DCSource(amplitude = numpy.random.poisson(10.0*max(0.0, 1.0-(input[i][j]/127.0))), start=timeStep*stepDuration, stop=(timeStep+1)*stepDuration)
                LGNBright[i*ImageNumPixelColumns + j].inject(bright)
                LGNDark  [i*ImageNumPixelColumns + j].inject(dark)

    # Segmentation signals deal here
    if numSegmentationLayers > 1 and timeStep == 0:
        surfaceOnTarget  = []
        surfaceOffTarget = []
        boundaryOnTarget = []
        for h in range(numSegmentationLayers-1):

            # Pick a segmentation signal location (may be separate surface and boundary, if it is possible to use both at the same time?)
            segmentationTargetLocationX, segmentationTargetLocationY = [int(numpy.sign(h-0.5))*segLocX,segLocY] # only works for nSegL<3 (to change in the NRP)
            if useSDPropToDist:
                segmentationTargetLocationSD = minSD + rateSD*numpy.sqrt((segmentationTargetLocationX)**2 + segmentationTargetLocationY**2)
            segmentLocationX = int(round(random.gauss(ImageNumPixelColumns/2 - segmentationTargetLocationX, segmentationTargetLocationSD)))
            segmentLocationY = int(round(random.gauss(ImageNumPixelRows/2    - segmentationTargetLocationY, segmentationTargetLocationSD)))
            print "Segmentation location [X, Y] = " + str([segmentLocationX,segmentLocationY])

            # Define surface segmentation signals (gives local DC inputs to surface and boundary segmentation networks)
            if useSurfaceSegmentation:
                for i in range(0, ImageNumPixelRows):         # Rows
                    for j in range(0, ImageNumPixelColumns):  # Columns

                        # Off signals are at borders of image (will flow in to other locations unless stopped by boundaries)
                        if i==0 or i==(ImageNumPixelRows-1) or j==0 or j==(ImageNumPixelColumns-1):
                            surfaceOffTarget.append(h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)

                        # On signals start the surface segmentation at a specific location and flow across closed shapes formed by boundaries
                        distance = numpy.sqrt(numpy.power(segmentLocationX-j, 2) + numpy.power(segmentLocationY-i, 2))
                        if distance < segmentationSignalSize:
                            surfaceOnTarget.append(h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)
                            inputToPlot[i][j] += 20           # Track segmentation signal locations

            # Define boundary segmentation signals
            if useBoundarySegmentation:
                for i in range(0, ImageNumPixelRows):         # Rows
                    for j in range(0, ImageNumPixelColumns):  # Columns

                        # On signals start the boundary segmentation at a specific location and flow along connected boundaries
                        distance = numpy.sqrt(numpy.power(segmentLocationX-j, 2) + numpy.power(segmentLocationY-i, 2))
                        if distance < segmentationSignalSize:
                            boundaryOnTarget.append(h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)
                            inputToPlot[i][j] += 20           # Track segmentation signal locations

        # Set a firing positive firing rate for concerned units of the segmentation top-down signal
        if len(surfaceOffTarget) > 0:
            SurfaceSegmentationOff[surfaceOffTarget].inject(sim.DCSource(amplitude=1.0, start=segmentationSignalStart, stop=simTime))
        if len(surfaceOnTarget)  > 0:
            SurfaceSegmentationOn [surfaceOnTarget] .inject(sim.DCSource(amplitude=1.0, start=segmentationSignalStart, stop=simTime))
        if len(boundaryOnTarget) > 0:
            BoundarySegmentationOn[boundaryOnTarget].inject(sim.DCSource(amplitude=1.0, start=segmentationSignalStart, stop=simTime))

    # Actual run of the network, using the input and the segmentation signals
    sim.run(stepDuration)

    # To store results for later plotting
    plotDensityLGNBright       =   [[0 for j in range(ImageNumPixelColumns)] for i in range(ImageNumPixelRows)]
    plotDensityLGNDark         =   [[0 for j in range(ImageNumPixelColumns)] for i in range(ImageNumPixelRows)]
    plotDensityOrientationV1   =  [[[0 for j in range(numPixelColumns)] for i in range(numPixelRows)] for k in range(numOrientations)]
    plotDensityOrientationV2   = [[[[0 for j in range(numPixelColumns)] for i in range(numPixelRows)] for h in range(numSegmentationLayers)] for k in range(numOrientations)]
    plotDensityBrightnessV4    =  [[[0 for j in range(ImageNumPixelColumns)] for i in range(ImageNumPixelRows)] for h in range(numSegmentationLayers)]
    plotDensityDarknessV4      =  [[[0 for j in range(ImageNumPixelColumns)] for i in range(ImageNumPixelRows)] for h in range(numSegmentationLayers)]

    # Record spike counts up to now from the actual neuron populations
    LGNBrightSpikeCountUpToNow = [value for (key, value) in sorted(LGNBright.get_spike_counts().items())]
    LGNDarkSpikeCountUpToNow   = [value for (key, value) in sorted(LGNDark  .get_spike_counts().items())]
    V1SpikeCountUpToNow        = [value for (key, value) in sorted(V1       .get_spike_counts().items())]
    V2SpikeCountUpToNow        = [value for (key, value) in sorted(V2       .get_spike_counts().items())]
    V4BrightSpikeCountUpToNow  = [value for (key, value) in sorted(V4Bright .get_spike_counts().items())]
    V4DarkSpikeCountUpToNow    = [value for (key, value) in sorted(V4Dark   .get_spike_counts().items())]

    # V1 and V2 sampling
    for k in range(0, numOrientations):                    # Orientations
        for i in range(0, numPixelRows):                   # Rows
            for j in range(0, numPixelColumns):            # Columns
                plotDensityOrientationV1[k][i][j] += V1SpikeCountUpToNow[k*numPixelRows*numPixelColumns + i*numPixelColumns + j] - cumplotDensityOrientationV1[k][i][j]  # spike count for the current step
                cumplotDensityOrientationV1[k][i][j] += plotDensityOrientationV1[k][i][j]  # update cumulative spikes
                for h in range(0, numSegmentationLayers):  # Segmentation layers
                    plotDensityOrientationV2[k][h][i][j] += V2SpikeCountUpToNow[h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j] - cumplotDensityOrientationV2[k][h][i][j]  # spike count for the current step
                    cumplotDensityOrientationV2[k][h][i][j] += plotDensityOrientationV2[k][h][i][j]  # update cumulative spikes

    # V4 sampling
    for h in range(0, numSegmentationLayers):
        for i in range(0, ImageNumPixelRows):  # Rows
            for j in range(0, ImageNumPixelColumns):  # Columns
                plotDensityBrightnessV4[h][i][j] += V4BrightSpikeCountUpToNow[h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j] - cumplotDensityBrightnessV4[h][i][j]
                plotDensityDarknessV4  [h][i][j] += V4DarkSpikeCountUpToNow  [h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j] - cumplotDensityDarknessV4  [h][i][j]
                cumplotDensityBrightnessV4[h][i][j] += plotDensityBrightnessV4[h][i][j]  # update cumulative spikes
                cumplotDensityDarknessV4  [h][i][j] += plotDensityDarknessV4  [h][i][j]  # update cumulative spikes

    # LGN sampling
    for i in range(0, ImageNumPixelRows):                  # Rows
        for j in range(0, ImageNumPixelColumns):           # Columns
            plotDensityLGNBright[i][j] += LGNBrightSpikeCountUpToNow[i*ImageNumPixelColumns + j] - cumplotDensityLGNBright[i][j]
            plotDensityLGNDark  [i][j] += LGNDarkSpikeCountUpToNow  [i*ImageNumPixelColumns + j] - cumplotDensityLGNDark  [i][j]
            cumplotDensityLGNBright[i][j] += plotDensityLGNBright[i][j]  # update cumulative spikes
            cumplotDensityLGNDark[i][j]   += plotDensityLGNDark  [i][j]  # update cumulative spikes

    # Set up images for stimulus and segmentation signal locations
    maxInput    = numpy.max(inputToPlot)
    newMaxInput = max(maxInput, newMaxInput)
    if maxInput > 0:
        outImagesInput.append(numpy.array(inputToPlot, dtype=numpy.uint16))

    # Set up images for boundaries
    maxDV1   = numpy.max(plotDensityOrientationV1)
    maxDV2   = numpy.max(plotDensityOrientationV2)
    newMaxV1 = max(maxDV1, newMaxV1)
    newMaxV2 = max(maxDV2, newMaxV2)
    if numOrientations == 8:
        rgbMap = numpy.array([[0.,.5,.5], [0.,0.,1.], [.5,0.,.5], [1.,0.,0.], [.5,0.,.5], [0.,0.,1.], [0.,.5,.5], [0.,1.,0.]])
    dataV1 = numpy.zeros((numPixelRows, numPixelColumns,3), dtype=numpy.uint8)

    # V1
    if maxDV1 > 0:
        for i in range(0, numPixelRows):         # Rows
            for j in range(0, numPixelColumns):  # Columns
                if numOrientations==2:           # Vertical and horizontal
                    dataV1[i][j] = [plotDensityOrientationV1[0][i][j], plotDensityOrientationV1[1][i][j], 0]
                if numOrientations==4:           # Vertical, horizontal, either diagonal
                    diagV1 = max(plotDensityOrientationV1[0][i][j], plotDensityOrientationV1[2][i][j])
                    dataV1[i][j] = [plotDensityOrientationV1[1][i][j], plotDensityOrientationV1[3][i][j], diagV1]
                if numOrientations==8:
                    tempV1 = [plotDensityOrientationV1[k][i][j] for k in numOrientations]
                    dataV1[i][j] = rgbMap[numpy.argmax(tempV1)]*numpy.max(tempV1)
    outImagesV1.append(dataV1)

    # V2
    for h in range(0, numSegmentationLayers):
        dataV2 = numpy.zeros((numPixelRows, numPixelColumns,3), dtype=numpy.uint8)
        if maxDV2 > 0:
            for i in range(0, numPixelRows):         # Rows
                for j in range(0, numPixelColumns):  # Columns
                    if numOrientations==2:           # Vertical and horizontal
                        dataV2[i][j] = [plotDensityOrientationV2[0][h][i][j], plotDensityOrientationV2[1][h][i][j], 0]
                    if numOrientations==4:           # Vertical, horizontal, either diagonal
                        diagV2 = max(plotDensityOrientationV2[0][h][i][j], plotDensityOrientationV2[2][h][i][j])
                        dataV2[i][j] = [plotDensityOrientationV2[1][h][i][j], plotDensityOrientationV2[3][h][i][j], diagV2]
                    if numOrientations==8:
                        tempV2 = [plotDensityOrientationV2[k][h][i][j] for k in numOrientations]
                        dataV2[i][j] = rgbMap[numpy.argmax(tempV2)]*numpy.max(tempV2)
        outImagesV2[h].append(dataV2)

    # Set up images for brightness (V4)
    maxDBrightness   = numpy.max(plotDensityBrightnessV4)
    maxDDarkness     = numpy.max(plotDensityDarknessV4)
    newMaxBrightness = max(maxDBrightness, newMaxBrightness)
    newMaxDarkness   = max(maxDDarkness,   newMaxDarkness)
    if maxDBrightness > 0 or maxDDarkness > 0:
        for h in range(0, numSegmentationLayers):
            outImagesV4[h].append(numpy.array(plotDensityBrightnessV4[h], dtype=numpy.uint8))

    # Set up images for LGN
    maxDLGNBright   = numpy.max(plotDensityLGNBright)
    maxDLGNDark     = numpy.max(plotDensityLGNDark)
    newMaxLGNBright = max(maxDLGNBright, newMaxLGNBright)
    newMaxLGNDark   = max(maxDLGNDark,   newMaxLGNDark)
    if maxDLGNBright > 0:
        outImagesLGNBright.append(numpy.array(plotDensityLGNBright, dtype=numpy.uint8))
    if maxDLGNBright > 0:
        outImagesLGNDark  .append(numpy.array(plotDensityLGNDark,   dtype=numpy.uint8))

# End of time steps: close the simulation
sim.end()

# Create animated gifs for the recorded neuron layers ; rescale firing rates to max value
duration = sloMoRate*stepDuration/1000.0
writeGif("InputAndSegSignals.GIF", [255/(1*(newMaxInput==0)     + newMaxInput)    *data for data in outImagesInput],     duration=duration)
writeGif("V1Orientations.GIF",     [255/(1*(newMaxV1==0)        + newMaxV1)       *data for data in outImagesV1],        duration=duration)
writeGif("LGNBright.GIF",          [255/(1*(newMaxLGNBright==0) + newMaxLGNBright)*data for data in outImagesLGNBright], duration=duration)
writeGif("LGNDark.GIF",            [255/(1*(newMaxLGNDark==0)   + newMaxLGNDark)  *data for data in outImagesLGNDark],   duration=duration)
for h in range(0,numSegmentationLayers):
    writeGif("V2OrientationsSeg"+str(h)+".GIF", [255/(1*(newMaxV2==0)         + newMaxV2)        *data for data in outImagesV2[h]], duration=duration)
    writeGif("V4BrightnessSeg"+str(h)  +".GIF", [255/(1*(newMaxBrightness==0) + newMaxBrightness)*data for data in outImagesV4[h]], duration=duration)
