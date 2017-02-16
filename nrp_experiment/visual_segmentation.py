#############################################################################
#############################################################################
### This file contains the setup of the neuronal network for segmentation ###
#############################################################################
#############################################################################

from hbp_nrp_cle.brainsim import simulator as sim
from pyNN.connectors import Connector
# from createFilters import createFilters, createPoolingConnectionsAndFilters
import numpy
import nest
import logging
from std_msgs.msg import String

logger = logging.getLogger(__name__)
# sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0)


####################################
### Define parameters down here! ###
####################################


# General parameters
ImageNumPixelRows = 5    # number of rows for the network (the input image is scaled to the dimensions of the network)
ImageNumPixelColumns = 5   # number of columns for the network (the input image is scaled to the dimensions of the network)
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
numOrientations = 2                       # number of orientations   (2, 4 or 8 ; 8 is experimental)
oriFilterSize   = 4                       # better as an even number (how V1 pools from LGN) ; 4 is original
V1PoolSize      = 3                       # better as an odd  number (pooling in V1)         ; 3 is original
V2PoolSize      = 7                       # better as an odd  number (pooling in V2)         ; 7 is original
phi             = 0.0*numpy.pi/2          # filters phase (sometimes useful to shift all orientations)
OppositeOrientationIndex = list(numpy.roll(range(numOrientations), numOrientations/2)) # perpendicular orientations
numPixelRows    = ImageNumPixelRows+1     # number of rows for the oriented grid (in between un-oriented pixels)
numPixelColumns = ImageNumPixelColumns+1  # same for columns

# Segmentation parameters
numSegmentationLayers        = 3          # number of segmentation layers (usual is 3, minimum is 1)
useSurfaceSegmentation       = 0          # use segmentation that flows across closed shapes
useBoundarySegmentation      = 1          # use segmentation that flows along connected boundaries
segmentationTargetLocationSD = 5          # standard deviation of where location actually ends up (originally 8) ; in pixels
segmentationSignalSize       = 20         # even number ; circle diameter where a segmentation signal is triggered ; in pixels
useSDPropToDist              = 0          # if 1, segmentationTargetLocationSD ~ dist(segmentationTargetLocation;fix.point)
if useSDPropToDist:
    minSD                    = 3          # minimum segmentationTargetLocationSD, (e.g. sending segm. signals around fovea)
    rateSD                   = 0.1        # how much segmentationTargetLocationSD grows with excentricity
if numSegmentationLayers == 1:
    useSurfaceSegmentation   = 0
    useBoundarySegmentation  = 0


###############################
### Some useful definitions ###
###############################


# Create a custom connector (use nest.Connect explicitly to go faster)
class MyConnector(Connector):
    def __init__(self, ST):
        self.source = [x[0] for x in ST]
        self.target = [x[1] for x in ST]
    def connect(self, projection):
        nest.Connect([projection.pre.all_cells[s] for s in self.source], [projection.post.all_cells[t] for t in self.target], 'one_to_one', syn_spec=projection.nest_synapse_model)

# Take filter parameters and build 2 oriented filters with different polarities
def createFilters(numOrientations, size, sigma2, Olambda, phi):

    # Initialize the filters
    filters1 = numpy.zeros((numOrientations, size, size))
    filters2 = numpy.zeros((numOrientations, size, size))

    # Fill them with gabors
    midSize = (size-1.)/2.
    maxValue = -1
    for k in range(0, numOrientations):
        theta = numpy.pi * (k + 1) / numOrientations + phi
        for i in range(0, size):
            for j in range(0, size):
                x = (i - midSize) * numpy.cos(theta) + (j-midSize)*numpy.sin(theta)
                y = -(i - midSize) * numpy.sin(theta) + (j-midSize)*numpy.cos(theta)
                filters1[k][i][j] = numpy.exp(-(x*x + y*y)/(2*sigma2)) * numpy.sin(2*numpy.pi*x/Olambda)
                filters2[k][i][j] = -filters1[k][i][j]

    # Rescale filters so max value is 1.0
    for k in range(0, numOrientations):
        maxValue = numpy.amax(numpy.abs(filters1[k]))
        filters1[k] /= maxValue
        filters2[k] /= maxValue
        filters1[k][numpy.abs(filters1[k]) < 0.3] = 0.0
        filters2[k][numpy.abs(filters2[k]) < 0.3] = 0.0

    return filters1, filters2

# Take filter parameters and build connection pooling and connection filters arrays
def createPoolingConnectionsAndFilters(numOrientations, VPoolSize, sigma2, Olambda, phi):

    # Set up layer23 pooling filters
    # Set up orientation kernels
    midSize = (VPoolSize - 1.) / 2.
    Vpoolingfilters = numpy.zeros((numOrientations, VPoolSize, VPoolSize))
    for k in range(0, numOrientations):
        theta = numpy.pi*(k+1)/numOrientations + phi
        # Make filter
        for i in range(0, VPoolSize):
            for j in range(0, VPoolSize):
                x = (i-midSize)*numpy.cos(theta) + (j-midSize)*numpy.sin(theta)
                y = -(i-midSize)*numpy.sin(theta) + (j-midSize)*numpy.cos(theta)
                Vpoolingfilters[k][i][j] = numpy.exp(-(x*x + y*y)/(2*sigma2)) * numpy.power(numpy.cos(2*numpy.pi*x/Olambda), sigma2+1)

    # Rescale filters so max value is 1.0
    maxValue = numpy.amax(numpy.abs(Vpoolingfilters))
    Vpoolingfilters /= maxValue
    Vpoolingfilters[Vpoolingfilters < 0.1] = 0.0
    Vpoolingfilters[Vpoolingfilters >= 0.1] = 1.0

    # Correct certain filters (so that every filter has the same number of active pixels) ; this applies to orientation like 22.5 deg
    for k in range(0, numOrientations):
        theta = 180*(k+1)/numOrientations + phi*180/numpy.pi
        if min(abs(theta-0),abs(theta-180)) < min(abs(theta-90),abs(theta-270)): # more horizontally oriented filter
            for i in range(0, VPoolSize):
                check = Vpoolingfilters[k,i:min(i+2,VPoolSize),:]
                Vpoolingfilters[k,min(i+1,VPoolSize-1),numpy.where(numpy.sum(check,axis=0) == 2.0)] = 0
        else:                                                                    # more vertically oriented filter
            for j in range(0, VPoolSize):
                check = Vpoolingfilters[k,:,j:min(j+2,VPoolSize)]
                Vpoolingfilters[k,numpy.where(numpy.sum(check,axis=1)==2.0),min(j+1,VPoolSize-1),] = 0

    # Two filters for different directions (1 = more to the right ; 2 = more to the left) ; starts from the basis of Vpoolingfilters
    Vpoolingconnections1 = Vpoolingfilters.copy()
    Vpoolingconnections2 = Vpoolingfilters.copy()

    # Do the pooling connections
    for k in range(0, numOrientations):

        # want only the end points of each filter line (remove all interior points)
        for i in range(1, VPoolSize - 1):
            for j in range(1, VPoolSize - 1):
                Vpoolingconnections1[k][i][j] = 0.0
                Vpoolingconnections2[k][i][j] = 0.0

        # segregates between right and left directions
        for i in range(0, VPoolSize):
            for j in range(0, VPoolSize):
                if j == (VPoolSize-1)/2:
                    Vpoolingconnections1[k][0][j] = 0.0
                    Vpoolingconnections2[k][VPoolSize-1][j] = 0.0
                elif j < (VPoolSize-1)/2:
                    Vpoolingconnections1[k][i][j] = 0.0
                else:
                    Vpoolingconnections2[k][i][j] = 0.0

    # Small correction for a very special case
    if numOrientations == 8 and VPoolSize == 3:
        Vpoolingconnections1[2,0,1] = 1.0
        Vpoolingconnections2[2,0,1] = 0.0

    return Vpoolingfilters, Vpoolingconnections1, Vpoolingconnections2


#########################################################
### Build orientation filters and connection patterns ###
#########################################################


# Set the orientation filters (orientation kernels, V1 and V2 layer23 pooling filters)
filters1, filters2 = createFilters(numOrientations, oriFilterSize, sigma2=0.75, Olambda=4, phi=phi)
V1PoolingFilters, V1PoolingConnections1, V1PoolingConnections2 = createPoolingConnectionsAndFilters(numOrientations, VPoolSize=V1PoolSize, sigma2=4.0,  Olambda=5, phi=phi)
V2PoolingFilters, V2PoolingConnections1, V2PoolingConnections2 = createPoolingConnectionsAndFilters(numOrientations, VPoolSize=V2PoolSize, sigma2=26.0, Olambda=9, phi=phi)

# Set up filters for filling-in stage (spreads in various directions)
numFlows = 2  # (brightness/darkness) right and down
flowFilter = [[1,0], [0,1]]  # down, right ; default is [[1,0],[0,1]]

# Specify flow orientation (all others block) and position of blockers (different subarrays are for different flow directions)
# BoundaryBlockFilter = [[[vertical, 1, 1], [vertical, 1, 0]], [[horizontal, 1, 1], [horizontal, 0, 1]]]
BoundaryBlockFilter = [[[numOrientations/2-1,1,1], [numOrientations/2-1,1,0]], [[numOrientations-1,1,1], [numOrientations-1,0,1]]]


########################################################################################################################
### Create the neuron layers ((Retina,) LGN, V1, V2, V4, Boundary and Surface Segmentation Layers) + Spike Detectors ###
########################################################################################################################


# Neural LGN cells will receive input values from LGN
LGNBright = sim.Population(ImageNumPixelRows*ImageNumPixelColumns, cellType, label="LGNBright")
LGNDark   = sim.Population(ImageNumPixelRows*ImageNumPixelColumns, cellType, label="LGNDark")

# Simple oriented neurons
V1Layer6P1 = sim.Population(numOrientations*numPixelRows*numPixelColumns, cellType)
V1Layer6P2 = sim.Population(numOrientations*numPixelRows*numPixelColumns, cellType)
V1Layer4P1 = sim.Population(numOrientations*numPixelRows*numPixelColumns, cellType)
V1Layer4P2 = sim.Population(numOrientations*numPixelRows*numPixelColumns, cellType)

# Complex cells
V1Layer23       = sim.Population(numOrientations*numPixelRows*numPixelColumns, cellType, label="V1LayerToPlot")
V1Layer23Pool   = sim.Population(numOrientations*numPixelRows*numPixelColumns, cellType)
V1Layer23Inter1 = sim.Population(numOrientations*numPixelRows*numPixelColumns, cellType)
V1Layer23Inter2 = sim.Population(numOrientations*numPixelRows*numPixelColumns, cellType)

###### All subsequent areas have multiple segmentation representations

# Area V2
V2Layer23Inter1 = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, cellType)
V2Layer23Inter2 = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, cellType)
V2Layer6        = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, cellType)
V2Layer4        = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, cellType)
V2Layer23       = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, cellType, label="V2LayerToPlot")
V2Layer23Pool   = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, cellType)

# Area V4
V4Brightness       = sim.Population(numSegmentationLayers*ImageNumPixelRows*ImageNumPixelColumns,          cellType, label="V4Brightness")
V4InterBrightness1 = sim.Population(numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns, cellType)
V4InterBrightness2 = sim.Population(numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns, cellType)
V4Darkness         = sim.Population(numSegmentationLayers*ImageNumPixelRows*ImageNumPixelColumns,          cellType, label="V4Darkness")
V4InterDarkness1   = sim.Population(numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns, cellType)
V4InterDarkness2   = sim.Population(numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns, cellType)

if numSegmentationLayers>1:
    if useSurfaceSegmentation==1:

        # Surface Segmentation cells
        SurfaceSegmentationOn        = sim.Population((numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns,          cellType, label="SurfaceSegmentationOn")
        SurfaceSegmentationOnInter1  = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, cellType)
        SurfaceSegmentationOnInter2  = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, cellType)
        SurfaceSegmentationOff       = sim.Population((numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns,          cellType, label="SurfaceSegmentationOff")
        SurfaceSegmentationOffInter1 = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, cellType)
        SurfaceSegmentationOffInter2 = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, cellType)

    if useBoundarySegmentation==1:

        # Boundary Segmentation cells
        BoundarySegmentationOn        = sim.Population((numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns,          cellType, label="BoundarySegmentationOn")
        BoundarySegmentationOnInter1  = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, cellType)
        BoundarySegmentationOnInter2  = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, cellType)
        BoundarySegmentationOnInter3  = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, cellType, label="BoundarySegmentationOnInter3")
        BoundarySegmentationOff       = sim.Population((numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns,          cellType, label="BoundarySegmentationOff")
        BoundarySegmentationOffInter1 = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, cellType)
        BoundarySegmentationOffInter2 = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, cellType)


######################################################################
### Neurons layers are defined, now set up connexions between them ###
######################################################################


synapseCount = 0

############### Area V1 ##################

oriFilterWeight = connections['LGN_ToV1Excite']
for k in range(0, numOrientations):                          # Orientations
    for i2 in range(-oriFilterSize/2, oriFilterSize/2):      # Filter rows
        for j2 in range(-oriFilterSize/2, oriFilterSize/2):  # Filter columns
            ST = []                                          # Source-Target vector containing indexes of neurons to connect within specific layers
            ST2 = []                                         # Second Source-Target vector for another connection
            for i in range(oriFilterSize/2, numPixelRows-oriFilterSize/2):         # Rows
                for j in range(oriFilterSize/2, numPixelColumns-oriFilterSize/2):  # Columns
                    if i+i2 >=0 and i+i2<ImageNumPixelRows and j+j2>=0 and j+j2<ImageNumPixelColumns:
                        # Dark inputs use reverse polarity filter
                        if abs(filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2]) > 0.1:
                            ST.append(((i+i2)*ImageNumPixelColumns + (j+j2),
                                       k*numPixelRows*numPixelColumns + i*numPixelColumns + j))
                        if abs(filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]) > 0.1:
                            ST2.append(((i+i2)*ImageNumPixelColumns + (j+j2),
                                        k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

            if len(ST)>0:
                # LGN -> Layer 6 and 4 (simple cells) connections (no connections at the edges, to avoid edge-effects) first polarity filter
                sim.Projection(LGNBright, V1Layer6P1, MyConnector(ST), sim.StaticSynapse(weight=oriFilterWeight*filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                sim.Projection(LGNDark,   V1Layer6P2, MyConnector(ST), sim.StaticSynapse(weight=oriFilterWeight*filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                sim.Projection(LGNBright, V1Layer4P1, MyConnector(ST), sim.StaticSynapse(weight=oriFilterWeight*filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                sim.Projection(LGNDark,   V1Layer4P2, MyConnector(ST), sim.StaticSynapse(weight=oriFilterWeight*filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                synapseCount += 4*len(ST)

            if len(ST2)>0:
                # LGN -> Layer 6 and 4 (simple cells) connections (no connections at the edges, to avoid edge-effects) second polarity filter
                sim.Projection(LGNBright, V1Layer6P2, MyConnector(ST2), sim.StaticSynapse(weight=oriFilterWeight*filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                sim.Projection(LGNDark,   V1Layer6P1, MyConnector(ST2), sim.StaticSynapse(weight=oriFilterWeight*filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                sim.Projection(LGNBright, V1Layer4P2, MyConnector(ST2), sim.StaticSynapse(weight=oriFilterWeight*filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                sim.Projection(LGNDark,   V1Layer4P1, MyConnector(ST2), sim.StaticSynapse(weight=oriFilterWeight*filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                synapseCount += 4*len(ST2)

# Excitatory connection from same orientation and polarity 1, input from layer 6
sim.Projection(V1Layer6P1, V1Layer4P1, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_6To4Excite']))
sim.Projection(V1Layer6P2, V1Layer4P2, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_6To4Excite']))
synapseCount += (len(V1Layer6P1)+len(V1Layer6P2))

ST = [] # Source-Target vector containing indexes of neurons to connect within specific layers
for k in range(0, numOrientations):          # Orientations
    for i in range(0, numPixelRows):         # Rows
        for j in range(0, numPixelColumns):  # Columns
            for i2 in range(-1,1):
                for j2 in range(-1,1):
                    if i2!=0 or j2!=0:
                        if i+i2 >=0 and i+i2 <numPixelRows and j+j2>=0 and j+j2<numPixelColumns:
                            ST.append((k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                       k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

# Surround inhibition from layer 6 of same orientation and polarity
sim.Projection(V1Layer6P1, V1Layer4P1, MyConnector(ST), sim.StaticSynapse(weight=connections['V1_6To4Inhib']))
sim.Projection(V1Layer6P2, V1Layer4P2, MyConnector(ST), sim.StaticSynapse(weight=connections['V1_6To4Inhib']))
synapseCount += 2*len(ST)

ST = []
ST2 = []
ST3 = []
ST4 = []
ST5 = []
ST6 = []
for k in range(0, numOrientations):                                # Orientations
    for i in range(0, numPixelRows):                               # Rows
        for j in range(0, numPixelColumns):                        # Columns
            for k2 in range(0, numOrientations):                   # Other orientations
                if k != k2:
                    ST.append((k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                               k2*numPixelRows*numPixelColumns + i*numPixelColumns + j))

            for i2 in range(-V1PoolSize/2+1, V1PoolSize/2+1):      # Filter rows (extra +1 to insure get top of odd-numbered filter)
                for j2 in range(-V1PoolSize/2+1, V1PoolSize/2+1):  # Filter columns

                    if V1PoolingFilters[k][i2+V1PoolSize/2][j2+V1PoolSize/2] > 0:
                        if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                            ST2.append((k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                        k*numPixelRows*numPixelColumns + i*numPixelColumns + j))
                            ST3.append((OppositeOrientationIndex[k]*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                        k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

                    if V1PoolingConnections1[k][i2+V1PoolSize/2][j2+V1PoolSize/2] > 0:
                        if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                            ST4.append((k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                        k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2)))

                    if V1PoolingConnections2[k][i2+V1PoolSize/2][j2+V1PoolSize/2] > 0:
                        if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                            ST5.append((k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                        k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2)))

            ST6.append((OppositeOrientationIndex[k]*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                        k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

# Layer 4 -> Layer23 (complex cell connections)
sim.Projection(V1Layer4P1, V1Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_4To23Excite']))
sim.Projection(V1Layer4P2, V1Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_4To23Excite']))
synapseCount += (len(V1Layer4P1)+len(V1Layer4P2))

# Cross-orientation inhibition
sim.Projection(V1Layer23, V1Layer23, MyConnector(ST), sim.StaticSynapse(weight=connections['V1_CrossOriInhib']))
synapseCount += len(ST)

# Pooling neurons in Layer 23 (excitation from same orientation, inhibition from orthogonal), V1PoolingFilters pools from both sides
sim.Projection(V1Layer23, V1Layer23Pool, MyConnector(ST2), sim.StaticSynapse(weight=connections['V1_ComplexExcite']))
sim.Projection(V1Layer23, V1Layer23Pool, MyConnector(ST3), sim.StaticSynapse(weight=connections['V1_ComplexInhib']))
synapseCount += (len(ST2) + len(ST3))

# Pooling neurons back to Layer 23 and to interneurons (ST4 for one side and ST5 for the other), V1PoolingConnections pools from only one side
sim.Projection(V1Layer23Pool, V1Layer23,       MyConnector(ST4), sim.StaticSynapse(weight=connections['V1_FeedbackExcite']))
sim.Projection(V1Layer23Pool, V1Layer23Inter1, MyConnector(ST4), sim.StaticSynapse(weight=connections['V1_FeedbackExcite']))
sim.Projection(V1Layer23Pool, V1Layer23Inter2, MyConnector(ST4), sim.StaticSynapse(weight=connections['V1_NegFeedbackInhib']))
sim.Projection(V1Layer23Pool, V1Layer23,       MyConnector(ST5), sim.StaticSynapse(weight=connections['V1_FeedbackExcite']))
sim.Projection(V1Layer23Pool, V1Layer23Inter2, MyConnector(ST5), sim.StaticSynapse(weight=connections['V1_FeedbackExcite']))
sim.Projection(V1Layer23Pool, V1Layer23Inter1, MyConnector(ST5), sim.StaticSynapse(weight=connections['V1_NegFeedbackInhib']))
synapseCount += 3*(len(ST4) + len(ST5))

# Connect interneurons to complex cell and each other
sim.Projection(V1Layer23Inter1, V1Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_InterInhib']))
sim.Projection(V1Layer23Inter2, V1Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_InterInhib']))
synapseCount += (len(V1Layer23Inter1) + len(V1Layer23Inter2))

# End-cutting (excitation from orthogonal interneuron)
sim.Projection(V1Layer23Inter1, V1Layer23, MyConnector(ST6), sim.StaticSynapse(weight=connections['V1_EndCutExcite']))
sim.Projection(V1Layer23Inter2, V1Layer23, MyConnector(ST6), sim.StaticSynapse(weight=connections['V1_EndCutExcite']))
synapseCount += 2*len(ST6)

# Connect Layer 23 cells to Layer 6 cells (folded feedback)
sim.Projection(V1Layer23, V1Layer6P1, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_23To6Excite']))
sim.Projection(V1Layer23, V1Layer6P2, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_23To6Excite']))
synapseCount += 2*len(V1Layer23)


############### Area V2  #################

inhibRange64=1
ST = []
ST2 = []
for h in range(0, numSegmentationLayers):        # segmentation layers
    for k in range(0, numOrientations):          # Orientations
        for i in range(0, numPixelRows):         # Rows
            for j in range(0, numPixelColumns):  # Columns
                ST.append((k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                           h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

                for i2 in range(-inhibRange64, inhibRange64+1):
                    for j2 in range(-inhibRange64, inhibRange64+1):
                        if i+i2 >=0 and i+i2 <numPixelRows and i2!=0 and j+j2 >=0 and j+j2 <numPixelColumns and j2!=0:
                            ST2.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                        h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

# V2 Layers 4 and 6 connections
sim.Projection(V1Layer23,  V2Layer6, MyConnector(ST), sim.StaticSynapse(weight=connections['V1_ToV2Excite']))
sim.Projection(V1Layer23,  V2Layer4, MyConnector(ST), sim.StaticSynapse(weight=connections['V1_ToV2Excite']))
sim.Projection(V2Layer6, V2Layer4, sim.OneToOneConnector(),   sim.StaticSynapse(weight=connections['V2_6To4Excite']))
synapseCount += (2*len(ST) + len(V2Layer6))

# Surround inhibition V2 Layer 6 -> 4
sim.Projection(V2Layer6,  V2Layer4, MyConnector(ST2), sim.StaticSynapse(weight=connections['V2_6To4Inhib']))
synapseCount += len(ST2)

ST = []
ST2 = []
ST3 = []
ST4 = []
ST5 = []
ST6 = []
for h in range(0, numSegmentationLayers):        # segmentation layers
    for k in range(0, numOrientations):          # Orientations
        for i in range(0, numPixelRows):         # Rows
            for j in range(0, numPixelColumns):  # Columns
                ST.append((h*numOrientations*numPixelRows*numPixelColumns + OppositeOrientationIndex[k]*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                           h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

                for i2 in range(-V2PoolSize/2+1, V2PoolSize/2+1):      # Filter rows (extra +1 to insure get top of odd-numbered filter)
                    for j2 in range(-V2PoolSize/2+1, V2PoolSize/2+1):  # Filter columns

                        if V2PoolingFilters[k][i2+V2PoolSize/2][j2+V2PoolSize/2] > 0:
                            if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                                ST2.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                            h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

                                for k2 in range(0, numOrientations):
                                    if k2 != k:
                                        if k2 == OppositeOrientationIndex[k]:
                                            for h2 in range(0, numSegmentationLayers):  # for all segmentation layers
                                                ST3.append((h2*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                            h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))
                                        else:
                                            ST4.append((h*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                        h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

                        if V2PoolingConnections1[k][i2+V2PoolSize/2][j2+V2PoolSize/2] > 0:
                            if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                                ST5.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                            h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2)))

                        if V2PoolingConnections2[k][i2+V2PoolSize/2][j2+V2PoolSize/2] > 0:
                            if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                                ST6.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                            h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2)))

# V2 Layer 4 -> V2 Layer23 (complex cell connections)
sim.Projection(V2Layer4, V2Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V2_4To23Excite']))
synapseCount += len(V2Layer4)

# Cross-orientation inhibition
sim.Projection(V2Layer23, V2Layer23, MyConnector(ST), sim.StaticSynapse(weight=connections['V2_CrossOriInhib']))
synapseCount += len(ST)

# Pooling neurons in V2Layer 23 (excitation from same orientation, inhibition from different + stronger for orthogonal orientation)
sim.Projection(V2Layer23, V2Layer23Pool, MyConnector(ST2), sim.StaticSynapse(weight=connections['V2_ComplexExcite']))
sim.Projection(V2Layer23, V2Layer23Pool, MyConnector(ST3), sim.StaticSynapse(weight=connections['V2_ComplexInhib']))
synapseCount += (len(ST2) + len(ST3))
if len(ST4)>0:  # non-orthogonal inhibition
    sim.Projection(V2Layer23, V2Layer23Pool, MyConnector(ST4), sim.StaticSynapse(weight=connections['V2_ComplexInhib2']))
    synapseCount += len(ST4)

# Pooling neurons back to Layer 23 and to interneurons (ST5 for one side and ST6 for the other), V2PoolingConnections pools from only one side
sim.Projection(V2Layer23Pool, V2Layer23,       MyConnector(ST5), sim.StaticSynapse(weight=connections['V2_FeedbackExcite']))
sim.Projection(V2Layer23Pool, V2Layer23Inter1, MyConnector(ST5), sim.StaticSynapse(weight=connections['V2_FeedbackExcite']))
sim.Projection(V2Layer23Pool, V2Layer23Inter2, MyConnector(ST5), sim.StaticSynapse(weight=connections['V2_NegFeedbackInhib']))
sim.Projection(V2Layer23Pool, V2Layer23,       MyConnector(ST6), sim.StaticSynapse(weight=connections['V2_FeedbackExcite']))
sim.Projection(V2Layer23Pool, V2Layer23Inter2, MyConnector(ST6), sim.StaticSynapse(weight=connections['V2_FeedbackExcite']))
sim.Projection(V2Layer23Pool, V2Layer23Inter1, MyConnector(ST6), sim.StaticSynapse(weight=connections['V2_NegFeedbackInhib']))
synapseCount += (3*len(ST5) + 3*len(ST6))

# Connect interneurons to complex cell
sim.Projection(V2Layer23Inter1, V2Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V2_InterInhib']))
sim.Projection(V2Layer23Inter2, V2Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V2_InterInhib']))
synapseCount += (len(V2Layer23Inter1) + len(V2Layer23Inter2))

# Connect Layer 23 cells to Layer 6 cells (folded feedback)
sim.Projection(V2Layer23, V2Layer6, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V2_23To6Excite']))
synapseCount += len(V2Layer23)

# # Feedback from V2 to V1 (layer 6)
# sim.Projection(V2Layer6, V1Layer6P1, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V2_ToV1FeedbackExcite']))
# sim.Projection(V2Layer6, V1Layer6P2, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V2_ToV1FeedbackExcite']))

############# Area V4 filling-in #############

ST = []
ST2 = []
ST3 = []
ST4 = []
for h in range(0, numSegmentationLayers):         # Segmentation layers
    for i in range(0, ImageNumPixelRows):         # Rows
        for j in range(0, ImageNumPixelColumns):  # Columns
            ST.append((i*ImageNumPixelColumns + j,
                       h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j))

            for k in range(0, numFlows):  # Flow directions
                ST2.append((h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                            h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j))

                # set up flow indices
                i2 = flowFilter[k][0]
                j2 = flowFilter[k][1]
                if i + i2 >= 0 and i + i2 < ImageNumPixelRows and j + j2 >= 0 and j + j2 < ImageNumPixelColumns:
                    ST3.append((h*ImageNumPixelRows*ImageNumPixelColumns + (i+i2)*ImageNumPixelColumns + (j+j2),
                                h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j))

                for k2 in range(0, len(BoundaryBlockFilter[k])):
                    for k3 in range(0, numOrientations):
                        if BoundaryBlockFilter[k][k2][0] != k3:
                            i2 = BoundaryBlockFilter[k][k2][1]
                            j2 = BoundaryBlockFilter[k][k2][2]
                            if i + i2 < numPixelRows and j + j2 < numPixelColumns:
                                ST4.append((h*numOrientations*numPixelRows*numPixelColumns + k3*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                            h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j))

# Brightness and darkness at V4 compete
sim.Projection(V4Darkness,   V4Brightness, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V4_BetweenColorsInhib']))
sim.Projection(V4Brightness, V4Darkness,   sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V4_BetweenColorsInhib']))
synapseCount += (len(V4Darkness) + len(V4Brightness))

# LGNBright->V4brightness and LGNDark->V4darkness
sim.Projection(LGNBright, V4Brightness, MyConnector(ST), sim.StaticSynapse(weight=connections['LGN_ToV4Excite']))
sim.Projection(LGNDark,   V4Darkness,   MyConnector(ST), sim.StaticSynapse(weight=connections['LGN_ToV4Excite']))
synapseCount += 2*len(ST)

# V4brightness<->Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
sim.Projection(V4Brightness,       V4InterBrightness1, MyConnector(ST2),               sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
sim.Projection(V4Brightness,       V4InterBrightness2, MyConnector(ST2),               sim.StaticSynapse(weight=connections['V4_BrightnessInhib']))
sim.Projection(V4InterBrightness1, V4Brightness,       MyConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=connections['V4_BrightnessInhib']))
sim.Projection(V4InterBrightness2, V4Brightness,       MyConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
synapseCount += 4*len(ST2)

# V4darkness<->Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
sim.Projection(V4Darkness,       V4InterDarkness1, MyConnector(ST2),               sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
sim.Projection(V4Darkness,       V4InterDarkness2, MyConnector(ST2),               sim.StaticSynapse(weight=connections['V4_BrightnessInhib']))
sim.Projection(V4InterDarkness1, V4Darkness,       MyConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=connections['V4_BrightnessInhib']))
sim.Projection(V4InterDarkness2, V4Darkness,       MyConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
synapseCount += 4*len(ST2)

# V4brightness neighbors<->Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
sim.Projection(V4Brightness,       V4InterBrightness2, MyConnector(ST3),               sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
sim.Projection(V4Brightness,       V4InterBrightness1, MyConnector(ST3),               sim.StaticSynapse(weight=connections['V4_BrightnessInhib']))
sim.Projection(V4InterBrightness2, V4Brightness,       MyConnector(numpy.fliplr(ST3)), sim.StaticSynapse(weight=connections['V4_BrightnessInhib']))
sim.Projection(V4InterBrightness1, V4Brightness,       MyConnector(numpy.fliplr(ST3)), sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
synapseCount += 4*len(ST3)

# V4darkness neighbors<->Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
sim.Projection(V4Darkness,       V4InterDarkness2, MyConnector(ST3),               sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
sim.Projection(V4Darkness,       V4InterDarkness1, MyConnector(ST3),               sim.StaticSynapse(weight=connections['V4_BrightnessInhib']))
sim.Projection(V4InterDarkness2, V4Darkness,       MyConnector(numpy.fliplr(ST3)), sim.StaticSynapse(weight=connections['V4_BrightnessInhib']))
sim.Projection(V4InterDarkness1, V4Darkness,       MyConnector(numpy.fliplr(ST3)), sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
synapseCount += 4*len(ST3)

# V2Layer23 -> V4 Interneurons (all boundaries block except for orientation of flow)
sim.Projection(V2Layer23, V4InterBrightness1, MyConnector(ST4), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
sim.Projection(V2Layer23, V4InterBrightness2, MyConnector(ST4), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
sim.Projection(V2Layer23, V4InterDarkness1,   MyConnector(ST4), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
sim.Projection(V2Layer23, V4InterDarkness2,   MyConnector(ST4), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
synapseCount += 4*len(ST4)

# Strong inhibition between segmentation layers (WHY TWICE?)
if numSegmentationLayers>1:
    ST = []
    for h in range(0, numSegmentationLayers-1):       # Num layers (not including baseline layer)
        for i in range(0, ImageNumPixelRows):         # Rows
            for j in range(0, ImageNumPixelColumns):  # Columns
                for k2 in range(0, numOrientations):
                    for h2 in range(h, numSegmentationLayers-1):
                        ST.append((h*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                   (h2+1)*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + i*numPixelColumns + j))

    # Boundaries in lower levels strongly inhibit boundaries in higher segmentation levels (lower levels can be inhibited by segmentation signals)
    sim.Projection(V2Layer23, V2Layer4, MyConnector(ST), sim.StaticSynapse(weight=connections['V2_SegmentInhib']))
    synapseCount += len(ST)

########### Surface segmentation network ############

if useSurfaceSegmentation:

    ST = []
    ST2 = []
    ST3 = []
    ST4 = []
    ST5 = []
    for h in range(0, numSegmentationLayers-1):         # Segmentation layers (not including baseline layer)
        for i in range(0, ImageNumPixelRows):           # Rows
            for j in range(0, ImageNumPixelColumns):    # Columns
                for k in range(0, numFlows):            # Flow directions
                    ST.append((h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                               h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j))

                    i2 = flowFilter[k][0]               # Vertical flow indices (surface segmentation signal flows through closed shapes)
                    j2 = flowFilter[k][1]               # Horizontal flow indices
                    if i + i2 >= 0 and i + i2 < ImageNumPixelRows and j + j2 >= 0 and j + j2 < ImageNumPixelColumns:
                        ST2.append((h*ImageNumPixelRows*ImageNumPixelColumns + (i+i2)*ImageNumPixelColumns + (j+j2),
                                    h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j))

                    for k2 in range(0, len(BoundaryBlockFilter[k])):
                        for k3 in range(0, numOrientations):
                            if BoundaryBlockFilter[k][k2][0] != k3:
                                i2 = BoundaryBlockFilter[k][k2][1]
                                j2 = BoundaryBlockFilter[k][k2][2]
                                if i + i2 < numPixelRows and j + j2 < numPixelColumns:
                                    for h2 in range(0, numSegmentationLayers):  # draw boundaries from all segmentation layers
                                        ST3.append((h2*numOrientations*numPixelRows*numPixelColumns + k3*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                    h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j))

                for k2 in range(0, numOrientations):
                    for h2 in range(h, numSegmentationLayers-1):
                        ST4.append((h*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                    (h2+1)*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + i*numPixelColumns + j))

                    for i2 in range(-2, 4):  # offset by (1,1) to reflect boundary grid is offset from surface grid
                        for j2 in range(-2, 4):
                            if i + i2 >= 0 and i + i2 < ImageNumPixelRows and j + j2 >= 0 and j + j2 < ImageNumPixelColumns:
                                for h2 in range(0, h+1):
                                    ST5.append((h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                                h2*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2)))

    # Off signals inhibit On Signals (can be separated by boundaries)
    sim.Projection(SurfaceSegmentationOff,       SurfaceSegmentationOn,  sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['S_SegmentOnOffInhib']))
    synapseCount += len(SurfaceSegmentationOff)

    # SurfaceSegmentationOn/Off <-> Interneurons ; fliplr to use the connections in the way "target indexes --> source indexes"
    sim.Projection(SurfaceSegmentationOn,        SurfaceSegmentationOnInter1,  MyConnector(ST),               sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
    sim.Projection(SurfaceSegmentationOnInter2,  SurfaceSegmentationOn,        MyConnector(numpy.fliplr(ST)), sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
    sim.Projection(SurfaceSegmentationOff,       SurfaceSegmentationOffInter1, MyConnector(ST),               sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
    sim.Projection(SurfaceSegmentationOffInter2, SurfaceSegmentationOff,       MyConnector(numpy.fliplr(ST)), sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
    synapseCount += 4*len(ST)

    # SurfaceSegmentationOn/Off <-> Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
    sim.Projection(SurfaceSegmentationOn,        SurfaceSegmentationOnInter2,  MyConnector(ST2),               sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
    sim.Projection(SurfaceSegmentationOnInter1,  SurfaceSegmentationOn,        MyConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
    sim.Projection(SurfaceSegmentationOff,       SurfaceSegmentationOffInter2, MyConnector(ST2),               sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
    sim.Projection(SurfaceSegmentationOffInter1, SurfaceSegmentationOff,       MyConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
    synapseCount += 4*len(ST2)

    # V2Layer23 -> Segmentation Interneurons (all boundaries block except for orientation of flow)
    sim.Projection(V2Layer23, SurfaceSegmentationOnInter1,  MyConnector(ST3), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
    sim.Projection(V2Layer23, SurfaceSegmentationOnInter2,  MyConnector(ST3), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
    sim.Projection(V2Layer23, SurfaceSegmentationOffInter1, MyConnector(ST3), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
    sim.Projection(V2Layer23, SurfaceSegmentationOffInter2, MyConnector(ST3), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
    synapseCount += 4*len(ST3)

    # V2Layer23 -> V2Layer4 strong inhibition (CHECK WHY THIS CONNECTION IS THERE TWICE WITH THE SAME CONNECTION PATTERN)
    # sim.Projection(V2Layer23, V2Layer4, MyConnector(ST4), sim.StaticSynapse(weight=connections['V2_SegmentInhib']))
    # synapseCount += len(ST4)

    # Segmentation -> V2Layer4 (gating) ; way for lower levels to be inhibited by higher ones : through segmentation network) SHOULDN'T IT BE ACTIVATORY????
    sim.Projection(SurfaceSegmentationOn, V2Layer4, MyConnector(ST5), sim.StaticSynapse(weight=connections['S_SegmentGatingInhib']))
    synapseCount += len(ST5)

########### Boundary segmentation network ############

if useBoundarySegmentation:

    ST = []
    ST2 = []
    ST3 = []
    ST4 = []
    for h in range(0, numSegmentationLayers-1):  # Num layers (not including baseline layer)
        for i in range(0, ImageNumPixelRows):  # Rows
            for j in range(0, ImageNumPixelColumns):  # Columns
                for k in range(0, numFlows):  # Flow directions
                    ST.append((h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                               h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j))

                    i2 = flowFilter[k][0]
                    j2 = flowFilter[k][1]
                    if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                        ST2.append((h*ImageNumPixelRows*ImageNumPixelColumns + (i+i2)*ImageNumPixelColumns + (j+j2),
                                    h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j))

                    for k2 in range(0, len(BoundaryBlockFilter[k])):
                        i2 = BoundaryBlockFilter[k][k2][1]
                        j2 = BoundaryBlockFilter[k][k2][2]
                        if i+i2 < numPixelRows and j+j2 < numPixelColumns:
                            for k3 in range(0, numOrientations):
                                for h2 in range(0, numSegmentationLayers):  # draw boundaries from all segmentation layers
                                    ST3.append((h2*numOrientations*numPixelRows*numPixelColumns + k3*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j))

                for k2 in range(0, numOrientations):
                    for i2 in range(-2, 4):  # offset by (1, 1) to reflect boundary grid is offset from surface grid
                        for j2 in range(-2, 4):
                            if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                                for h2 in range(0, h+1):
                                    ST4.append((h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                                h2*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2)))

    # BoundarySegmentationOn<->Interneurons (flow out on interneuron 1 flow in on interneuron 2)
    sim.Projection(BoundarySegmentationOn,       BoundarySegmentationOnInter1, MyConnector(ST),               sim.StaticSynapse(weight=connections['B_SegmentSpreadExcite']))
    sim.Projection(BoundarySegmentationOnInter2, BoundarySegmentationOn,       MyConnector(numpy.fliplr(ST)), sim.StaticSynapse(weight=connections['B_SegmentSpreadExcite']))
    synapseCount += 2*len(ST)

    sim.Projection(BoundarySegmentationOn,       BoundarySegmentationOnInter2, MyConnector(ST2),               sim.StaticSynapse(weight=connections['B_SegmentSpreadExcite']))
    sim.Projection(BoundarySegmentationOnInter1, BoundarySegmentationOn,       MyConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=connections['B_SegmentSpreadExcite']))
    synapseCount += 2*len(ST2)

    # Inhibition from third interneuron (itself inhibited by the presence of a boundary)
    sim.Projection(BoundarySegmentationOnInter3, BoundarySegmentationOnInter1, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['B_SegmentTonicInhib']))
    sim.Projection(BoundarySegmentationOnInter3, BoundarySegmentationOnInter2, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['B_SegmentTonicInhib']))
    synapseCount += 2*len(BoundarySegmentationOnInter3)

    # V2layer23 -> Segmentation Interneurons (all boundaries open flow by inhibiting third interneuron)
    # BoundarySegmentationOnInter3.inject(sim.DCSource(amplitude=1000.0, start=0.0, stop=0.0))
    sim.Projection(V2Layer23, BoundarySegmentationOnInter3, MyConnector(ST3), sim.StaticSynapse(weight=connections['B_SegmentOpenFlowInhib']))
    synapseCount += len(ST3)

    # BoundarySegmentation -> V2layer4 (gating)
    sim.Projection(BoundarySegmentationOn, V2Layer4, MyConnector(ST4), sim.StaticSynapse(weight=connections['B_SegmentGatingInhib']))
    synapseCount += len(ST4)


##############################################
### Create and return the needed circuitry ###
##############################################


# # Return only the populations that need to be updated during the simulation
# network = LGNBright + LGNDark \
#           + V1Layer6P1 + V1Layer6P2 + V1Layer4P1 + V1Layer4P2 + V1Layer23 + V1Layer23Pool + V1Layer23Inter1 + V1Layer23Inter2 \
#           + V2Layer6 + V2Layer4 + V2Layer23 + V2Layer23Pool + V2Layer23Inter1 + V2Layer23Inter2 \
#           + V4Brightness + V4Darkness + V4InterBrightness1 + V4InterBrightness2 + V4InterDarkness1 + V4InterDarkness2
# circuit = LGNBright + LGNDark + network.get_population("V1LayerToPlot") + network.get_population("V2LayerToPlot") + V4Brightness + V4Darkness
# if useSurfaceSegmentation:
#     network += (SurfaceSegmentationOn + SurfaceSegmentationOnInter1 + SurfaceSegmentationOnInter2
#                 + SurfaceSegmentationOff + SurfaceSegmentationOffInter1 + SurfaceSegmentationOffInter2)
#     circuit += (network.get_population("SurfaceSegmentationOn") + network.get_population("SurfaceSegmentationOff"))
# if useBoundarySegmentation:
#     network += (BoundarySegmentationOn + BoundarySegmentationOnInter1 + BoundarySegmentationOnInter2 + BoundarySegmentationOnInter3
#                     + BoundarySegmentationOff + BoundarySegmentationOffInter1 + BoundarySegmentationOffInter2)
#     circuit += BoundarySegmentationOnInter3 + BoundarySegmentationOn + BoundarySegmentationOff
