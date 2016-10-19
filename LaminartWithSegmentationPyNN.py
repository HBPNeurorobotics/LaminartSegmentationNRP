# ! /usr/bin/env python

# NEST implementation of the LAMINART model of visual cortex.
# created by Greg Francis (gfrancis@purdue.edu) as part of the Human Brain Project.
# 16 June 2014

# Notes:
# nest.Connect(source, target)
# NEST simulation time is in milliseconds

import os, sys
import matplotlib.pyplot as plt
import random
import numpy
from images2gif import writeGif
import setInput
from createFilters import createFilters, createPoolingConnectionsAndFilters
# import pyNN.nest as sim
from pyNN.utility import get_simulator, init_logging, normalized_filename
sim, options = get_simulator()

##################
### Parameters ###
##################

# How time goes
dt = 1.0                    # (ms)
synDelay = 1.0              # (ms) ??? check, car avec toutes les layers ca peut faire de la merde
stepDuration = 50.0         # (ms) time step for stimulus and segmentation signal updates
simTime = 1000.0            # (ms)
nTimeSteps = numpy.int(simTime/stepDuration)
sim.setup(timestep=dt, max_delay=synDelay)

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
    'cm'         : 0.5}     # (nF)
normalCellType = sim.IF_curr_alpha(**normalCellParams)  # any neuron in the network
tonicCellType = sim.IF_curr_alpha(**tonicCellParams)    # tonically active interneurons (inter3 in segmentation network)

# Segmentation parameters
useSDPropToDist = 0					# if 1, segmentationTargetLocationSD ~ dist(segmentationTargetLocation;fix.point)
distanceToTarget = 70				# distance between the target and the fixation point ; in pixels (TO REFINE)
startSegmentationSignal = 100		# in milliseconds (originally 100)
segmentationTargetLocationSD = 8	# standard deviation of where location actually ends up (originally 8) ; in pixels
segmentationSignalSize = 20			# even number ; circle diameter, where a segmentation signal is triggered ; in pixels

#########################################################
### Build orientation filters and connection patterns ###
#########################################################

# Set the number of segmentation layers ; one of these is the baseline layer (commonly use 3) minimum is 1
segmentationTargetLocationX, segmentationTargetLocationY = [20,0]
numSegmentationLayers = 3
UseSurfaceSegmentation = 0
UseBoundarySegmentation = 1
if numSegmentationLayers == 1:
    UseSurfaceSegmentation = 0
    UseBoundarySegmentation = 0

# Boundary coordinates (boundaries exist at positions between retinotopic coordinates, so add extra pixel on each side to insure a boundary could exists for retinal pixel)
numPixelRows = ImageNumPixelRows + 1        # height for oriented neurons (placed between un-oriented pixels)
numPixelColumns = ImageNumPixelColumns + 1  # width for oriented neurons (placed between un-oriented pixels)

# Set the orientation filters (orientation kernels, V1 and V2 layer23 pooling filters)
sys.stdout.write('\nSetting up orientation filters...\n')

OppositeOrientationIndex = list(numpy.roll(range(numOrientations), numOrientations/2))
# For numOrientations = 2, orientation indexes = [vertical, horizontal] -> opposite orientation indexes = [horizontal, vertical]
# For numOrientations = 4, orientation indexes = [ /, |, \, - ] -> opposite orientation indexes = [ \, -, \, | ]
# For numOrientations = 8, [ /h, /, /v, |, \v, \, \h, - ] -> [ \v, \, \h, -, /h, /, /v, | ] ([h,v] = [more horizontal, more vertical])

oriFilterSize = 4  # better as an even number (filtering from LGN to V1) ; 4 is original
V1PoolSize = 3     # better as an odd number (pooling in V1) ; 3 is original
V2PoolSize = 7     # better as an odd number (pooling in V2) ; 7 is original
filters1, filters2 = createFilters(numOrientations, oriFilterSize, sigma2=0.75, Olambda=4)
V1poolingfilters, V1poolingconnections1, V1poolingconnections2 = createPoolingConnectionsAndFilters(numOrientations, VPoolSize=V1PoolSize, sigma2=4.0, Olambda=5)
V2poolingfilters, V2poolingconnections1, V2poolingconnections2 = createPoolingConnectionsAndFilters(numOrientations, VPoolSize=V2PoolSize, sigma2=26.0, Olambda=9)

# Set up filters for filling-in stage (spreads in various directions).
# Interneurons receive inhib. input from all but boundary ori. that matches flow direction. Up, Right (Down and Left are defined implicitly by these)
numFlows = 2  # (brightness/darkness) right and down
flowFilter = [ [1,0], [0,1]]  # down, right

# Specify flow orientation (all others block) and position of blockers
# Different subarrays are for different flow directions
# (1 added to each offset position because boundary grid is (1,1) offset from brightness grid)
if numOrientations==2:
    BoundaryBlockFilter = [ [[0, 1, 1], [0, 1, 0]], [[1, 1, 1], [1, 0, 1]] ]
if numOrientations==4:
    BoundaryBlockFilter = [ [[1, 1, 1], [1, 1, 0]], [[3, 1, 1], [3, 0, 1]] ]
if numOrientations == 8:
    BoundaryBlockFilter = [ [[3, 1, 1], [3, 1, 0]], [[7, 1, 1], [7, 0, 1]] ]

########################################################################################################################
### Create the neuron layers ((Retina,) LGN, V1, V2, V4, Boundary and Surface Segmentation Layers) + Spike Detectors ###
########################################################################################################################

sys.stdout.write('Done. \nDefining cells...')
sys.stdout.flush()

sys.stdout.write('Input,...')
sys.stdout.flush()
# Pre-LGN layers set up as dc generators
# LGNbrightInput = nest.create("dc_generator", ImageNumPixelRows*ImageNumPixelColumns)
# LGNdarkInput =   nest.create("dc_generator", ImageNumPixelRows*ImageNumPixelColumns)
# DO SOMETHING LIKE THIS HERE BUT WITH TUNABLE CONTENT AND AN INPUT FOR EACH PIXEL
# sim.DCSource(amplitude=1000.0, start=100.0, stop=700.0)
LGNbrightInput = sim.create(normalCellType, n=ImageNumPixelRows*ImageNumPixelColumns) # temporary solution for test purpose (to change to actual input)
LGNdarkInput =   sim.create(normalCellType, n=ImageNumPixelRows*ImageNumPixelColumns) # temporary solution for test purpose (to change to actual input)

# LGN
sys.stdout.write('LGN,...')
sys.stdout.flush()

# Neural LGN cells will receive input values from LGNbrightInputs or from the retina, depending on useRetina
LGNbright = sim.create(normalCellType, n=ImageNumPixelRows*ImageNumPixelColumns)
LGNdark =   sim.create(normalCellType, n=ImageNumPixelRows*ImageNumPixelColumns)

# Area V1
sys.stdout.write('V1,...')
sys.stdout.flush()

# Simple oriented (H or V)
layer4P1 = sim.create(normalCellType, n=numOrientations*numPixelRows*numPixelColumns)
layer4P2 = sim.create(normalCellType, n=numOrientations*numPixelRows*numPixelColumns)
layer6P1 = sim.create(normalCellType, n=numOrientations*numPixelRows*numPixelColumns)
layer6P2 = sim.create(normalCellType, n=numOrientations*numPixelRows*numPixelColumns)

# Complex cells
layer23 =       sim.create(normalCellType, n=numOrientations*numPixelRows*numPixelColumns)
layer23Pool =   sim.create(normalCellType, n=numOrientations*numPixelRows*numPixelColumns)
layer23Inter1 = sim.create(normalCellType, n=numOrientations*numPixelRows*numPixelColumns)
layer23Inter2 = sim.create(normalCellType, n=numOrientations*numPixelRows*numPixelColumns)

###### All subsequent areas have multiple segmentation representations

# Area V2
sys.stdout.write('V2,...')
sys.stdout.flush()

V2layer4 =        sim.create(normalCellType, n=numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns)
V2layer6 =        sim.create(normalCellType, n=numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns)
V2layer23 =       sim.create(normalCellType, n=numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns)
V2layer23Pool =   sim.create(normalCellType, n=numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns)
V2layer23Inter1 = sim.create(normalCellType, n=numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns)
V2layer23Inter2 = sim.create(normalCellType, n=numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns)

# Area V4
sys.stdout.write('V4,...')
sys.stdout.flush()

V4brightness =       sim.create(normalCellType, n=numSegmentationLayers*ImageNumPixelRows*ImageNumPixelColumns)
V4InterBrightness1 = sim.create(normalCellType, n=numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
V4InterBrightness2 = sim.create(normalCellType, n=numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
V4darkness =         sim.create(normalCellType, n=numSegmentationLayers*ImageNumPixelRows*ImageNumPixelColumns)
V4InterDarkness1 =   sim.create(normalCellType, n=numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
V4InterDarkness2 =   sim.create(normalCellType, n=numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns)

if numSegmentationLayers>1:
    if UseSurfaceSegmentation==1:
        # Surface Segmentation cells
        sys.stdout.write('Surface,...')
        sys.stdout.flush()

        SurfaceSegmentationOn =        sim.create(normalCellType, n=(numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns)
        SurfaceSegmentationOnInter1 =  sim.create(normalCellType, n=(numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
        SurfaceSegmentationOnInter2 =  sim.create(normalCellType, n=(numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
        SurfaceSegmentationOff =       sim.create(normalCellType, n=(numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns)
        SurfaceSegmentationOffInter1 = sim.create(normalCellType, n=(numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
        SurfaceSegmentationOffInter2 = sim.create(normalCellType, n=(numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
        SurfaceSegmentationOnSignal =  sim.create(normalCellType, n=(numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns)
        SurfaceSegmentationOffSignal = sim.create(normalCellType, n=(numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns)

    if UseBoundarySegmentation==1:
        # Boundary Segmentation cells
        sys.stdout.write('Boundary,...')
        sys.stdout.flush()

        BoundarySegmentationOn =        sim.create(normalCellType, n=(numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns)
        BoundarySegmentationOnInter1 =  sim.create(normalCellType, n=(numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
        BoundarySegmentationOnInter2 =  sim.create(normalCellType, n=(numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
        BoundarySegmentationOnInter3 =  sim.create(tonicCellType,  n=(numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
        BoundarySegmentationOff =       sim.create(normalCellType, n=(numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns)
        BoundarySegmentationOffInter1 = sim.create(normalCellType, n=(numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
        BoundarySegmentationOffInter2 = sim.create(normalCellType, n=(numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns)
        BoundarySegmentationOnSignal =  sim.create(normalCellType, n=(numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns)
        BoundarySegmentationOffSignal = sim.create(normalCellType, n=(numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns)

######################################################################
### Neurons layers are defined, now set up connexions between them ###
######################################################################

############ LGN and Input ###############

inputToLGNBright = sim.Projection(LGNbrightInput, LGNbright, sim.OneToOneConnector(), sim.StaticSynapse(weight=700*weightScale))
inputToLGNDark =   sim.Projection(LGNdarkInput,   LGNdark,   sim.OneToOneConnector(), sim.StaticSynapse(weight=700*weightScale))
synapseCount = len(LGNbrightInput) + len(LGNdarkInput)

############### Area V1 ##################

sys.stdout.write('done. \nSetting up V1, Layers 4 and 6...')
sys.stdout.flush()

OfilterWeight = 400*weightScale  # originally 200, 370 too big, 300 too small, 310 too big (remove weightScale?)
for k in range(0, numOrientations):                         # Orientations
    for i2 in range(-oriFilterSize/2, oriFilterSize/2):     # Filter rows
        for j2 in range(-oriFilterSize/2, oriFilterSize/2): # Filter columns
            ST = []     # Source-Target vector containing indexes of neurons to connect within specific layers
            ST2 = []    # Second Source-Target vector for another connection
            for i in range(oriFilterSize/2, numPixelRows-oriFilterSize/2):          # Rows
                for j in range(oriFilterSize/2, numPixelColumns-oriFilterSize/2):   # Columns
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
                LGNbrightToV16P1 = sim.Projection(LGNbright, layer6P1, sim.FromListConnector(ST), sim.StaticSynapse(weight=OfilterWeight*filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                LGNdarkToV16P2 =   sim.Projection(LGNdark,   layer6P2, sim.FromListConnector(ST), sim.StaticSynapse(weight=OfilterWeight*filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                LGNbrightToV14P1 = sim.Projection(LGNbright, layer4P1, sim.FromListConnector(ST), sim.StaticSynapse(weight=OfilterWeight*filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                LGNdarkToV14P2 =   sim.Projection(LGNdark,   layer4P2, sim.FromListConnector(ST), sim.StaticSynapse(weight=OfilterWeight*filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                synapseCount += 4*len(ST)

            if len(ST2)>0:
                # LGN -> Layer 6 and 4 (simple cells) connections (no connections at the edges, to avoid edge-effects) second polarity filter
                LGNbrightToV16P2 = sim.Projection(LGNbright, layer6P2, sim.FromListConnector(ST2), sim.StaticSynapse(weight=OfilterWeight*filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                LGNdarkToV16P1 =   sim.Projection(LGNdark,   layer6P1, sim.FromListConnector(ST2), sim.StaticSynapse(weight=OfilterWeight*filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                LGNbrightToV14P2 = sim.Projection(LGNbright, layer4P2, sim.FromListConnector(ST2), sim.StaticSynapse(weight=OfilterWeight*filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                LGNdarkToV14P1 =   sim.Projection(LGNdark,   layer4P1, sim.FromListConnector(ST2), sim.StaticSynapse(weight=OfilterWeight*filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                synapseCount += 4*len(ST2)

# Excitatory connection from same orientation and polarity 1, input from layer 6
V16P1ExcitV14P1 = sim.Projection(layer6P1, layer4P1, sim.OneToOneConnector(), sim.StaticSynapse(weight=1.0*weightScale))
V16P2EcxitV14P2 = sim.Projection(layer6P2, layer4P2, sim.OneToOneConnector(), sim.StaticSynapse(weight=1.0*weightScale))
synapseCount += (len(layer6P1)+len(layer6P2))

ST = [] # Source-Target vector containing indexes of neurons to connect within specific layers
for k in range(0, numOrientations): # Orientations
    for i in range(0, numPixelRows):  # Rows
        for j in range(0, numPixelColumns):  # Columns
            for i2 in range(-1,1):
                for j2 in range(-1,1):
                    if i2!=0 or j2!=0:
                        if i+i2 >=0 and i+i2 <numPixelRows and j+j2>=0 and j+j2<numPixelColumns:
                            ST.append((k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                       k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

# Surround inhibition from layer 6 of same orientation and polarity
V16P1SurroundInhibV14P1 = sim.Projection(layer6P1, layer4P1, sim.FromListConnector(ST), sim.StaticSynapse(weight=-1.0*weightScale))
V16P2SurroundInhibV14P2 = sim.Projection(layer6P2, layer4P2, sim.FromListConnector(ST), sim.StaticSynapse(weight=-1.0*weightScale))
synapseCount += 2*len(ST)

sys.stdout.write('done. \nSetting up V1, Layers 23 and 6 (feedback)...')
sys.stdout.flush()

# Layer 4 -> Layer23 (complex cell connections)
V14P1ComplexExcitV123 = sim.Projection(layer4P1, layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=500.0*weightScale))
V14P2ComplexExcitV123 = sim.Projection(layer4P2, layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=500.0*weightScale))
synapseCount += (len(layer4P1)+len(layer4P2))

ST = []
ST2 = []
ST3 = []
ST4 = []
ST5 = []
ST6 = []
for k in range(0, numOrientations):                                 # Orientations
    for i in range(0, numPixelRows):                                # Rows
        for j in range(0, numPixelColumns):                         # Columns
            for k2 in range(0, numOrientations):                    # Other orientations
                if k != k2:
                    ST.append((k*numPixelRows*numPixelColumns + i*numPixelColumns + j, k2*numPixelRows*numPixelColumns + i*numPixelColumns + j))

            for i2 in range(-V1PoolSize/2+1, V1PoolSize/2+1):       # Filter rows (extra +1 to insure get top of odd-numbered filter)
                for j2 in range(-V1PoolSize/2+1, V1PoolSize/2+1):   # Filter columns

                    if V1poolingfilters[k][i2+V1PoolSize/2][j2+V1PoolSize/2] > 0:
                        if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                            ST2.append((k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                        k*numPixelRows*numPixelColumns + i*numPixelColumns + j))
                            ST3.append((OppositeOrientationIndex[k]*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                        k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

                    if V1poolingconnections1[k][i2+V1PoolSize/2][j2+V1PoolSize/2] > 0:
                        if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                            ST4.append((k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                        k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2)))

                    if V1poolingconnections2[k][i2+V1PoolSize/2][j2+V1PoolSize/2] > 0:
                        if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                            ST5.append((k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                        k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2)))

            ST6.append((OppositeOrientationIndex[k]*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                        k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

# Cross-orientation inhibition
V123CrossOriInh = sim.Projection(layer23, layer23, sim.FromListConnector(ST), sim.StaticSynapse(weight=-1000.0*weightScale))
synapseCount += len(ST)

# Pooling neurons in Layer 23 (excitation from same orientation, inhibition from orthogonal), V1poolingfilters pools from both sides
V123PoolingExc = sim.Projection(layer23, layer23Pool, sim.FromListConnector(ST2), sim.StaticSynapse(weight= 500.0*weightScale))
V123PoolingInh = sim.Projection(layer23, layer23Pool, sim.FromListConnector(ST3), sim.StaticSynapse(weight=-500.0*weightScale))
synapseCount += (len(ST2) + len(ST3))

# Pooling neurons back to Layer 23 and to interneurons (ST4 for one side and ST5 for the other), V1poolingconnections pools from only one side
V123PoolBackExcR =      sim.Projection(layer23Pool, layer23,       sim.FromListConnector(ST4), sim.StaticSynapse(weight=  500.0*weightScale))
V123PoolBackInter1Exc = sim.Projection(layer23Pool, layer23Inter1, sim.FromListConnector(ST4), sim.StaticSynapse(weight=  500.0*weightScale))
V123PoolBackInter2Inh = sim.Projection(layer23Pool, layer23Inter2, sim.FromListConnector(ST4), sim.StaticSynapse(weight=-1500.0*weightScale))
V123PoolBackExcL =      sim.Projection(layer23Pool, layer23,       sim.FromListConnector(ST5), sim.StaticSynapse(weight=  500.0*weightScale))
V123PoolBackInter2Exc = sim.Projection(layer23Pool, layer23Inter2, sim.FromListConnector(ST5), sim.StaticSynapse(weight=  500.0*weightScale))
V123PoolBackInter1Inh = sim.Projection(layer23Pool, layer23Inter1, sim.FromListConnector(ST5), sim.StaticSynapse(weight=-1500.0*weightScale))
synapseCount += 3*(len(ST4) + len(ST5))

# Connect interneurons to complex cell and each other
V123Inter1ToV123 = sim.Projection(layer23Inter1, layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=-1500.0*weightScale))
V123Inter2ToV123 = sim.Projection(layer23Inter2, layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=-1500.0*weightScale))
synapseCount += (len(layer23Inter1) + len(layer23Inter2))

# End-cutting (excitation from orthogonal interneuron)
V123Inter1EndCutExcit =  sim.Projection(layer23Inter1, layer23, sim.FromListConnector(ST6), sim.StaticSynapse(weight= 1500.0*weightScale))
V123Inter2EndCutExcit =  sim.Projection(layer23Inter1, layer23, sim.FromListConnector(ST6), sim.StaticSynapse(weight= 1500.0*weightScale))
synapseCount += 2*len(ST6)

# Connect Layer 23 cells to Layer 6 cells (folded feedback)
V123FoldedFeedbackV16P1 = sim.Projection(layer23, layer6P1, sim.OneToOneConnector(), sim.StaticSynapse(weight=100.0*weightScale))
V123FoldedFeedbackV16P2 = sim.Projection(layer23, layer6P2, sim.OneToOneConnector(), sim.StaticSynapse(weight=100.0*weightScale))
synapseCount += 2*len(layer23)


############### Area V2  #################

sys.stdout.write('done. \nSetting up V2, Layers 4 and 6...')
sys.stdout.flush()

inhibRange64=1
ST = []
ST2 = []
for h in range(0, numSegmentationLayers):        # segmentation layers
    for k in range(0, numOrientations):          # Orientations
        for i in range(0, numPixelRows):         # Rows
            for j in range(0, numPixelColumns):  # Columns
                ST.append((k*numPixelRows*numPixelColumns + i*numPixelColumns + j, h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

                for i2 in range(-inhibRange64, inhibRange64+1):
                    for j2 in range(-inhibRange64, inhibRange64+1):
                        if i+i2 >=0 and i+i2 <numPixelRows and i2!=0 and j+j2 >=0 and j+j2 <numPixelColumns and j2!=0:
                            ST2.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                        h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

# V2 Layers 4 and 6 connections
V123ToV26 = sim.Projection(layer23,  V2layer6, sim.FromListConnector(ST), sim.StaticSynapse(weight=10000.0*weightScale))
V123ToV24 = sim.Projection(layer23,  V2layer4, sim.FromListConnector(ST), sim.StaticSynapse(weight=10000.0*weightScale))
V26ToV24 =  sim.Projection(V2layer6, V2layer4, sim.OneToOneConnector(),   sim.StaticSynapse(weight=    1.0*weightScale))
synapseCount += (2*len(ST) + len(V2layer6))

# Surround inhibition V2 Layer 6 -> 4
V26SurroundInhibV24 = sim.Projection(V2layer6,  V2layer4, sim.FromListConnector(ST2), sim.StaticSynapse(weight=-20.0*weightScale))
synapseCount += len(ST2)

sys.stdout.write('done. \nSetting up V2, Layers 23 and 6 (feedback)...')
sys.stdout.flush()

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

                        if V2poolingfilters[k][i2+V2PoolSize/2][j2+V2PoolSize/2] > 0:
                            if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                                ST2.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                            h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

                                for k2 in range(0, numOrientations):
                                    if k2 != k:
                                        if k2 == OppositeOrientationIndex[k]:
                                            ST3.append((h*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                        h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))
                                        else:
                                            ST4.append((h*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                        h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

                        if V2poolingconnections1[k][i2+V2PoolSize/2][j2+V2PoolSize/2] > 0:
                            if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                                ST5.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                            h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2)))

                        if V2poolingconnections2[k][i2+V2PoolSize/2][j2+V2PoolSize/2] > 0:
                            if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                                ST6.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                            h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2)))

# V2 Layer 4 -> V2 Layer23 (complex cell connections)
V24ToV223 = sim.Projection(V2layer4, V2layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=500.0*weightScale))
synapseCount += len(V2layer4)

# Cross-orientation inhibition
V223CrossOriInh = sim.Projection(V2layer23, V2layer23, sim.FromListConnector(ST), sim.StaticSynapse(weight=-1200.0*weightScale))
synapseCount += len(ST)

# Pooling neurons in V2Layer 23 (excitation from same orientation, inhibition from orthogonal + stronger for opposite orientation)
V223PoolingExc =     sim.Projection(V2layer23, V2layer23Pool, sim.FromListConnector(ST2), sim.StaticSynapse(weight=  500.0*weightScale))
V223PoolingInhOpp =  sim.Projection(V2layer23, V2layer23Pool, sim.FromListConnector(ST3), sim.StaticSynapse(weight=-1000.0*weightScale))
synapseCount += (len(ST2) + len(ST3))
if len(ST4)>0:
    V223PoolingInh = sim.Projection(V2layer23, V2layer23Pool, sim.FromListConnector(ST4), sim.StaticSynapse(weight=  100.0*weightScale))
    synapseCount += len(ST4)

# Pooling neurons back to Layer 23 and to interneurons (ST5 for one side and ST6 for the other), V1poolingconnections pools from only one side
V223PoolBackExcR =      sim.Projection(V2layer23Pool, V2layer23,       sim.FromListConnector(ST5), sim.StaticSynapse(weight= 500.0*weightScale))
V223PoolBackInter1Exc = sim.Projection(V2layer23Pool, V2layer23Inter1, sim.FromListConnector(ST5), sim.StaticSynapse(weight= 500.0*weightScale))
V223PoolBackInter2Inh = sim.Projection(V2layer23Pool, V2layer23Inter2, sim.FromListConnector(ST5), sim.StaticSynapse(weight=-800.0*weightScale))
V223PoolBackExcL =      sim.Projection(V2layer23Pool, V2layer23,       sim.FromListConnector(ST6), sim.StaticSynapse(weight= 500.0*weightScale))
V223PoolBackInter2Exc = sim.Projection(V2layer23Pool, V2layer23Inter2, sim.FromListConnector(ST6), sim.StaticSynapse(weight= 500.0*weightScale))
V223PoolBackInter1Inh = sim.Projection(V2layer23Pool, V2layer23Inter1, sim.FromListConnector(ST6), sim.StaticSynapse(weight=-800.0*weightScale))
synapseCount += (3*len(ST5) + 3*len(ST6))

# Connect interneurons to complex cell
V2Inter1ToV223 = sim.Projection(V2layer23Inter1, V2layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=-1500.0*weightScale))
V2Inter2ToV223 = sim.Projection(V2layer23Inter2, V2layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=-1500.0*weightScale))
synapseCount += (len(V2layer23Inter1) + len(V2layer23Inter2))

# Connect Layer 23 cells to Layer 6 cells (folded feedback)
V223FoldedFeedbackV26 = sim.Projection(V2layer23, V2layer6, sim.OneToOneConnector(), sim.StaticSynapse(weight=100.0*weightScale))
synapseCount += len(V2layer23)

############# Area V4 filling-in #############

sys.stdout.write('done. \nSetting up V4...')
sys.stdout.flush()

ST = []
ST2 = []
ST3 = []
ST4 = []
for h in range(0, numSegmentationLayers):  # Segmentation layers
    for i in range(0, ImageNumPixelRows):  # Rows
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
V4DarkToBrightInhib = sim.Projection(V4darkness,   V4brightness, sim.OneToOneConnector(), sim.StaticSynapse(weight=-5000.0*weightScale))
V4BrightToDarkInhib = sim.Projection(V4brightness, V4darkness,   sim.OneToOneConnector(), sim.StaticSynapse(weight=-5000.0*weightScale))
synapseCount += (len(V4darkness) + len(V4brightness))

# LGNbright->V4brightness and LGNdark->V4darkness
LGNBrightToV4Bright = sim.Projection(LGNbright, V4brightness, sim.FromListConnector(ST), sim.StaticSynapse(weight=280.0*weightScale))
LGNDarkToV4Dark =     sim.Projection(LGNdark,   V4darkness,   sim.FromListConnector(ST), sim.StaticSynapse(weight=280.0*weightScale))
synapseCount += 2*len(ST)

# V4brightness<->Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
V4BrightInterExcite = sim.Projection(V4brightness,       V4InterBrightness1, sim.FromListConnector(ST2),               sim.StaticSynapse(weight= 2000.0*weightScale))
V4BrightInterInhib =  sim.Projection(V4brightness,       V4InterBrightness2, sim.FromListConnector(ST2),               sim.StaticSynapse(weight=-2000.0*weightScale))
V4InterBrightInhib =  sim.Projection(V4InterBrightness1, V4brightness,       sim.FromListConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=-2000.0*weightScale))
V4InterBrightExcite = sim.Projection(V4InterBrightness2, V4brightness,       sim.FromListConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight= 2000.0*weightScale))
synapseCount += 4*len(ST2)

# V4darkness<->Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
V4DarkInterExcite = sim.Projection(V4darkness,       V4InterDarkness1, sim.FromListConnector(ST2),               sim.StaticSynapse(weight= 2000.0*weightScale))
V4DarkInterInhib =  sim.Projection(V4darkness,       V4InterDarkness2, sim.FromListConnector(ST2),               sim.StaticSynapse(weight=-2000.0*weightScale))
V4InterDarkInhib =  sim.Projection(V4InterDarkness1, V4darkness,       sim.FromListConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=-2000.0*weightScale))
V4InterDarkExcite = sim.Projection(V4InterDarkness2, V4darkness,       sim.FromListConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight= 2000.0*weightScale))
synapseCount += 4*len(ST2)

# V4brightness neighbors<->Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
V4BrightNeighborInterExcite = sim.Projection(V4brightness,       V4InterBrightness2, sim.FromListConnector(ST3),               sim.StaticSynapse(weight= 2000.0*weightScale))
V4BrightNeighborInterInhib =  sim.Projection(V4brightness,       V4InterBrightness1, sim.FromListConnector(ST3),               sim.StaticSynapse(weight=-2000.0*weightScale))
V4InterBrightNeighborInhib =  sim.Projection(V4InterBrightness2, V4brightness,       sim.FromListConnector(numpy.fliplr(ST3)), sim.StaticSynapse(weight=-2000.0*weightScale))
V4InterBrightNeighborExcite = sim.Projection(V4InterBrightness1, V4brightness,       sim.FromListConnector(numpy.fliplr(ST3)), sim.StaticSynapse(weight= 2000.0*weightScale))
synapseCount += 4*len(ST3)

# V4darkness neighbors<->Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
V4DarkNeighborInterExcite = sim.Projection(V4darkness,       V4InterDarkness2, sim.FromListConnector(ST3),               sim.StaticSynapse(weight= 2000.0*weightScale))
V4DarkNeighborInterInhib =  sim.Projection(V4darkness,       V4InterDarkness1, sim.FromListConnector(ST3),               sim.StaticSynapse(weight=-2000.0*weightScale))
V4InterDarkNeighborInhib =  sim.Projection(V4InterDarkness2, V4darkness,       sim.FromListConnector(numpy.fliplr(ST3)), sim.StaticSynapse(weight=-2000.0*weightScale))
V4InterDarkNeighborExcite = sim.Projection(V4InterDarkness1, V4darkness,       sim.FromListConnector(numpy.fliplr(ST3)), sim.StaticSynapse(weight= 2000.0*weightScale))
synapseCount += 4*len(ST3)

# V2layer23 -> V4 Interneurons (all boundaries block except for orientation of flow)
V223BoundInhibV4InterBright1 = sim.Projection(V2layer23, V4InterBrightness1, sim.FromListConnector(ST4), sim.StaticSynapse(weight=-5000.0*weightScale))
V223BoundInhibV4InterBright2 = sim.Projection(V2layer23, V4InterBrightness2, sim.FromListConnector(ST4), sim.StaticSynapse(weight=-5000.0*weightScale))
V223BoundInhibV4InterDark1 =   sim.Projection(V2layer23, V4InterDarkness1,   sim.FromListConnector(ST4), sim.StaticSynapse(weight=-5000.0*weightScale))
V223BoundInhibV4InterDark2 =   sim.Projection(V2layer23, V4InterDarkness2,   sim.FromListConnector(ST4), sim.StaticSynapse(weight=-5000.0*weightScale))
synapseCount += 4*len(ST4)

# Strong inhibition between segmentation layers
if numSegmentationLayers>1:
    ST = []
    for h in range(0, numSegmentationLayers-1):  # Num layers (not including baseline layer)
        for i in range(0, ImageNumPixelRows):  # Rows
            for j in range(0, ImageNumPixelColumns):  # Columns
                for k2 in range(0, numOrientations):
                    for h2 in range(h, numSegmentationLayers-1):
                        ST.append((h*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                   (h2+1)*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + i*numPixelColumns + j))

    # Boundaries in lower levels strongly inhibit boundaries in higher segmentation levels (lower levels can be inhibited by segmentation signals)
    V2InterSegmentInhib = sim.Projection(V2layer23, V2layer4, sim.FromListConnector(ST), sim.StaticSynapse(weight=-20000.0*weightScale))
    synapseCount += len(ST)

########### Surface segmentation network ############

if numSegmentationLayers>1 and UseSurfaceSegmentation==1:
    sys.stdout.write('done. \nSetting up surface segmentation network...')
    sys.stdout.flush()

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

    # Input from segmentation signals and Off signals inhibit On Signals (can be separated by boundaries)
    SurfSegmOnSignalInput =  sim.Projection(SurfaceSegmentationOnSignal,  SurfaceSegmentationOn,  sim.OneToOneConnector(), sim.StaticSynapse(weight=    1.0*weightScale))
    SurfSegmOffSignalInput = sim.Projection(SurfaceSegmentationOffSignal, SurfaceSegmentationOff, sim.OneToOneConnector(), sim.StaticSynapse(weight=    1.0*weightScale))
    SurfSegmOffInhibOn =     sim.Projection(SurfaceSegmentationOff,       SurfaceSegmentationOn,  sim.OneToOneConnector(), sim.StaticSynapse(weight=-5000.0*weightScale))
    synapseCount += (len(SurfaceSegmentationOnSignal) + len(SurfaceSegmentationOffSignal) + len(SurfaceSegmentationOff))

    # SurfaceSegmentationOn/Off <-> Interneurons ; fliplr to use the connections in the way "target indexes --> source indexes"
    SurfSegmOnToInter1 =  sim.Projection(SurfaceSegmentationOn,        SurfaceSegmentationOnInter1,  sim.FromListConnector(ST),               sim.StaticSynapse(weight=1000.0*weightScale))
    Inter2ToSurfSegmOn =  sim.Projection(SurfaceSegmentationOnInter2,  SurfaceSegmentationOn,        sim.FromListConnector(numpy.fliplr(ST)), sim.StaticSynapse(weight=1000.0*weightScale))
    SurfSegmOffToInter1 = sim.Projection(SurfaceSegmentationOff,       SurfaceSegmentationOffInter1, sim.FromListConnector(ST),               sim.StaticSynapse(weight=1000.0*weightScale))
    Inter2ToSurfSegmOff = sim.Projection(SurfaceSegmentationOffInter2, SurfaceSegmentationOff,       sim.FromListConnector(numpy.fliplr(ST)), sim.StaticSynapse(weight=1000.0*weightScale))
    synapseCount += 4*len(ST)

    # Mutual inhibition of interneurons
    SurfSegmOnInterInhib1To2 =  sim.Projection(SurfaceSegmentationOnInter1,  SurfaceSegmentationOnInter2,  sim.OneToOneConnector(), sim.StaticSynapse(weight=-200.0*weightScale))
    SurfSegmOnInterInhib2To1 =  sim.Projection(SurfaceSegmentationOnInter2,  SurfaceSegmentationOnInter1,  sim.OneToOneConnector(), sim.StaticSynapse(weight=-200.0*weightScale))
    SurfSegmOffInterInhib1To2 = sim.Projection(SurfaceSegmentationOffInter1, SurfaceSegmentationOffInter2, sim.OneToOneConnector(), sim.StaticSynapse(weight=-200.0*weightScale))
    SurfSegmOffInterInhib2To1 = sim.Projection(SurfaceSegmentationOffInter2, SurfaceSegmentationOffInter1, sim.OneToOneConnector(), sim.StaticSynapse(weight=-200.0*weightScale))
    synapseCount += (len(SurfaceSegmentationOffInter1) + len(SurfaceSegmentationOffInter2) + len(SurfaceSegmentationOnInter1) + len(SurfaceSegmentationOnInter2))

    # SurfaceSegmentationOn/Off <-> Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
    SurfSegmOnToInter2 =  sim.Projection(SurfaceSegmentationOn,        SurfaceSegmentationOnInter2,  sim.FromListConnector(ST2),               sim.StaticSynapse(weight=1000.0*weightScale))
    Inter1ToSurfSegmOn =  sim.Projection(SurfaceSegmentationOnInter1,  SurfaceSegmentationOn,        sim.FromListConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=1000.0*weightScale))
    SurfSegmOffToInter2 = sim.Projection(SurfaceSegmentationOff,       SurfaceSegmentationOffInter2, sim.FromListConnector(ST2),               sim.StaticSynapse(weight=1000.0*weightScale))
    Inter1ToSurfSegmOff = sim.Projection(SurfaceSegmentationOffInter1, SurfaceSegmentationOff,       sim.FromListConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=1000.0*weightScale))
    synapseCount += 4*len(ST2)

    # V2layer23 -> Segmentation Interneurons (all boundaries block except for orientation of flow)
    V223ToSurfSegmOnInter1 =  sim.Projection(V2layer23, SurfaceSegmentationOnInter1,  sim.FromListConnector(ST3), sim.StaticSynapse(weight=-5000.0*weightScale))
    V223ToSurfSegmOnInter2 =  sim.Projection(V2layer23, SurfaceSegmentationOnInter2,  sim.FromListConnector(ST3), sim.StaticSynapse(weight=-5000.0*weightScale))
    V223ToSurfSegmOffInter1 = sim.Projection(V2layer23, SurfaceSegmentationOffInter1, sim.FromListConnector(ST3), sim.StaticSynapse(weight=-5000.0*weightScale))
    V223ToSurfSegmOffInter2 = sim.Projection(V2layer23, SurfaceSegmentationOffInter2, sim.FromListConnector(ST3), sim.StaticSynapse(weight=-5000.0*weightScale))
    synapseCount += 4*len(ST3)

    # V2layer23 -> V2layer4 strong inhibition (check why it is there twice, regarding connection matrices)
    V223InhibV24 = sim.Projection(V2layer23, V2layer4, sim.FromListConnector(ST4), sim.StaticSynapse(weight=-20000.0*weightScale))
    synapseCount += len(ST4)

    # Segmentation -> V2layer4 (gating) ; way for lower levels to be inhibited by higher ones : through segmentation network)
    SurfSegmOnInhibV24 = sim.Projection(SurfaceSegmentationOn, V2layer4, sim.FromListConnector(ST5), sim.StaticSynapse(weight=-5000.0*weightScale))
    synapseCount += len(ST5)

########### Boundary segmentation network ############

if numSegmentationLayers>1 and UseBoundarySegmentation==1:
    sys.stdout.write('done. \nSetting up boundary segmentation network...')
    sys.stdout.flush()

    ST = []
    ST2 = []
    ST3 = []
    ST4 = []
    for h in range(0, numSegmentationLayers-1):         # Num layers (not including baseline layer)
        for i in range(0, ImageNumPixelRows):           # Rows
            for j in range(0, ImageNumPixelColumns):    # Columns
                for k in range(0, numFlows):            # Flow directions
                    ST.append((h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                               h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j))

                    i2 = flowFilter[k][0]               # Vertical flow indices (boundary segmentation signal flows along connected boundaries)
                    j2 = flowFilter[k][1]               # Horizontal flow indices
                    if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                        ST2.append((h*ImageNumPixelRows*ImageNumPixelColumns + (i+i2)*ImageNumPixelColumns + (j+j2),
                                    h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j))

                    for k2 in range(0, len(BoundaryBlockFilter[k])): # if BoundaryBlockFilter[k][k2][0] != k:
                        i2 = BoundaryBlockFilter[k][k2][1]
                        j2 = BoundaryBlockFilter[k][k2][2]
                        if i+i2 < numPixelRows and j+j2 < numPixelColumns:
                            for k3 in range(0, numOrientations):
                                for h2 in range(0, numSegmentationLayers): # draw boundaries from all segmentation layers
                                    ST3.append((h2*numOrientations*numPixelRows*numPixelColumns + k3*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j))

                for k2 in range(0, numOrientations):
                    for i2 in range(-2, 4):              # Offset by (1,1) to reflect boundary grid is offset from surface grid
                        for j2 in range(-2, 4):
                            if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                                for h2 in range(0, h+1): # Segmentation spreading inhibits boundaries at lower levels
                                    ST4.append((h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                                h2*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2)))

    # Input from segmentation signals /// and Off signals inhibit On Signals (removed) ///
    BoundSegmOnSignalInput =  sim.Projection(BoundarySegmentationOnSignal,  BoundarySegmentationOn,  sim.OneToOneConnector(), sim.StaticSynapse(weight=0.5*weightScale))
    BoundSegmOffSignalInput = sim.Projection(BoundarySegmentationOffSignal, BoundarySegmentationOff, sim.OneToOneConnector(), sim.StaticSynapse(weight=0.5*weightScale))
    # BoundSegmOffInhibOn =   sim.Projection(BoundarySegmentationOff,       BoundarySegmentationOn,  sim.OneToOneConnector(), sim.StaticSynapse(weight=-5000.0*weightScale))
    synapseCount += (len(BoundarySegmentationOnSignal) + len(BoundarySegmentationOffSignal))  # + len(BoundarySegmentationOff))

    # BoundarySegmentationOn<->Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
    BoundSegmOnToInter1 =  sim.Projection(BoundarySegmentationOn,       BoundarySegmentationOnInter1, sim.FromListConnector(ST),               sim.StaticSynapse(weight=2000.0*weightScale))
    Inter2ToBoundSegmOn =  sim.Projection(BoundarySegmentationOnInter2, BoundarySegmentationOn,       sim.FromListConnector(numpy.fliplr(ST)), sim.StaticSynapse(weight=2000.0*weightScale))
    synapseCount += 2*len(ST)

    # Mutual inhibition of interneurons (may not need this: flow only when Inter3 is inhibited - 19 Dec 2014)
    BoundSegmOnInterInhib1To2 = sim.Projection(BoundarySegmentationOnInter1, BoundarySegmentationOnInter2, sim.OneToOneConnector(), sim.StaticSynapse(weight=-200.0*weightScale))
    BoundSegmOnInterInhib2To1 = sim.Projection(BoundarySegmentationOnInter2, BoundarySegmentationOnInter1, sim.OneToOneConnector(), sim.StaticSynapse(weight=-200.0*weightScale))
    synapseCount += 2*len(BoundarySegmentationOnInter1)

    # Inhibition from third interneuron (itself inhibited by the presence of a boundary)
    BoundSegmOnInter3Inhib1Inter1 = sim.Projection(BoundarySegmentationOnInter3, BoundarySegmentationOnInter1, sim.OneToOneConnector(), sim.StaticSynapse(weight=-20000.0*weightScale))
    BoundSegmOnInter3Inhib1Inter2 = sim.Projection(BoundarySegmentationOnInter3, BoundarySegmentationOnInter2, sim.OneToOneConnector(), sim.StaticSynapse(weight=-20000.0*weightScale))
    synapseCount += 2*len(BoundarySegmentationOnInter3)

    # SurfaceSegmentationOn <-> Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
    BoundSegmOnToInter2 = sim.Projection(BoundarySegmentationOn,       BoundarySegmentationOnInter2, sim.FromListConnector(ST2),               sim.StaticSynapse(weight=2000.0*weightScale))
    Inter1ToBoundSegmOn = sim.Projection(BoundarySegmentationOnInter1, BoundarySegmentationOn,       sim.FromListConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=2000.0*weightScale))
    synapseCount += 2*len(ST2)

    # V2layer23 -> Segmentation Interneurons (all boundaries open flow by inhibiting third interneuron)
    V223ToBoundSegmOnInter3 =  sim.Projection(V2layer23, BoundarySegmentationOnInter3,  sim.FromListConnector(ST3), sim.StaticSynapse(weight=-150.0*weightScale))
    synapseCount += len(ST3)

    # BoundarySegmentation -> V2layer4 (gating)
    BoundSegmOnInhibV24 = sim.Projection(BoundarySegmentationOn, V2layer4, sim.FromListConnector(ST4), sim.StaticSynapse(weight=-5000.0*weightScale))
    synapseCount += len(ST4)

sys.stdout.write('done. \n'+str(synapseCount)+' network connections created.\n')
sys.stdout.flush()

########################################################################################
### Network is defined, now set up stimulus, segmentation signal and run everything! ###
########################################################################################

for timeStep in range(nTimeSteps):

    # Set LGN input, using the image input
    brightPulse = sim.DCSource(start=timeStep*stepDuration, stop=(timeStep+1)*stepDuration) # bright input
    darkPulse =   sim.DCSource(start=timeStep*stepDuration, stop=(timeStep+1)*stepDuration) # dark input
    # input = getInputFromNRPRetinaModel(timeStep) # something like this will happen in the NRP and input would be a 2D numpy array
    for i in range(ImageNumPixelRows):
        for j in range(ImageNumPixelColumns):
            brightPulse.set_parameters(amplitude=input[i,j])
            darkPulse.set_parameters(amplitude=(254.0-input[i,j]))
            LGNbrightInput[i*ImageNumPixelColumns + j].inject(brightPulse)
            LGNdarkInput[i*ImageNumPixelColumns + j].inject(darkPulse)

    # Pick central locations of segmentation signals from a gaussian distribution
    segmentationInput = sim.DCSource(amplitude=1000.0, start=timeStep*stepDuration, stop=(timeStep+1)*stepDuration)
    if useSDPropToDist:
        segmentationTargetLocationSDRight = segmentationTargetLocationSD * numpy.sqrt((distanceToTarget + segmentationTargetLocationX)**2 + segmentationTargetLocationY**2)/distanceToTarget
        segmentationTargetLocationSDLeft =  segmentationTargetLocationSD * numpy.sqrt((distanceToTarget - segmentationTargetLocationX)**2 + segmentationTargetLocationY**2)/distanceToTarget
    else:
        segmentationTargetLocationSDRight = segmentationTargetLocationSDLeft = segmentationTargetLocationSD
    print "SegmentationTargetLocationSDRight: " + str(segmentationTargetLocationSDRight)
    print "SegmentationTargetLocationSDLeft: " +  str(segmentationTargetLocationSDLeft)
    adjustLocationXLeft =  int(round(random.gauss(segmentationTargetLocationX, segmentationTargetLocationSDLeft)))
    adjustLocationYLeft =  int(round(random.gauss(segmentationTargetLocationY, segmentationTargetLocationSDLeft)))
    adjustLocationXRight = int(round(random.gauss(segmentationTargetLocationX, segmentationTargetLocationSDRight)))
    adjustLocationYRight = int(round(random.gauss(segmentationTargetLocationY, segmentationTargetLocationSDRight)))

    # Define surface segmentation signals (gives local DC inputs to surface and boundary segmentation networks)
    if UseSurfaceSegmentation==1:
        # Left side On segmentaion signal + Off segmentation signal
        segmentLocationX = ImageNumPixelColumns/2 - (adjustLocationXLeft-1)
        segmentLocationY = ImageNumPixelRows/2 - adjustLocationYLeft
        sys.stdout.write('\Left center surface segment signal = '+str(segmentLocationX)+", "+str(segmentLocationY))
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
    if UseBoundarySegmentation==1:
        # Left side
        segmentLocationX = ImageNumPixelColumns/2 - (adjustLocationXLeft-1)
        segmentLocationY = ImageNumPixelRows/2 - adjustLocationYLeft
        sys.stdout.write('\nLeft center boundary segment signal = '+str(segmentLocationX)+', '+str(segmentLocationY))
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