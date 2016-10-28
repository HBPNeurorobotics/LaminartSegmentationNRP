# ! /usr/bin/env python

# NEST implementation of the LAMINART model of visual cortex.
# created by Greg Francis (gfrancis@purdue.edu) as part of the Human Brain Project.
# 16 June 2014

# Notes:
# nest.Connect(source, target)
# NEST simulation time is in milliseconds

import os, sys
import matplotlib.pyplot as plt
import numpy
from createFilters import createFilters, createPoolingConnectionsAndFilters

# Function that builds all the layers of the network and connect them according to the Laminart model, enhanced with segmentation ; returns the full network and all the connections
def buildNetworkAndConnections(sim, ImageNumPixelRows, ImageNumPixelColumns, numOrientations, oriFilterSize, V1PoolSize, V2PoolSize, connections,
                               normalCellType, tonicCellType, numSegmentationLayers, useBoundarySegmentation, useSurfaceSegmentation):

    #########################################################
    ### Build orientation filters and connection patterns ###
    #########################################################

    sys.stdout.write('\nSetting up orientation filters...\n')

    # Boundary coordinates (boundaries exist at positions between retinotopic coordinates, so add extra pixel on each side to insure a boundary could exists for retinal pixel)
    numPixelRows = ImageNumPixelRows + 1        # height for oriented neurons (placed between un-oriented pixels)
    numPixelColumns = ImageNumPixelColumns + 1  # width for oriented neurons (placed between un-oriented pixels)

    # Set the orientation filters (orientation kernels, V1 and V2 layer23 pooling filters)
    filters1, filters2 = createFilters(numOrientations, oriFilterSize, sigma2=0.75, Olambda=4)

    print filters1

    V1poolingfilters, V1poolingconnections1, V1poolingconnections2 = createPoolingConnectionsAndFilters(numOrientations, VPoolSize=V1PoolSize, sigma2=4.0, Olambda=5)
    V2poolingfilters, V2poolingconnections1, V2poolingconnections2 = createPoolingConnectionsAndFilters(numOrientations, VPoolSize=V2PoolSize, sigma2=26.0, Olambda=9)

    OppositeOrientationIndex = list(numpy.roll(range(numOrientations), numOrientations/2))
    # For numOrientations = 2, orientation indexes = [vertical, horizontal] -> opposite orientation indexes = [horizontal, vertical]
    # For numOrientations = 4, orientation indexes = [ /, |, \, - ] -> opposite orientation indexes = [ \, -, \, | ]
    # For numOrientations = 8, [ /h, /, /v, |, \v, \, \h, - ] -> [ \v, \, \h, -, /h, /, /v, | ] ([h,v] = [more horizontal, more vertical])

    # Set up filters for filling-in stage (spreads in various directions).
    # Interneurons receive inhib. input from all but boundary ori. that matches flow direction. Up, Right (Down and Left are defined implicitly by these)
    numFlows = 2  # (brightness/darkness) right and down
    flowFilter = [[1,0], [0,1]]  # down, right

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

    # LGN
    sys.stdout.write('LGN,...')
    sys.stdout.flush()

    # Neural LGN cells will receive input values from LGN
    LGNBrightInput = sim.Population(ImageNumPixelRows*ImageNumPixelColumns, sim.SpikeSourcePoisson(), label="LGNBrightInput")
    LGNDarkInput   = sim.Population(ImageNumPixelRows*ImageNumPixelColumns, sim.SpikeSourcePoisson(), label="LGNDarkInput")
    LGNBright      = sim.Population(ImageNumPixelRows*ImageNumPixelColumns, normalCellType,           label="LGNBright")
    LGNDark        = sim.Population(ImageNumPixelRows*ImageNumPixelColumns, normalCellType)

    # Area V1
    sys.stdout.write('V1,...')
    sys.stdout.flush()

    # Simple oriented neurons
    V1Layer4P1 = sim.Population(numOrientations*numPixelRows*numPixelColumns, normalCellType)
    V1Layer4P2 = sim.Population(numOrientations*numPixelRows*numPixelColumns, normalCellType)
    V1Layer6P1 = sim.Population(numOrientations*numPixelRows*numPixelColumns, normalCellType, label="V1Layer6P1")
    V1Layer6P2 = sim.Population(numOrientations*numPixelRows*numPixelColumns, normalCellType)

    # Complex cells
    V1Layer23       = sim.Population(numOrientations*numPixelRows*numPixelColumns, normalCellType)
    V1Layer23Pool   = sim.Population(numOrientations*numPixelRows*numPixelColumns, normalCellType)
    V1Layer23Inter1 = sim.Population(numOrientations*numPixelRows*numPixelColumns, normalCellType)
    V1Layer23Inter2 = sim.Population(numOrientations*numPixelRows*numPixelColumns, normalCellType)

    ###### All subsequent areas have multiple segmentation representations

    # Area V2
    sys.stdout.write('V2,...')
    sys.stdout.flush()

    V2Layer4        = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, normalCellType)
    V2Layer6        = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, normalCellType)
    V2Layer23       = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, normalCellType, label="V2Layer23")
    V2Layer23Pool   = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, normalCellType)
    V2Layer23Inter1 = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, normalCellType)
    V2Layer23Inter2 = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, normalCellType)

    # Area V4
    sys.stdout.write('V4,...')
    sys.stdout.flush()

    V4Brightness       = sim.Population(numSegmentationLayers*ImageNumPixelRows*ImageNumPixelColumns,          normalCellType)
    V4InterBrightness1 = sim.Population(numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns, normalCellType)
    V4InterBrightness2 = sim.Population(numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns, normalCellType)
    V4Darkness         = sim.Population(numSegmentationLayers*ImageNumPixelRows*ImageNumPixelColumns,          normalCellType)
    V4InterDarkness1   = sim.Population(numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns, normalCellType)
    V4InterDarkness2   = sim.Population(numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns, normalCellType)

    if numSegmentationLayers>1:
        if useSurfaceSegmentation==1:
            # Surface Segmentation cells
            sys.stdout.write('Surface,...')
            sys.stdout.flush()

            SurfaceSegmentationOn        = sim.Population((numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns,          normalCellType)
            SurfaceSegmentationOnInter1  = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, normalCellType)
            SurfaceSegmentationOnInter2  = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, normalCellType)
            SurfaceSegmentationOff       = sim.Population((numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns,          normalCellType)
            SurfaceSegmentationOffInter1 = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, normalCellType)
            SurfaceSegmentationOffInter2 = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, normalCellType)
            SurfaceSegmentationOnSignal  = sim.Population((numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns,          normalCellType, label="SurfaceSegmentationOnSignal")
            SurfaceSegmentationOffSignal = sim.Population((numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns,          normalCellType, label="SurfaceSegmentationOffSignal")

        if useBoundarySegmentation==1:
            # Boundary Segmentation cells
            sys.stdout.write('Boundary,...')
            sys.stdout.flush()

            BoundarySegmentationOn        = sim.Population((numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns,          normalCellType)
            BoundarySegmentationOnInter1  = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, normalCellType)
            BoundarySegmentationOnInter2  = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, normalCellType)
            BoundarySegmentationOnInter3  = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, tonicCellType)
            BoundarySegmentationOff       = sim.Population((numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns,          normalCellType)
            BoundarySegmentationOffInter1 = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, normalCellType)
            BoundarySegmentationOffInter2 = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, normalCellType)
            BoundarySegmentationOnSignal  = sim.Population((numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns,          normalCellType, label="BoundarySegmentationOnSignal")
            BoundarySegmentationOffSignal = sim.Population((numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns,          normalCellType, label="BoundarySegmentationOffSignal")


    ######################################################################
    ### Neurons layers are defined, now set up connexions between them ###
    ######################################################################

    ############ LGN and Input ###############

    inputToLGNBright = sim.Projection(LGNBrightInput, LGNBright, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['brightInputToLGN']))
    inputToLGNDark   = sim.Projection(LGNDarkInput,   LGNDark,   sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['darkInputToLGN']))
    synapseCount = 0 #len(LGNBrightInput) + len(LGNDarkInput)

    ############### Area V1 ##################

    sys.stdout.write('done. \nSetting up V1, Layers 4 and 6...')
    sys.stdout.flush()

    oriFilterWeight = connections['LGN_ToV1Excite']
    for k in range(0, numOrientations):                         # Orientations
        for i2 in range(-oriFilterSize/2, oriFilterSize/2):     # Filter rows
            for j2 in range(-oriFilterSize/2, oriFilterSize/2): # Filter columns
                ST = []                                         # Source-Target vector containing indexes of neurons to connect within specific layers
                ST2 = []                                        # Second Source-Target vector for another connection
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
                    LGNBrightToV16P1 = sim.Projection(LGNBright, V1Layer6P1, sim.FromListConnector(ST), sim.StaticSynapse(weight=oriFilterWeight*filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                    LGNDarkToV16P2   = sim.Projection(LGNDark,   V1Layer6P2, sim.FromListConnector(ST), sim.StaticSynapse(weight=oriFilterWeight*filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                    LGNBrightToV14P1 = sim.Projection(LGNBright, V1Layer4P1, sim.FromListConnector(ST), sim.StaticSynapse(weight=oriFilterWeight*filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                    LGNDarkToV14P2   = sim.Projection(LGNDark,   V1Layer4P2, sim.FromListConnector(ST), sim.StaticSynapse(weight=oriFilterWeight*filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                    synapseCount += 4*len(ST)

                if len(ST2)>0:
                    # LGN -> Layer 6 and 4 (simple cells) connections (no connections at the edges, to avoid edge-effects) second polarity filter
                    LGNBrightToV16P2 = sim.Projection(LGNBright, V1Layer6P2, sim.FromListConnector(ST2), sim.StaticSynapse(weight=oriFilterWeight*filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                    LGNDarkToV16P1   = sim.Projection(LGNDark,   V1Layer6P1, sim.FromListConnector(ST2), sim.StaticSynapse(weight=oriFilterWeight*filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                    LGNBrightToV14P2 = sim.Projection(LGNBright, V1Layer4P2, sim.FromListConnector(ST2), sim.StaticSynapse(weight=oriFilterWeight*filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                    LGNDarkToV14P1   = sim.Projection(LGNDark,   V1Layer4P1, sim.FromListConnector(ST2), sim.StaticSynapse(weight=oriFilterWeight*filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]))
                    synapseCount += 4*len(ST2)

    # Excitatory connection from same orientation and polarity 1, input from layer 6
    V16P1ExcitV14P1 = sim.Projection(V1Layer6P1, V1Layer4P1, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_6To4Excite']))
    V16P2EcxitV14P2 = sim.Projection(V1Layer6P2, V1Layer4P2, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_6To4Excite']))
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
    V16P1SurroundInhibV14P1 = sim.Projection(V1Layer6P1, V1Layer4P1, sim.FromListConnector(ST), sim.StaticSynapse(weight=connections['V1_6To4Inhib']))
    V16P2SurroundInhibV14P2 = sim.Projection(V1Layer6P2, V1Layer4P2, sim.FromListConnector(ST), sim.StaticSynapse(weight=connections['V1_6To4Inhib']))
    synapseCount += 2*len(ST)

    sys.stdout.write('done. \nSetting up V1, Layers 23 and 6 (feedback)...')
    sys.stdout.flush()

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

    # Layer 4 -> Layer23 (complex cell connections)
    V14P1ComplexExcitV123 = sim.Projection(V1Layer4P1, V1Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_ComplexExcite']))
    V14P2ComplexExcitV123 = sim.Projection(V1Layer4P2, V1Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_ComplexExcite']))
    synapseCount += (len(V1Layer4P1)+len(V1Layer4P2))

    # Cross-orientation inhibition
    V123CrossOriInh = sim.Projection(V1Layer23, V1Layer23, sim.FromListConnector(ST), sim.StaticSynapse(weight=connections['V1_CrossInhib']))
    synapseCount += len(ST)

    # Pooling neurons in Layer 23 (excitation from same orientation, inhibition from orthogonal), V1poolingfilters pools from both sides
    V123PoolingExc = sim.Projection(V1Layer23, V1Layer23Pool, sim.FromListConnector(ST2), sim.StaticSynapse(weight=connections['V1_ComplexExcite']))
    V123PoolingInh = sim.Projection(V1Layer23, V1Layer23Pool, sim.FromListConnector(ST3), sim.StaticSynapse(weight=connections['V1_ComplexInhib']))
    synapseCount += (len(ST2) + len(ST3))

    # Pooling neurons back to Layer 23 and to interneurons (ST4 for one side and ST5 for the other), V1poolingconnections pools from only one side
    V123PoolBackExcR =      sim.Projection(V1Layer23Pool, V1Layer23,       sim.FromListConnector(ST4), sim.StaticSynapse(weight=connections['V1_FeedbackExcite']))
    V123PoolBackInter1Exc = sim.Projection(V1Layer23Pool, V1Layer23Inter1, sim.FromListConnector(ST4), sim.StaticSynapse(weight=connections['V1_FeedbackExcite']))
    V123PoolBackInter2Inh = sim.Projection(V1Layer23Pool, V1Layer23Inter2, sim.FromListConnector(ST4), sim.StaticSynapse(weight=connections['V1_NegFeedbackInhib']))
    V123PoolBackExcL      = sim.Projection(V1Layer23Pool, V1Layer23,       sim.FromListConnector(ST5), sim.StaticSynapse(weight=connections['V1_FeedbackExcite']))
    V123PoolBackInter2Exc = sim.Projection(V1Layer23Pool, V1Layer23Inter2, sim.FromListConnector(ST5), sim.StaticSynapse(weight=connections['V1_FeedbackExcite']))
    V123PoolBackInter1Inh = sim.Projection(V1Layer23Pool, V1Layer23Inter1, sim.FromListConnector(ST5), sim.StaticSynapse(weight=connections['V1_NegFeedbackInhib']))
    synapseCount += 3*(len(ST4) + len(ST5))

    # Connect interneurons to complex cell and each other
    V123Inter1ToV123 = sim.Projection(V1Layer23Inter1, V1Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_InterInhib']))
    V123Inter2ToV123 = sim.Projection(V1Layer23Inter2, V1Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_InterInhib']))
    synapseCount += (len(V1Layer23Inter1) + len(V1Layer23Inter2))

    # End-cutting (excitation from orthogonal interneuron)
    V123Inter1EndCutExcit = sim.Projection(V1Layer23Inter1, V1Layer23, sim.FromListConnector(ST6), sim.StaticSynapse(weight=connections['V1_EndCutExcite']))
    V123Inter2EndCutExcit = sim.Projection(V1Layer23Inter2, V1Layer23, sim.FromListConnector(ST6), sim.StaticSynapse(weight=connections['V1_EndCutExcite']))
    synapseCount += 2*len(ST6)

    # Connect Layer 23 cells to Layer 6 cells (folded feedback)
    V123FoldedFeedbackV16P1 = sim.Projection(V1Layer23, V1Layer6P1, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_23To6Excite']))
    V123FoldedFeedbackV16P2 = sim.Projection(V1Layer23, V1Layer6P2, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V1_23To6Excite']))
    synapseCount += 2*len(V1Layer23)


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
                    ST.append((k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                               h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

                    for i2 in range(-inhibRange64, inhibRange64+1):
                        for j2 in range(-inhibRange64, inhibRange64+1):
                            if i+i2 >=0 and i+i2 <numPixelRows and i2!=0 and j+j2 >=0 and j+j2 <numPixelColumns and j2!=0:
                                ST2.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                            h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j))

    # V2 Layers 4 and 6 connections
    V123ToV26 = sim.Projection(V1Layer23,  V2Layer6, sim.FromListConnector(ST), sim.StaticSynapse(weight=connections['V1_ToV2Excite']))
    V123ToV24 = sim.Projection(V1Layer23,  V2Layer4, sim.FromListConnector(ST), sim.StaticSynapse(weight=connections['V1_ToV2Excite']))
    V26ToV24  = sim.Projection(V2Layer6, V2Layer4, sim.OneToOneConnector(),   sim.StaticSynapse(weight=connections['V2_6To4Excite']))
    synapseCount += (2*len(ST) + len(V2Layer6))

    # Surround inhibition V2 Layer 6 -> 4
    V26SurroundInhibV24 = sim.Projection(V2Layer6,  V2Layer4, sim.FromListConnector(ST2), sim.StaticSynapse(weight=connections['V2_6To4Inhib']))
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
    V24ToV223 = sim.Projection(V2Layer4, V2Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V2_ComplexExcite']))
    synapseCount += len(V2Layer4)

    # Cross-orientation inhibition
    V223CrossOriInh = sim.Projection(V2Layer23, V2Layer23, sim.FromListConnector(ST), sim.StaticSynapse(weight=connections['V2_OrientInhib']))
    synapseCount += len(ST)

    # Pooling neurons in V2Layer 23 (excitation from same orientation, inhibition from different + stronger for orthogonal orientation)
    V223PoolingExc     = sim.Projection(V2Layer23, V2Layer23Pool, sim.FromListConnector(ST2), sim.StaticSynapse(weight=connections['V2_ComplexExcite']))
    V223PoolingInhOpp  = sim.Projection(V2Layer23, V2Layer23Pool, sim.FromListConnector(ST3), sim.StaticSynapse(weight=connections['V2_ComplexInhib']))
    synapseCount += (len(ST2) + len(ST3))
    if len(ST4)>0:  # non-orthogonal inhibition
        V223PoolingInh = sim.Projection(V2Layer23, V2Layer23Pool, sim.FromListConnector(ST4), sim.StaticSynapse(weight=connections['V2_ComplexInhib2']))
        synapseCount += len(ST4)

    # Pooling neurons back to Layer 23 and to interneurons (ST5 for one side and ST6 for the other), V1poolingconnections pools from only one side
    V223PoolBackExcR      = sim.Projection(V2Layer23Pool, V2Layer23,       sim.FromListConnector(ST5), sim.StaticSynapse(weight=connections['V2_FeedbackExcite']))
    V223PoolBackInter1Exc = sim.Projection(V2Layer23Pool, V2Layer23Inter1, sim.FromListConnector(ST5), sim.StaticSynapse(weight=connections['V2_FeedbackExcite']))
    V223PoolBackInter2Inh = sim.Projection(V2Layer23Pool, V2Layer23Inter2, sim.FromListConnector(ST5), sim.StaticSynapse(weight=connections['V2_NegFeedbackInhib']))
    V223PoolBackExcL      = sim.Projection(V2Layer23Pool, V2Layer23,       sim.FromListConnector(ST6), sim.StaticSynapse(weight=connections['V2_FeedbackExcite']))
    V223PoolBackInter2Exc = sim.Projection(V2Layer23Pool, V2Layer23Inter2, sim.FromListConnector(ST6), sim.StaticSynapse(weight=connections['V2_FeedbackExcite']))
    V223PoolBackInter1Inh = sim.Projection(V2Layer23Pool, V2Layer23Inter1, sim.FromListConnector(ST6), sim.StaticSynapse(weight=connections['V2_NegFeedbackInhib']))
    synapseCount += (3*len(ST5) + 3*len(ST6))

    # Connect interneurons to complex cell
    V2Inter1ToV223 = sim.Projection(V2Layer23Inter1, V2Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V2_InterInhib']))
    V2Inter2ToV223 = sim.Projection(V2Layer23Inter2, V2Layer23, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V2_InterInhib']))
    synapseCount += (len(V2Layer23Inter1) + len(V2Layer23Inter2))

    # Connect Layer 23 cells to Layer 6 cells (folded feedback)
    V223FoldedFeedbackV26 = sim.Projection(V2Layer23, V2Layer6, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V2_23To6Excite']))
    synapseCount += len(V2Layer23)

    ############# Area V4 filling-in #############

    sys.stdout.write('done. \nSetting up V4...')
    sys.stdout.flush()

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
    V4DarkToBrightInhib = sim.Projection(V4Darkness,   V4Brightness, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V4_BetweenColorsInhib']))
    V4BrightToDarkInhib = sim.Projection(V4Brightness, V4Darkness,   sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V4_BetweenColorsInhib']))
    synapseCount += (len(V4Darkness) + len(V4Brightness))

    # LGNBright->V4brightness and LGNDark->V4darkness
    LGNBrightToV4Bright = sim.Projection(LGNBright, V4Brightness, sim.FromListConnector(ST), sim.StaticSynapse(weight=connections['LGN_ToV4Excite']))
    LGNDarkToV4Dark     = sim.Projection(LGNDark,   V4Darkness,   sim.FromListConnector(ST), sim.StaticSynapse(weight=connections['LGN_ToV4Excite']))
    synapseCount += 2*len(ST)

    # V4brightness<->Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
    V4BrightInterExcite = sim.Projection(V4Brightness,       V4InterBrightness1, sim.FromListConnector(ST2),               sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
    V4BrightInterInhib  = sim.Projection(V4Brightness,       V4InterBrightness2, sim.FromListConnector(ST2),               sim.StaticSynapse(weight=connections['V4_BrightnessInhib']))
    V4InterBrightInhib  = sim.Projection(V4InterBrightness1, V4Brightness,       sim.FromListConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=connections['V4_BrightnessInhib']))
    V4InterBrightExcite = sim.Projection(V4InterBrightness2, V4Brightness,       sim.FromListConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
    synapseCount += 4*len(ST2)

    # V4darkness<->Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
    V4DarkInterExcite = sim.Projection(V4Darkness,       V4InterDarkness1, sim.FromListConnector(ST2),               sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
    V4DarkInterInhib  = sim.Projection(V4Darkness,       V4InterDarkness2, sim.FromListConnector(ST2),               sim.StaticSynapse(weight=connections['V4_BrightnessInhib']))
    V4InterDarkInhib  = sim.Projection(V4InterDarkness1, V4Darkness,       sim.FromListConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=connections['V4_BrightnessInhib']))
    V4InterDarkExcite = sim.Projection(V4InterDarkness2, V4Darkness,       sim.FromListConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
    synapseCount += 4*len(ST2)

    # V4brightness neighbors<->Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
    V4BrightNeighborInterExcite = sim.Projection(V4Brightness,       V4InterBrightness2, sim.FromListConnector(ST3),               sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
    V4BrightNeighborInterInhib  = sim.Projection(V4Brightness,       V4InterBrightness1, sim.FromListConnector(ST3),               sim.StaticSynapse(weight=connections['V4_BrightnessInhib']))
    V4InterBrightNeighborInhib  = sim.Projection(V4InterBrightness2, V4Brightness,       sim.FromListConnector(numpy.fliplr(ST3)), sim.StaticSynapse(weight=connections['V4_BrightnessInhib']))
    V4InterBrightNeighborExcite = sim.Projection(V4InterBrightness1, V4Brightness,       sim.FromListConnector(numpy.fliplr(ST3)), sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
    synapseCount += 4*len(ST3)

    # V4darkness neighbors<->Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
    V4DarkNeighborInterExcite = sim.Projection(V4Darkness,       V4InterDarkness2, sim.FromListConnector(ST3),               sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
    V4DarkNeighborInterInhib  = sim.Projection(V4Darkness,       V4InterDarkness1, sim.FromListConnector(ST3),               sim.StaticSynapse(weight=connections['V4_BrightnessInhib']))
    V4InterDarkNeighborInhib  = sim.Projection(V4InterDarkness2, V4Darkness,       sim.FromListConnector(numpy.fliplr(ST3)), sim.StaticSynapse(weight=connections['V4_BrightnessInhib']))
    V4InterDarkNeighborExcite = sim.Projection(V4InterDarkness1, V4Darkness,       sim.FromListConnector(numpy.fliplr(ST3)), sim.StaticSynapse(weight=connections['V4_BrightnessExcite']))
    synapseCount += 4*len(ST3)

    # V2Layer23 -> V4 Interneurons (all boundaries block except for orientation of flow)
    V223BoundInhibV4InterBright1 = sim.Projection(V2Layer23, V4InterBrightness1, sim.FromListConnector(ST4), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
    V223BoundInhibV4InterBright2 = sim.Projection(V2Layer23, V4InterBrightness2, sim.FromListConnector(ST4), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
    V223BoundInhibV4InterDark1   = sim.Projection(V2Layer23, V4InterDarkness1,   sim.FromListConnector(ST4), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
    V223BoundInhibV4InterDark2   = sim.Projection(V2Layer23, V4InterDarkness2,   sim.FromListConnector(ST4), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
    synapseCount += 4*len(ST4)

    # Strong inhibition between segmentation layers
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
        V2InterSegmentInhib = sim.Projection(V2Layer23, V2Layer4, sim.FromListConnector(ST), sim.StaticSynapse(weight=connections['V2_SegmentInhib']))
        synapseCount += len(ST)

    ########### Surface segmentation network ############

    if numSegmentationLayers>1 and useSurfaceSegmentation==1:
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
        SurfSegmOnSignalInput  = sim.Projection(SurfaceSegmentationOnSignal,  SurfaceSegmentationOn,  sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['S_SegmentSignalExcite']))
        SurfSegmOffSignalInput = sim.Projection(SurfaceSegmentationOffSignal, SurfaceSegmentationOff, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['S_SegmentSignalExcite']))
        SurfSegmOffInhibOn     = sim.Projection(SurfaceSegmentationOff,       SurfaceSegmentationOn,  sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['S_SegmentOnOffInhib']))
        synapseCount += (len(SurfaceSegmentationOnSignal) + len(SurfaceSegmentationOffSignal) + len(SurfaceSegmentationOff))

        # SurfaceSegmentationOn/Off <-> Interneurons ; fliplr to use the connections in the way "target indexes --> source indexes"
        SurfSegmOnToInter1  = sim.Projection(SurfaceSegmentationOn,        SurfaceSegmentationOnInter1,  sim.FromListConnector(ST),               sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
        Inter2ToSurfSegmOn  = sim.Projection(SurfaceSegmentationOnInter2,  SurfaceSegmentationOn,        sim.FromListConnector(numpy.fliplr(ST)), sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
        SurfSegmOffToInter1 = sim.Projection(SurfaceSegmentationOff,       SurfaceSegmentationOffInter1, sim.FromListConnector(ST),               sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
        Inter2ToSurfSegmOff = sim.Projection(SurfaceSegmentationOffInter2, SurfaceSegmentationOff,       sim.FromListConnector(numpy.fliplr(ST)), sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
        synapseCount += 4*len(ST)

        # SurfaceSegmentationOn/Off <-> Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
        SurfSegmOnToInter2  = sim.Projection(SurfaceSegmentationOn,        SurfaceSegmentationOnInter2,  sim.FromListConnector(ST2),               sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
        Inter1ToSurfSegmOn  = sim.Projection(SurfaceSegmentationOnInter1,  SurfaceSegmentationOn,        sim.FromListConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
        SurfSegmOffToInter2 = sim.Projection(SurfaceSegmentationOff,       SurfaceSegmentationOffInter2, sim.FromListConnector(ST2),               sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
        Inter1ToSurfSegmOff = sim.Projection(SurfaceSegmentationOffInter1, SurfaceSegmentationOff,       sim.FromListConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=connections['S_SegmentSpreadExcite']))
        synapseCount += 4*len(ST2)

        # Mutual inhibition of interneurons
        SurfSegmOnInterInhib1To2  = sim.Projection(SurfaceSegmentationOnInter1,  SurfaceSegmentationOnInter2,  sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['S_SegmentInterInhib']))
        SurfSegmOnInterInhib2To1  = sim.Projection(SurfaceSegmentationOnInter2,  SurfaceSegmentationOnInter1,  sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['S_SegmentInterInhib']))
        SurfSegmOffInterInhib1To2 = sim.Projection(SurfaceSegmentationOffInter1, SurfaceSegmentationOffInter2, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['S_SegmentInterInhib']))
        SurfSegmOffInterInhib2To1 = sim.Projection(SurfaceSegmentationOffInter2, SurfaceSegmentationOffInter1, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['S_SegmentInterInhib']))
        synapseCount += (len(SurfaceSegmentationOffInter1) + len(SurfaceSegmentationOffInter2) + len(SurfaceSegmentationOnInter1) + len(SurfaceSegmentationOnInter2))

        # V2Layer23 -> Segmentation Interneurons (all boundaries block except for orientation of flow)
        V223ToSurfSegmOnInter1  = sim.Projection(V2Layer23, SurfaceSegmentationOnInter1,  sim.FromListConnector(ST3), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
        V223ToSurfSegmOnInter2  = sim.Projection(V2Layer23, SurfaceSegmentationOnInter2,  sim.FromListConnector(ST3), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
        V223ToSurfSegmOffInter1 = sim.Projection(V2Layer23, SurfaceSegmentationOffInter1, sim.FromListConnector(ST3), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
        V223ToSurfSegmOffInter2 = sim.Projection(V2Layer23, SurfaceSegmentationOffInter2, sim.FromListConnector(ST3), sim.StaticSynapse(weight=connections['V2_BoundaryInhib']))
        synapseCount += 4*len(ST3)

        # V2Layer23 -> V2Layer4 strong inhibition (CHECK WHY THIS CONNECTION IS THERE TWICE WITH THE SAME CONNECTION PATTERN)
        V223InhibV24 = sim.Projection(V2Layer23, V2Layer4, sim.FromListConnector(ST4), sim.StaticSynapse(weight=connections['V2_SegmentInhib']))
        synapseCount += len(ST4)

        # Segmentation -> V2Layer4 (gating) ; way for lower levels to be inhibited by higher ones : through segmentation network)
        SurfSegmOnInhibV24 = sim.Projection(SurfaceSegmentationOn, V2Layer4, sim.FromListConnector(ST5), sim.StaticSynapse(weight=connections['S_SegmentOnOffInhib']))
        synapseCount += len(ST5)

    ########### Boundary segmentation network ############

    if numSegmentationLayers>1 and useBoundarySegmentation==1:
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
        BoundSegmOnSignalInput  = sim.Projection(BoundarySegmentationOnSignal,  BoundarySegmentationOn,  sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['B_SegmentSignalExcite']))
        BoundSegmOffSignalInput = sim.Projection(BoundarySegmentationOffSignal, BoundarySegmentationOff, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['B_SegmentSignalExcite']))
        # BoundSegmOffInhibOn   = sim.Projection(BoundarySegmentationOff,       BoundarySegmentationOn,  sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['B_SegmentOnOffInhib']))
        synapseCount += (len(BoundarySegmentationOnSignal) + len(BoundarySegmentationOffSignal))  # + len(BoundarySegmentationOff))

        # BoundarySegmentationOn<->Interneurons ; fliplr to use the connections in the way "target indexes --> source indexes"
        BoundSegmOnToInter1 =  sim.Projection(BoundarySegmentationOn,       BoundarySegmentationOnInter1, sim.FromListConnector(ST),               sim.StaticSynapse(weight=connections['B_SegmentSpreadExcite']))
        Inter2ToBoundSegmOn =  sim.Projection(BoundarySegmentationOnInter2, BoundarySegmentationOn,       sim.FromListConnector(numpy.fliplr(ST)), sim.StaticSynapse(weight=connections['B_SegmentSpreadExcite']))
        synapseCount += 2*len(ST)

        # SurfaceSegmentationOn <-> Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
        BoundSegmOnToInter2 = sim.Projection(BoundarySegmentationOn,       BoundarySegmentationOnInter2, sim.FromListConnector(ST2),               sim.StaticSynapse(weight=connections['B_SegmentSpreadExcite']))
        Inter1ToBoundSegmOn = sim.Projection(BoundarySegmentationOnInter1, BoundarySegmentationOn,       sim.FromListConnector(numpy.fliplr(ST2)), sim.StaticSynapse(weight=connections['B_SegmentSpreadExcite']))
        synapseCount += 2*len(ST2)

        # Mutual inhibition of interneurons (MAY NOT NEED THIS: flow only when Inter3 is inhibited - 19 Dec 2014)
        BoundSegmOnInterInhib1To2 = sim.Projection(BoundarySegmentationOnInter1, BoundarySegmentationOnInter2, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['B_SegmentInterInhib']))
        BoundSegmOnInterInhib2To1 = sim.Projection(BoundarySegmentationOnInter2, BoundarySegmentationOnInter1, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['B_SegmentInterInhib']))
        synapseCount += 2*len(BoundarySegmentationOnInter1)

        # Inhibition from third interneuron (itself inhibited by the presence of a boundary, see next connection)
        BoundSegmOnInter3Inhib1Inter1 = sim.Projection(BoundarySegmentationOnInter3, BoundarySegmentationOnInter1, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['B_SegmentTonicInhib']))
        BoundSegmOnInter3Inhib1Inter2 = sim.Projection(BoundarySegmentationOnInter3, BoundarySegmentationOnInter2, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['B_SegmentTonicInhib']))
        synapseCount += 2*len(BoundarySegmentationOnInter3)

        # V2Layer23 -> Segmentation Interneurons (all boundaries open flow by inhibiting third interneuron)
        V223ToBoundSegmOnInter3 =  sim.Projection(V2Layer23, BoundarySegmentationOnInter3,  sim.FromListConnector(ST3), sim.StaticSynapse(weight=connections['B_SegmentOpenFlowInhib']))
        synapseCount += len(ST3)

        # BoundarySegmentation -> V2Layer4 (gating)
        BoundSegmOnInhibV24 = sim.Projection(BoundarySegmentationOn, V2Layer4, sim.FromListConnector(ST4), sim.StaticSynapse(weight=connections['B_SegmentOnOffInhib']))
        synapseCount += len(ST4)

    sys.stdout.write('done. \n'+str(synapseCount)+' network connections created.\n')
    sys.stdout.flush()

    # Return only the populations that need to be updated online during the simulation and that we want to make a gif of
    network = LGNBrightInput + LGNDarkInput + V2Layer23 + LGNBright + V1Layer6P1
    if useSurfaceSegmentation:
        network += (SurfaceSegmentationOnSignal + SurfaceSegmentationOffSignal)
    if useBoundarySegmentation:
        network += (BoundarySegmentationOnSignal + BoundarySegmentationOffSignal)
    return sim, network
