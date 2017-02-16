# This file contains the setup of the neuronal network running the Husky experiment with neuronal image recognition
import os, sys, numpy, matplotlib.pyplot as plt
from createFilters import createFilters, createPoolingConnectionsAndFilters
# from pyNN.connectors import Connector
import pyNN.spiNNaker as sim

# Function that builds all the layers of the network and connect them according to the Laminart model, enhanced with segmentation ; returns the full network and all the connections
def buildNetworkAndConnections(ImageNumPixelRows, ImageNumPixelColumns, numOrientations, oriFilterSize, V1PoolSize, V2PoolSize, phi,
                               connections, numSegmentationLayers, useBoundarySegmentation, useSurfaceSegmentation):

    #########################################################
    ### Build orientation filters and connection patterns ###
    #########################################################
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

    sys.stdout.write('\nSetting up orientation filters...')

    # Boundary coordinates (boundaries exist at positions between retinotopic coordinates, so add extra pixel on each side to insure a boundary could exists for retinal pixel)
    numPixelRows = ImageNumPixelRows + 1        # height for oriented neurons (placed between un-oriented pixels)
    numPixelColumns = ImageNumPixelColumns + 1  # width for oriented neurons (placed between un-oriented pixels)

    # Set the orientation filters (orientation kernels, V1 and V2 layer23 pooling filters)
    filters1, filters2 = createFilters(numOrientations, oriFilterSize, sigma2=0.75, Olambda=4, phi=phi)
    V1PoolingFilters, V1PoolingConnections1, V1PoolingConnections2 = createPoolingConnectionsAndFilters(numOrientations, VPoolSize=V1PoolSize, sigma2=4.0, Olambda=5, phi=phi)
    V2PoolingFilters, V2PoolingConnections1, V2PoolingConnections2 = createPoolingConnectionsAndFilters(numOrientations, VPoolSize=V2PoolSize, sigma2=26.0, Olambda=9, phi=phi)

    OppositeOrientationIndex = list(numpy.roll(range(numOrientations), numOrientations/2))
    # For numOrientations = 2, orientation indexes = [vertical, horizontal] -> opposite orientation indexes = [horizontal, vertical]
    # For numOrientations = 4, orientation indexes = [ /, |, \, - ] -> opposite orientation indexes = [ \, -, \, | ]
    # For numOrientations = 8, [ /h, /, /v, |, \v, \, \h, - ] -> [ \v, \, \h, -, /h, /, /v, | ] ([h,v] = [more horizontal, more vertical])

    # Set up filters for filling-in stage (spreads in various directions).
    # Interneurons receive inhib. input from all but boundary ori. that matches flow direction. Up, Right (Down and Left are defined implicitly by these)
    numFlows = 2  # (brightness/darkness) right and down
    flowFilter = [[1,0], [0,1]]  # down, right ; default is [[1,0],[0,1]]

    # Specify flow orientation (all others block) and position of blockers (different subarrays are for different flow directions)
    # BoundaryBlockFilter = [[[vertical, 1, 1], [vertical, 1, 0]], [[horizontal, 1, 1], [horizontal, 0, 1]]]
    # 1 added to each offset position because boundary grid is (1,1) offset from brightness grid
    BoundaryBlockFilter = [[[numOrientations/2-1, 1, 1], [numOrientations/2-1, 1, 0]], [[numOrientations-1, 1, 1], [numOrientations-1, 0, 1]]]

    ########################################################################################################################
    ### Create the neuron layers ((Retina,) LGN, V1, V2, V4, Boundary and Surface Segmentation Layers) + Spike Detectors ###
    ########################################################################################################################

    sys.stdout.write('Done. \nDefining cells...')
    sys.stdout.flush()

    # LGN
    sys.stdout.write('LGN,...')
    sys.stdout.flush()

    # Neural LGN cells will receive input values from LGN
    LGNBright      = sim.Population(ImageNumPixelRows*ImageNumPixelColumns, sim.IF_curr_exp, cellParams, label="LGNBright")
    LGNDark        = sim.Population(ImageNumPixelRows*ImageNumPixelColumns, sim.IF_curr_exp, cellParams, label="LGNDark")

    # Area V1
    sys.stdout.write('V1,...')
    sys.stdout.flush()

    # Simple oriented neurons
    V1Layer6P1 = sim.Population(numOrientations*numPixelRows*numPixelColumns, sim.IF_curr_exp, cellParams)
    V1Layer6P2 = sim.Population(numOrientations*numPixelRows*numPixelColumns, sim.IF_curr_exp, cellParams)
    V1Layer4P1 = sim.Population(numOrientations*numPixelRows*numPixelColumns, sim.IF_curr_exp, cellParams)
    V1Layer4P2 = sim.Population(numOrientations*numPixelRows*numPixelColumns, sim.IF_curr_exp, cellParams)

    # Complex cells
    V1Layer23       = sim.Population(numOrientations*numPixelRows*numPixelColumns, sim.IF_curr_exp, cellParams, label="V1LayerToPlot")
    V1Layer23Pool   = sim.Population(numOrientations*numPixelRows*numPixelColumns, sim.IF_curr_exp, cellParams)
    V1Layer23Inter1 = sim.Population(numOrientations*numPixelRows*numPixelColumns, sim.IF_curr_exp, cellParams)
    V1Layer23Inter2 = sim.Population(numOrientations*numPixelRows*numPixelColumns, sim.IF_curr_exp, cellParams)

    ###### All subsequent areas have multiple segmentation representations

    # Area V2
    sys.stdout.write('V2,...')
    sys.stdout.flush()

    V2Layer23Inter1 = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, sim.IF_curr_exp, cellParams)
    V2Layer23Inter2 = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, sim.IF_curr_exp, cellParams)
    V2Layer6        = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, sim.IF_curr_exp, cellParams)
    V2Layer4        = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, sim.IF_curr_exp, cellParams)
    V2Layer23       = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, sim.IF_curr_exp, cellParams, label="V2LayerToPlot")
    V2Layer23Pool   = sim.Population(numSegmentationLayers*numOrientations*numPixelRows*numPixelColumns, sim.IF_curr_exp, cellParams)

    # Area V4
    sys.stdout.write('V4,...')
    sys.stdout.flush()

    V4Brightness       = sim.Population(numSegmentationLayers*ImageNumPixelRows*ImageNumPixelColumns,          sim.IF_curr_exp, cellParams, label="V4Brightness")
    V4InterBrightness1 = sim.Population(numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns, sim.IF_curr_exp, cellParams)
    V4InterBrightness2 = sim.Population(numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns, sim.IF_curr_exp, cellParams)
    V4Darkness         = sim.Population(numSegmentationLayers*ImageNumPixelRows*ImageNumPixelColumns,          sim.IF_curr_exp, cellParams, label="V4Darkness")
    V4InterDarkness1   = sim.Population(numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns, sim.IF_curr_exp, cellParams)
    V4InterDarkness2   = sim.Population(numSegmentationLayers*numFlows*ImageNumPixelRows*ImageNumPixelColumns, sim.IF_curr_exp, cellParams)

    if numSegmentationLayers>1:
        if useSurfaceSegmentation==1:
            # Surface Segmentation cells
            sys.stdout.write('Surface,...')
            sys.stdout.flush()

            SurfaceSegmentationOn        = sim.Population((numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns,          sim.IF_curr_exp, cellParams, label="SurfaceSegmentationOn")
            SurfaceSegmentationOnInter1  = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, sim.IF_curr_exp, cellParams)
            SurfaceSegmentationOnInter2  = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, sim.IF_curr_exp, cellParams)
            SurfaceSegmentationOff       = sim.Population((numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns,          sim.IF_curr_exp, cellParams, label="SurfaceSegmentationOff")
            SurfaceSegmentationOffInter1 = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, sim.IF_curr_exp, cellParams)
            SurfaceSegmentationOffInter2 = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, sim.IF_curr_exp, cellParams)

        if useBoundarySegmentation==1:
            # Boundary Segmentation cells
            sys.stdout.write('Boundary,...')
            sys.stdout.flush()

            BoundarySegmentationOn        = sim.Population((numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns,          sim.IF_curr_exp, cellParams, label="BoundarySegmentationOn")
            BoundarySegmentationOnInter1  = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, sim.IF_curr_exp, cellParams)
            BoundarySegmentationOnInter2  = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, sim.IF_curr_exp, cellParams)
            BoundarySegmentationOnInter3  = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, sim.IF_curr_exp, cellParams, label="BoundarySegmentationOnInter3")
            BoundarySegmentationOff       = sim.Population((numSegmentationLayers-1)*ImageNumPixelRows*ImageNumPixelColumns,          sim.IF_curr_exp, cellParams, label="BoundarySegmentationOff")
            BoundarySegmentationOffInter1 = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, sim.IF_curr_exp, cellParams)
            BoundarySegmentationOffInter2 = sim.Population((numSegmentationLayers-1)*numFlows*ImageNumPixelRows*ImageNumPixelColumns, sim.IF_curr_exp, cellParams)

    ######################################################################
    ### Neurons layers are defined, now set up connexions between them ###
    ######################################################################

    synapseCount = 0

    ############### Area V1 ##################

    sys.stdout.write('done. \nSetting up V1, Layers 4 and 6...')
    sys.stdout.flush()

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
                                           k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                           oriFilterWeight*filters1[k][i2+oriFilterSize/2][j2+oriFilterSize/2], 1.0))
                            if abs(filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2]) > 0.1:
                                ST2.append(((i+i2)*ImageNumPixelColumns + (j+j2),
                                            k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                            oriFilterWeight*filters2[k][i2+oriFilterSize/2][j2+oriFilterSize/2], 1.0))

                if len(ST)>0:
                    # LGN -> Layer 6 and 4 (simple cells) connections (no connections at the edges, to avoid edge-effects) first polarity filter
                    sim.Projection(LGNBright, V1Layer6P1, sim.FromListConnector(ST))
                    sim.Projection(LGNDark,   V1Layer6P2, sim.FromListConnector(ST))
                    sim.Projection(LGNBright, V1Layer4P1, sim.FromListConnector(ST))
                    sim.Projection(LGNDark,   V1Layer4P2, sim.FromListConnector(ST))
                    synapseCount += 4*len(ST)

                if len(ST2)>0:
                    # LGN -> Layer 6 and 4 (simple cells) connections (no connections at the edges, to avoid edge-effects) second polarity filter
                    sim.Projection(LGNBright, V1Layer6P2, sim.FromListConnector(ST2))
                    sim.Projection(LGNDark,   V1Layer6P1, sim.FromListConnector(ST2))
                    sim.Projection(LGNBright, V1Layer4P2, sim.FromListConnector(ST2))
                    sim.Projection(LGNDark,   V1Layer4P1, sim.FromListConnector(ST2))
                    synapseCount += 4*len(ST2)

    # Excitatory connection from same orientation and polarity 1, input from layer 6
    sim.Projection(V1Layer6P1, V1Layer4P1, sim.OneToOneConnector(weights=connections['V1_6To4Excite']))
    sim.Projection(V1Layer6P2, V1Layer4P2, sim.OneToOneConnector(weights=connections['V1_6To4Excite']))
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
                                           k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                           connections['V1_6To4Inhib'], 1.0))

    # Surround inhibition from layer 6 of same orientation and polarity
    sim.Projection(V1Layer6P1, V1Layer4P1, sim.FromListConnector(ST))
    sim.Projection(V1Layer6P2, V1Layer4P2, sim.FromListConnector(ST))
    synapseCount += 2*len(ST)

    sys.stdout.write('done. \nSetting up V1, Layers 23 and 6 (feedback)...')
    sys.stdout.flush()

    ST = []
    ST2 = []
    ST3 = []
    ST4E = []
    ST4I = []
    ST5E = []
    ST5I = []
    ST6 = []
    for k in range(0, numOrientations):                                # Orientations
        for i in range(0, numPixelRows):                               # Rows
            for j in range(0, numPixelColumns):                        # Columns
                for k2 in range(0, numOrientations):                   # Other orientations
                    if k != k2:
                        ST.append((k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                   k2*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                   connections['V1_CrossOriInhib'], 1.0))

                for i2 in range(-V1PoolSize/2+1, V1PoolSize/2+1):      # Filter rows (extra +1 to insure get top of odd-numbered filter)
                    for j2 in range(-V1PoolSize/2+1, V1PoolSize/2+1):  # Filter columns

                        if V1PoolingFilters[k][i2+V1PoolSize/2][j2+V1PoolSize/2] > 0:
                            if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                                ST2.append((k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                            k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                            connections['V1_ComplexExcite'], 1.0))
                                ST3.append((OppositeOrientationIndex[k]*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                            k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                            connections['V1_ComplexInhib'], 1.0))

                        if V1PoolingConnections1[k][i2+V1PoolSize/2][j2+V1PoolSize/2] > 0:
                            if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                                ST4E.append((k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                             k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                             connections['V1_FeedbackExcite'], 1.0))
                                ST4I.append((k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                             k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                             connections['V1_NegFeedbackInhib'], 1.0))


                        if V1PoolingConnections2[k][i2+V1PoolSize/2][j2+V1PoolSize/2] > 0:
                            if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                                ST5E.append((k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                             k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                             connections['V1_FeedbackExcite'], 1.0))
                                ST5I.append((k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                             k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                             connections['V1_NegFeedbackInhib'], 1.0))

                ST6.append((OppositeOrientationIndex[k]*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                            k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                            connections['V1_EndCutExcite'], 1.0))

    # Layer 4 -> Layer23 (complex cell connections)
    sim.Projection(V1Layer4P1, V1Layer23, sim.OneToOneConnector(weights=connections['V1_4To23Excite']))
    sim.Projection(V1Layer4P2, V1Layer23, sim.OneToOneConnector(weights=connections['V1_4To23Excite']))
    synapseCount += (len(V1Layer4P1)+len(V1Layer4P2))

    # Cross-orientation inhibition
    sim.Projection(V1Layer23, V1Layer23, sim.FromListConnector(ST))
    synapseCount += len(ST)

    # Pooling neurons in Layer 23 (excitation from same orientation, inhibition from orthogonal), V1PoolingFilters pools from both sides
    sim.Projection(V1Layer23, V1Layer23Pool, sim.FromListConnector(ST2))
    sim.Projection(V1Layer23, V1Layer23Pool, sim.FromListConnector(ST3))
    synapseCount += (len(ST2) + len(ST3))

    # Pooling neurons back to Layer 23 and to interneurons (ST4 for one side and ST5 for the other), V1poolingconnections pools from only one side
    sim.Projection(V1Layer23Pool, V1Layer23,       sim.FromListConnector(ST4E))
    sim.Projection(V1Layer23Pool, V1Layer23Inter1, sim.FromListConnector(ST4E))
    sim.Projection(V1Layer23Pool, V1Layer23Inter2, sim.FromListConnector(ST4I))
    sim.Projection(V1Layer23Pool, V1Layer23,       sim.FromListConnector(ST5E))
    sim.Projection(V1Layer23Pool, V1Layer23Inter2, sim.FromListConnector(ST5E))
    sim.Projection(V1Layer23Pool, V1Layer23Inter1, sim.FromListConnector(ST5I))
    synapseCount += 3*(len(ST4E) + len(ST5E))

    # Connect interneurons to complex cell and each other
    sim.Projection(V1Layer23Inter1, V1Layer23, sim.OneToOneConnector(weights=connections['V1_InterInhib']))
    sim.Projection(V1Layer23Inter2, V1Layer23, sim.OneToOneConnector(weights=connections['V1_InterInhib']))
    synapseCount += (len(V1Layer23Inter1) + len(V1Layer23Inter2))

    # End-cutting (excitation from orthogonal interneuron)
    sim.Projection(V1Layer23Inter1, V1Layer23, sim.FromListConnector(ST6))
    sim.Projection(V1Layer23Inter2, V1Layer23, sim.FromListConnector(ST6))
    synapseCount += 2*len(ST6)

    # Connect Layer 23 cells to Layer 6 cells (folded feedback)
    sim.Projection(V1Layer23, V1Layer6P1, sim.OneToOneConnector(weights=connections['V1_23To6Excite']))
    sim.Projection(V1Layer23, V1Layer6P2, sim.OneToOneConnector(weights=connections['V1_23To6Excite']))
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
                               h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                               connections['V1_ToV2Excite'], 1.0))

                    for i2 in range(-inhibRange64, inhibRange64+1):
                        for j2 in range(-inhibRange64, inhibRange64+1):
                            if i+i2 >=0 and i+i2 <numPixelRows and i2!=0 and j+j2 >=0 and j+j2 <numPixelColumns and j2!=0:
                                ST2.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                            h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                            connections['V2_6To4Inhib'], 1.0))

    # V2 Layers 4 and 6 connections
    sim.Projection(V1Layer23,  V2Layer6, sim.FromListConnector(ST))
    sim.Projection(V1Layer23,  V2Layer4, sim.FromListConnector(ST))
    sim.Projection(V2Layer6, V2Layer4, sim.OneToOneConnector(weights=connections['V2_6To4Excite']))
    synapseCount += (2*len(ST) + len(V2Layer6))

    # Surround inhibition V2 Layer 6 -> 4
    sim.Projection(V2Layer6,  V2Layer4, sim.FromListConnector(ST2))
    synapseCount += len(ST2)

    sys.stdout.write('done. \nSetting up V2, Layers 23 and 6 (feedback)...')
    sys.stdout.flush()

    ST = []
    ST2 = []
    ST3 = []
    ST4 = []
    ST5E = []
    ST5I = []
    ST6E = []
    ST6I = []
    for h in range(0, numSegmentationLayers):        # segmentation layers
        for k in range(0, numOrientations):          # Orientations
            for i in range(0, numPixelRows):         # Rows
                for j in range(0, numPixelColumns):  # Columns
                    ST.append((h*numOrientations*numPixelRows*numPixelColumns + OppositeOrientationIndex[k]*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                               h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                               connections['V2_CrossOriInhib'], 1.0))

                    for i2 in range(-V2PoolSize/2+1, V2PoolSize/2+1):      # Filter rows (extra +1 to insure get top of odd-numbered filter)
                        for j2 in range(-V2PoolSize/2+1, V2PoolSize/2+1):  # Filter columns

                            if V2PoolingFilters[k][i2+V2PoolSize/2][j2+V2PoolSize/2] > 0:
                                if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                                    ST2.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                                connections['V2_ComplexExcite'], 1.0))

                                    for k2 in range(0, numOrientations):
                                        if k2 != k:
                                            if k2 == OppositeOrientationIndex[k]:
                                                for h2 in range(numSegmentationLayers):  # for all segmentation layers
                                                    ST3.append((h2*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                                h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                                                connections['V2_ComplexInhib'], 1.0))
                                            else:
                                                ST4.append((h*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                            h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                                            connections['V2_ComplexInhib2'], 1.0))

                            if V2PoolingConnections1[k][i2+V2PoolSize/2][j2+V2PoolSize/2] > 0:
                                if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                                   ST5E.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                                h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                connections['V2_FeedbackExcite'], 1.0))
                                   ST5I.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                                h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                connections['V2_NegFeedbackInhib'], 1.0))

                            if V2PoolingConnections2[k][i2+V2PoolSize/2][j2+V2PoolSize/2] > 0:
                                if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                                   ST6E.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                                h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                connections['V2_FeedbackExcite'], 1.0))
                                   ST6I.append((h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                                h*numOrientations*numPixelRows*numPixelColumns + k*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                connections['V2_NegFeedbackInhib'], 1.0))

    # V2 Layer 4 -> V2 Layer23 (complex cell connections)
    sim.Projection(V2Layer4, V2Layer23, sim.OneToOneConnector(weights=connections['V2_4To23Excite']))
    synapseCount += len(V2Layer4)

    # Cross-orientation inhibition
    sim.Projection(V2Layer23, V2Layer23, sim.FromListConnector(ST))
    synapseCount += len(ST)

    # Pooling neurons in V2Layer 23 (excitation from same orientation, inhibition from different + stronger for orthogonal orientation)
    sim.Projection(V2Layer23, V2Layer23Pool, sim.FromListConnector(ST2))
    sim.Projection(V2Layer23, V2Layer23Pool, sim.FromListConnector(ST3))
    synapseCount += (len(ST2) + len(ST3))
    if len(ST4)>0:  # non-orthogonal inhibition
        sim.Projection(V2Layer23, V2Layer23Pool, sim.FromListConnector(ST4))
        synapseCount += len(ST4)

    # Pooling neurons back to Layer 23 and to interneurons (ST5 for one side and ST6 for the other), V1poolingconnections pools from only one side
    sim.Projection(V2Layer23Pool, V2Layer23,       sim.FromListConnector(ST5E))
    sim.Projection(V2Layer23Pool, V2Layer23Inter1, sim.FromListConnector(ST5E))
    sim.Projection(V2Layer23Pool, V2Layer23Inter2, sim.FromListConnector(ST5I))
    sim.Projection(V2Layer23Pool, V2Layer23,       sim.FromListConnector(ST6E))
    sim.Projection(V2Layer23Pool, V2Layer23Inter2, sim.FromListConnector(ST6E))
    sim.Projection(V2Layer23Pool, V2Layer23Inter1, sim.FromListConnector(ST6I))
    synapseCount += (3*len(ST5E) + 3*len(ST6E))

    # Connect interneurons to complex cell
    sim.Projection(V2Layer23Inter1, V2Layer23, sim.OneToOneConnector(weights=connections['V2_InterInhib']))
    sim.Projection(V2Layer23Inter2, V2Layer23, sim.OneToOneConnector(weights=connections['V2_InterInhib']))
    synapseCount += (len(V2Layer23Inter1) + len(V2Layer23Inter2))

    # Connect Layer 23 cells to Layer 6 cells (folded feedback)
    sim.Projection(V2Layer23, V2Layer6, sim.OneToOneConnector(weights=connections['V2_23To6Excite']))
    synapseCount += len(V2Layer23)

    # # Feedback from V2 to V1 (layer 6)
    # sim.Projection(V2Layer6, V1Layer6P1, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V2_ToV1FeedbackExcite']))
    # sim.Projection(V2Layer6, V1Layer6P2, sim.OneToOneConnector(), sim.StaticSynapse(weight=connections['V2_ToV1FeedbackExcite']))

    ############# Area V4 filling-in #############

    sys.stdout.write('done. \nSetting up V4...')
    sys.stdout.flush()

    ST = []
    ST2AE = []
    ST2AI = []
    ST2BE = []
    ST2BI = []
    ST3AE = []
    ST3AI = []
    ST3BE = []
    ST3BI = []
    ST4 = []
    for h in range(0, numSegmentationLayers):         # Segmentation layers
        for i in range(0, ImageNumPixelRows):         # Rows
            for j in range(0, ImageNumPixelColumns):  # Columns
                ST.append((i*ImageNumPixelColumns + j,
                           h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                           connections['LGN_ToV4Excite'], 1.0))

                for k in range(0, numFlows):  # Flow directions
                    ST2AE.append((h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                  h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                  connections['V4_BrightnessExcite'], 1.0))
                    ST2AI.append((h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                  h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                  connections['V4_BrightnessInhib'], 1.0))
                    ST2BE.append((h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                  h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                  connections['V4_BrightnessExcite'], 1.0))
                    ST2BI.append((h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                  h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                  connections['V4_BrightnessInhib'], 1.0))

                    # set up flow indices
                    i2 = flowFilter[k][0]
                    j2 = flowFilter[k][1]
                    if i + i2 >= 0 and i + i2 < ImageNumPixelRows and j + j2 >= 0 and j + j2 < ImageNumPixelColumns:
                        ST3AE.append((h*ImageNumPixelRows*ImageNumPixelColumns + (i+i2)*ImageNumPixelColumns + (j+j2),
                                      h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                      connections['V4_BrightnessExcite'], 1.0))
                        ST3AI.append((h*ImageNumPixelRows*ImageNumPixelColumns + (i+i2)*ImageNumPixelColumns + (j+j2),
                                      h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                      connections['V4_BrightnessInhib'], 1.0))
                        ST3BE.append((h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                      h*ImageNumPixelRows*ImageNumPixelColumns + (i+i2)*ImageNumPixelColumns + (j+j2),
                                      connections['V4_BrightnessExcite'], 1.0))
                        ST3BI.append((h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                      h*ImageNumPixelRows*ImageNumPixelColumns + (i+i2)*ImageNumPixelColumns + (j+j2),
                                      connections['V4_BrightnessInhib'], 1.0))

                    for k2 in range(0, len(BoundaryBlockFilter[k])):
                        for k3 in range(0, numOrientations):
                            if BoundaryBlockFilter[k][k2][0] != k3:
                                i2 = BoundaryBlockFilter[k][k2][1]
                                j2 = BoundaryBlockFilter[k][k2][2]
                                if i + i2 < numPixelRows and j + j2 < numPixelColumns:
                                    ST4.append((h*numOrientations*numPixelRows*numPixelColumns + k3*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                                connections['V2_BoundaryInhib'], 1.0))

    # Brightness and darkness at V4 compete
    sim.Projection(V4Darkness,   V4Brightness, sim.OneToOneConnector(weights=connections['V4_BetweenColorsInhib']))
    sim.Projection(V4Brightness, V4Darkness,   sim.OneToOneConnector(weights=connections['V4_BetweenColorsInhib']))
    synapseCount += (len(V4Darkness) + len(V4Brightness))

    # LGNBright->V4brightness and LGNDark->V4darkness
    sim.Projection(LGNBright, V4Brightness, sim.FromListConnector(ST))
    sim.Projection(LGNDark,   V4Darkness,   sim.FromListConnector(ST))
    synapseCount += 2*len(ST)

    # V4brightness<->Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
    sim.Projection(V4Brightness,       V4InterBrightness1, sim.FromListConnector(ST2AE))
    sim.Projection(V4Brightness,       V4InterBrightness2, sim.FromListConnector(ST2AI))
    sim.Projection(V4InterBrightness1, V4Brightness,       sim.FromListConnector(ST2BI))
    sim.Projection(V4InterBrightness2, V4Brightness,       sim.FromListConnector(ST2BE))
    synapseCount += 4*len(ST2AE)

    # V4darkness<->Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
    sim.Projection(V4Darkness,       V4InterDarkness1, sim.FromListConnector(ST2AE))
    sim.Projection(V4Darkness,       V4InterDarkness2, sim.FromListConnector(ST2AI))
    sim.Projection(V4InterDarkness1, V4Darkness,       sim.FromListConnector(ST2BI))
    sim.Projection(V4InterDarkness2, V4Darkness,       sim.FromListConnector(ST2BE))
    synapseCount += 4*len(ST2AE)

    # V4brightness neighbors<->Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
    sim.Projection(V4Brightness,       V4InterBrightness2, sim.FromListConnector(ST3AE))
    sim.Projection(V4Brightness,       V4InterBrightness1, sim.FromListConnector(ST3AI))
    sim.Projection(V4InterBrightness2, V4Brightness,       sim.FromListConnector(ST3BI))
    sim.Projection(V4InterBrightness1, V4Brightness,       sim.FromListConnector(ST3BE))
    synapseCount += 4*len(ST3AE)

    # V4darkness neighbors<->Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
    sim.Projection(V4Darkness,       V4InterDarkness2, sim.FromListConnector(ST3AE))
    sim.Projection(V4Darkness,       V4InterDarkness1, sim.FromListConnector(ST3AI))
    sim.Projection(V4InterDarkness2, V4Darkness,       sim.FromListConnector(ST3BI))
    sim.Projection(V4InterDarkness1, V4Darkness,       sim.FromListConnector(ST3BE))
    synapseCount += 4*len(ST3AE)

    # V2Layer23 -> V4 Interneurons (all boundaries block except for orientation of flow)
    sim.Projection(V2Layer23, V4InterBrightness1, sim.FromListConnector(ST4))
    sim.Projection(V2Layer23, V4InterBrightness2, sim.FromListConnector(ST4))
    sim.Projection(V2Layer23, V4InterDarkness1,   sim.FromListConnector(ST4))
    sim.Projection(V2Layer23, V4InterDarkness2,   sim.FromListConnector(ST4))
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
                                       (h2+1)*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                       connections['V2_SegmentInhib'], 1.0))

        # Boundaries in lower levels strongly inhibit boundaries in higher segmentation levels (lower levels can be inhibited by segmentation signals)
        sim.Projection(V2Layer23, V2Layer4, sim.FromListConnector(ST))
        synapseCount += len(ST)

    ########### Surface segmentation network ############

    if numSegmentationLayers>1 and useSurfaceSegmentation==1:
        sys.stdout.write('done. \nSetting up surface segmentation network...')
        sys.stdout.flush()

        STA = []
        STB = []
        ST2A = []
        ST2B = []
        ST3 = []
        ST4 = []
        ST5 = []
        for h in range(0, numSegmentationLayers-1):         # Segmentation layers (not including baseline layer)
            for i in range(0, ImageNumPixelRows):           # Rows
                for j in range(0, ImageNumPixelColumns):    # Columns
                    for k in range(0, numFlows):            # Flow directions
                        STA.append((h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                    h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                    connections['S_SegmentSpreadExcite'], 1.0))
                        STB.append((h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                    h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                    connections['S_SegmentSpreadExcite'], 1.0))

                        i2 = flowFilter[k][0]               # Vertical flow indices (surface segmentation signal flows through closed shapes)
                        j2 = flowFilter[k][1]               # Horizontal flow indices
                        if i + i2 >= 0 and i + i2 < ImageNumPixelRows and j + j2 >= 0 and j + j2 < ImageNumPixelColumns:
                            ST2A.append((h*ImageNumPixelRows*ImageNumPixelColumns + (i+i2)*ImageNumPixelColumns + (j+j2),
                                         h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                         connections['S_SegmentSpreadExcite'], 1.0))
                            ST2B.append((h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                         h*ImageNumPixelRows*ImageNumPixelColumns + (i+i2)*ImageNumPixelColumns + (j+j2),
                                         connections['S_SegmentSpreadExcite'], 1.0))

                        for k2 in range(0, len(BoundaryBlockFilter[k])):
                            for k3 in range(0, numOrientations):
                                if BoundaryBlockFilter[k][k2][0] != k3:
                                    i2 = BoundaryBlockFilter[k][k2][1]
                                    j2 = BoundaryBlockFilter[k][k2][2]
                                    if i + i2 < numPixelRows and j + j2 < numPixelColumns:
                                        for h2 in range(0, numSegmentationLayers):  # draw boundaries from all segmentation layers
                                            ST3.append((h2*numOrientations*numPixelRows*numPixelColumns + k3*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                        h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                                        connections['V2_BoundaryInhib'], 1.0))

                    for k2 in range(0, numOrientations):
                        for h2 in range(h, numSegmentationLayers-1):
                            ST4.append((h*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + i*numPixelColumns + j,
                                        (h2+1)*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + i*numPixelColumns + j))

                        for i2 in range(-2, 4):  # offset by (1,1) to reflect boundary grid is offset from surface grid
                            for j2 in range(-2, 4):
                                if i + i2 >= 0 and i + i2 < ImageNumPixelRows and j + j2 >= 0 and j + j2 < ImageNumPixelColumns:
                                    for h2 in range(0, h+1):
                                        ST5.append((h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                                    h2*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                    connections['S_SegmentGatingInhib'], 1.0))

        # Off signals inhibit On Signals (can be separated by boundaries)
        sim.Projection(SurfaceSegmentationOff,       SurfaceSegmentationOn,  sim.OneToOneConnector(weights=connections['S_SegmentOnOffInhib']))
        synapseCount += len(SurfaceSegmentationOff)

        # SurfaceSegmentationOn/Off <-> Interneurons ; fliplr to use the connections in the way "target indexes --> source indexes"
        sim.Projection(SurfaceSegmentationOn,        SurfaceSegmentationOnInter1,  sim.FromListConnector(STA))
        sim.Projection(SurfaceSegmentationOnInter2,  SurfaceSegmentationOn,        sim.FromListConnector(STB))
        sim.Projection(SurfaceSegmentationOff,       SurfaceSegmentationOffInter1, sim.FromListConnector(STA))
        sim.Projection(SurfaceSegmentationOffInter2, SurfaceSegmentationOff,       sim.FromListConnector(STB))
        synapseCount += 4*len(STA)

        # SurfaceSegmentationOn/Off <-> Interneurons (flow out on interneuron 1 flow in on interneuron 2) ; fliplr to use the connections in the way "target indexes --> source indexes"
        sim.Projection(SurfaceSegmentationOn,        SurfaceSegmentationOnInter2,  sim.FromListConnector(ST2A))
        sim.Projection(SurfaceSegmentationOnInter1,  SurfaceSegmentationOn,        sim.FromListConnector(ST2B))
        sim.Projection(SurfaceSegmentationOff,       SurfaceSegmentationOffInter2, sim.FromListConnector(ST2A))
        sim.Projection(SurfaceSegmentationOffInter1, SurfaceSegmentationOff,       sim.FromListConnector(ST2B))
        synapseCount += 4*len(ST2A)

        # V2Layer23 -> Segmentation Interneurons (all boundaries block except for orientation of flow)
        sim.Projection(V2Layer23, SurfaceSegmentationOnInter1,  sim.FromListConnector(ST3))
        sim.Projection(V2Layer23, SurfaceSegmentationOnInter2,  sim.FromListConnector(ST3))
        sim.Projection(V2Layer23, SurfaceSegmentationOffInter1, sim.FromListConnector(ST3))
        sim.Projection(V2Layer23, SurfaceSegmentationOffInter2, sim.FromListConnector(ST3))
        synapseCount += 4*len(ST3)

        # V2Layer23 -> V2Layer4 strong inhibition (CHECK WHY THIS CONNECTION IS THERE TWICE WITH THE SAME CONNECTION PATTERN)
        # sim.Projection(V2Layer23, V2Layer4, sim.FromListConnector(ST4), sim.StaticSynapse(weight=connections['V2_SegmentInhib']))
        # synapseCount += len(ST4)

        # Segmentation -> V2Layer4 (gating) ; way for lower levels to be inhibited by higher ones : through segmentation network) SHOULDN'T IT BE ACTIVATORY????
        sim.Projection(SurfaceSegmentationOn, V2Layer4, sim.FromListConnector(ST5))
        synapseCount += len(ST5)

    ########### Boundary segmentation network ############

    if numSegmentationLayers>1 and useBoundarySegmentation==1:
        sys.stdout.write('done. \nSetting up boundary segmentation network...')
        sys.stdout.flush()

        STA = []
        STB = []
        ST2A = []
        ST2B = []
        ST3 = []
        ST4 = []
        for h in range(0, numSegmentationLayers-1):  # Num layers (not including baseline layer)
            for i in range(0, ImageNumPixelRows):  # Rows
                for j in range(0, ImageNumPixelColumns):  # Columns
                    for k in range(0, numFlows):  # Flow directions
                        STA.append((h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                    h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                    connections['B_SegmentSpreadExcite'], 1.0))
                        STB.append((h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                    h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                    connections['B_SegmentSpreadExcite'], 1.0))

                        i2 = flowFilter[k][0]
                        j2 = flowFilter[k][1]
                        if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                            ST2A.append((h*ImageNumPixelRows*ImageNumPixelColumns + (i+i2)*ImageNumPixelColumns + (j+j2),
                                         h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                         connections['B_SegmentSpreadExcite'], 1.0))
                            ST2B.append((h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                         h*ImageNumPixelRows*ImageNumPixelColumns + (i+i2)*ImageNumPixelColumns + (j+j2),
                                         connections['B_SegmentSpreadExcite'], 1.0))

                        for k2 in range(0, len(BoundaryBlockFilter[k])):
                            i2 = BoundaryBlockFilter[k][k2][1]
                            j2 = BoundaryBlockFilter[k][k2][2]
                            if i+i2 < numPixelRows and j+j2 < numPixelColumns:
                                for k3 in range(0, numOrientations):
                                    for h2 in range(0, numSegmentationLayers):  # draw boundaries from all segmentation layers
                                        ST3.append((h2*numOrientations*numPixelRows*numPixelColumns + k3*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                    h*numFlows*ImageNumPixelRows*ImageNumPixelColumns + k*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                                    connections['B_SegmentOpenFlowInhib'], 1.0))

                    for k2 in range(0, numOrientations):
                        for i2 in range(-2, 4):  # offset by (1, 1) to reflect boundary grid is offset from surface grid
                            for j2 in range(-2, 4):
                                if i+i2 >= 0 and i+i2 < ImageNumPixelRows and j+j2 >= 0 and j+j2 < ImageNumPixelColumns:
                                    for h2 in range(0, h+1):
                                        ST4.append((h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j,
                                                    h2*numOrientations*numPixelRows*numPixelColumns + k2*numPixelRows*numPixelColumns + (i+i2)*numPixelColumns + (j+j2),
                                                    connections['B_SegmentGatingInhib'], 1.0))

        # BoundarySegmentationOn<->Interneurons (flow out on interneuron 1 flow in on interneuron 2)
        sim.Projection(BoundarySegmentationOn,       BoundarySegmentationOnInter1, sim.FromListConnector(STA))
        sim.Projection(BoundarySegmentationOnInter2, BoundarySegmentationOn,       sim.FromListConnector(STB))
        synapseCount += 2*len(STA)

        sim.Projection(BoundarySegmentationOn,       BoundarySegmentationOnInter2, sim.FromListConnector(ST2A))
        sim.Projection(BoundarySegmentationOnInter1, BoundarySegmentationOn,       sim.FromListConnector(ST2B))
        synapseCount += 2*len(ST2A)

        # Inhibition from third interneuron (itself inhibited by the presence of a boundary)
        sim.Projection(BoundarySegmentationOnInter3, BoundarySegmentationOnInter1, sim.OneToOneConnector(weights=connections['B_SegmentTonicInhib']))
        sim.Projection(BoundarySegmentationOnInter3, BoundarySegmentationOnInter2, sim.OneToOneConnector(weights=connections['B_SegmentTonicInhib']))
        synapseCount += 2*len(BoundarySegmentationOnInter3)

        # V2layer23 -> Segmentation Interneurons (all boundaries open flow by inhibiting third interneuron)
        # BoundarySegmentationOnInter3.inject(sim.DCSource(amplitude=1000.0, start=0.0, stop=0.0))
        sim.Projection(V2Layer23, BoundarySegmentationOnInter3, sim.FromListConnector(ST3))
        synapseCount += len(ST3)

        # BoundarySegmentation -> V2layer4 (gating)
        sim.Projection(BoundarySegmentationOn, V2Layer4, sim.FromListConnector(ST4))
        synapseCount += len(ST4)

    sys.stdout.write('done. \n'+str(synapseCount)+' network connections created.\n')
    sys.stdout.flush()

    # Return only the populations that need to be updated online during the simulation and that we want to make a gif of
    # fullNet = LGNBright + LGNDark \
    #           + V1Layer6P1 + V1Layer6P2 + V1Layer4P1 + V1Layer4P2 + V1Layer23 + V1Layer23Pool + V1Layer23Inter1 + V1Layer23Inter2 \
    #           + V2Layer6 + V2Layer4 + V2Layer23 + V2Layer23Pool + V2Layer23Inter1 + V2Layer23Inter2 \
    #           + V4Brightness + V4Darkness + V4InterBrightness1 + V4InterBrightness2 + V4InterDarkness1 + V4InterDarkness2
    # netToSend = LGNBright + LGNDark + fullNet.get_population("V1LayerToPlot") + fullNet.get_population("V2LayerToPlot") + V4Brightness + V4Darkness
    # if useSurfaceSegmentation:
    #     fullNet += (SurfaceSegmentationOn + SurfaceSegmentationOnInter1 + SurfaceSegmentationOnInter2
    #                 + SurfaceSegmentationOff + SurfaceSegmentationOffInter1 + SurfaceSegmentationOffInter2)
    #     netToSend += (fullNet.get_population("SurfaceSegmentationOn") + fullNet.get_population("SurfaceSegmentationOff"))
    # if useBoundarySegmentation:
    #     fullNet += (BoundarySegmentationOn + BoundarySegmentationOnInter1 + BoundarySegmentationOnInter2 + BoundarySegmentationOnInter3
    #                     + BoundarySegmentationOff + BoundarySegmentationOffInter1 + BoundarySegmentationOffInter2)
    #     netToSend += BoundarySegmentationOnInter3 + BoundarySegmentationOn + BoundarySegmentationOff

    fullNet = (LGNBright, LGNDark
             , V1Layer6P1, V1Layer6P2, V1Layer4P1, V1Layer4P2, V1Layer23, V1Layer23Pool, V1Layer23Inter1, V1Layer23Inter2 \
             , V2Layer6, V2Layer4, V2Layer23, V2Layer23Pool, V2Layer23Inter1, V2Layer23Inter2 \
             , V4Brightness, V4Darkness, V4InterBrightness1, V4InterBrightness2, V4InterDarkness1, V4InterDarkness2)
    netToSend = LGNBright, LGNDark, V1Layer23, V2Layer23, V4Brightness, V4Darkness
    if useSurfaceSegmentation:
        fullNet += (SurfaceSegmentationOn, SurfaceSegmentationOnInter1, SurfaceSegmentationOnInter2,
                    SurfaceSegmentationOff, SurfaceSegmentationOffInter1, SurfaceSegmentationOffInter2)
        netToSend += (SurfaceSegmentationOn, SurfaceSegmentationOff)
    if useBoundarySegmentation:
        fullNet += (BoundarySegmentationOn, BoundarySegmentationOnInter1, BoundarySegmentationOnInter2,          BoundarySegmentationOnInter3,
                        BoundarySegmentationOff, BoundarySegmentationOffInter1, BoundarySegmentationOffInter2)
        netToSend += (BoundarySegmentationOnInter3, BoundarySegmentationOn, BoundarySegmentationOff)
    return netToSend
