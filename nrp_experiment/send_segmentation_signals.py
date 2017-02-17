# Imported Python Transfer Function
# Input transfer function that sends segmentation signals, following top-down and bottom-up cues (dumb choice of position for now)
@nrp.MapSpikeSource("BoundarySegmentationSignal",   nrp.map_neurons(
    range((nrp.config.brain_root.numSegmentationLayers-1) * nrp.config.brain_root.ImageNumPixelRows * nrp.config.brain_root.ImageNumPixelColumns),
    lambda i: nrp.brain.BoundarySegmentationOn[i]), nrp.dc_source)
@nrp.Robot2Neuron()
def send_segmentation_signals(t, BoundarySegmentationSignal):
    return
    if t < 3:
        return

    import random
    import numpy
    # Loop through all non-basic segmentation layers and send signals around top-down / bottom-up selected targets -->
    surfaceOnTarget = []
    surfaceOffTarget = []
    boundaryOnTarget = []
    ImageNumPixelRows = nrp.config.brain_root.ImageNumPixelRows
    ImageNumPixelColumns = nrp.config.brain_root.ImageNumPixelColumns
    numSegmentationLayers = nrp.config.brain_root.numSegmentationLayers
    for h in range(numSegmentationLayers-1):
        # Look for the best place to send a segmentation signal (DUMB CRITERION) -->
        # HERE WE SHOULD DEVELOP THE CRITERION (TASK-RELATED SALIENCY MAX) AND ALSO IMPLEMENT AND TASK RELATED TOP-DOWN SIGNALS
        segmentLocationRow = int(random.random() * ImageNumPixelRows)
        segmentLocationCol = int(random.random() * ImageNumPixelColumns)
        for i in range(0, ImageNumPixelRows):         # Rows
            for j in range(0, ImageNumPixelColumns):  # Columns
                BoundarySegmentationSignal[h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j].amplitude = 0.
                # On signals start the boundary segmentation at a specific location and flow along connected boundaries
                distance = numpy.sqrt(numpy.power(segmentLocationRow-i, 2) + numpy.power(segmentLocationCol-j, 2))
                if distance > nrp.config.brain_root.segmentationSignalSize:
                    boundaryOnTarget.append(h*ImageNumPixelRows*ImageNumPixelColumns + i*ImageNumPixelColumns + j)
    # Set a firing positive firing rate for concerned units of the segmentation top-down signal
    if len(boundaryOnTarget) > 0:
        BoundarySegmentationSignal[boundaryOnTarget].amplitude = 1.0
