# Imported Python Transfer Function
import numpy as np, sensor_msgs.msg
from cv_bridge import CvBridge
@nrp.MapVariable("cumplotDensityV1", scope=nrp.GLOBAL, initial_value=[[[0 for col in range(6)] for row in range(6)] for ori in range(2)])
@nrp.MapSpikeSink("V1LayerToPlot", nrp.brain.V1Layer23, nrp.spike_recorder)
@nrp.MapRobotPublisher('output_segmented', Topic('/robot/output_segmented', sensor_msgs.msg.Image))
@nrp.Neuron2Robot()
def plot_V1_activity(t, cumplotDensityV1, V1LayerToPlot, output_segmented):
    import numpy as np
    min_idx = 372
    numOrientations = 2
    numPixelRows = 5+1
    numPixelColumns = 5+1
    # Record spike count up to now from V1
    plotDensityV1 = [[[0 for col in range(numPixelColumns)] for row in range(numPixelRows)] for ori in range(numOrientations)]
    plotDensityV1 =  np.zeros(2*6*6)
    for (idx, time) in V1LayerToPlot.times.tolist():
        if idx - min_idx <= 0 or idx - min_idx > 72:
            continue
        plotDensityV1[int(idx) - min_idx] = plotDensityV1[int(idx) - min_idx] + 1
    plotDensityV1 = np.reshape(plotDensityV1, (2,6,6))
    # V1 sampling and update of the cumulative spike count
#    for k in range(numOrientations):                    # Orientations
#        for i in range(numPixelRows):                   # Rows
#            for j in range(numPixelColumns):            # Columns
#                plotDensityV1[k][i][j] = plotDensityV1[k][i][j] + V1SpikeCountUpToNow[k*numPixelRows*numPixelColumns + i*numPixelColumns + j] - cumplotDensityV1.value[k][i][j]
#                cumplotDensityV1.value[k][i][j] = cumplotDensityV1.value[k][i][j] + plotDensityV1[k][i][j]  # update cumulative spike count
    # Set up an image to plot for V1 oriented neurons activity
    dataV1 = np.zeros((numPixelRows, numPixelColumns,3), dtype=np.uint8)
    for i in range(numPixelRows):         # Rows
        for j in range(numPixelColumns):  # Columns
            if numOrientations==2:          # Vertical and horizontal
                dataV1[i][j] = [10*plotDensityV1[0][i][j], 10*plotDensityV1[1][i][j], 0]
            if numOrientations==4:          # Vertical, horizontal, both diagonals
                diagV1 = max(plotDensityV1[0][i][j], plotDensityV1[2][i][j])
                dataV1[i][j] = [plotDensityV1[1][i][j], plotDensityV1[3][i][j], diagV1]
    msg_frame = CvBridge().cv2_to_imgmsg(dataV1, 'rgb8')
    output_segmented.send_message(msg_frame)
    # Publish the V1 activity density map
    # csv_recorder.record_entry(*dataV1)
