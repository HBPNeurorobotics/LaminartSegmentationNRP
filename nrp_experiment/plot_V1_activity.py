# Imported Python Transfer Function
import numpy as np, sensor_msgs.msg
from cv_bridge import CvBridge
@nrp.MapSpikeSink("V2LayerToPlot", nrp.brain.V2Layer23, nrp.spike_recorder)
@nrp.MapRobotPublisher('output_segmented_1', Topic('/robot/output_segmented_1', sensor_msgs.msg.Image))
@nrp.MapRobotPublisher('output_segmented_2', Topic('/robot/output_segmented_2', sensor_msgs.msg.Image))
@nrp.MapRobotPublisher('output_segmented_3', Topic('/robot/output_segmented_3', sensor_msgs.msg.Image))
@nrp.Neuron2Robot()
def plot_V1_activity(t, V2LayerToPlot, output_segmented_1, output_segmented_2, output_segmented_3):
    import numpy as np
    min_idx = int(V2LayerToPlot.neurons[0])
    nOri = nrp.config.brain_root.numOrientations
    numPixelRows = nrp.config.brain_root.ImageNumPixelRows + 1
    numPixelColumns =  nrp.config.brain_root.ImageNumPixelColumns + 1
    nNeuronsPerSegLayer = nOri * numPixelRows * numPixelColumns
    output_segmented = [ output_segmented_1,
                         output_segmented_2,
                         output_segmented_3 ]
    # Record spike count up to now from V1
    plotDensityV1 =  np.zeros(3*nNeuronsPerSegLayer)
    for (idx, time) in V2LayerToPlot.times.tolist():
        plotDensityV1[int(idx) - min_idx] = plotDensityV1[int(idx) - min_idx] + 1
    for seglayer in range(3):
        #seglayerPlot = plotDensityV1[seglayer * nNeuronsPerSegLayer : (seglayer + 1) * nNeuronsPerSegLayer]
        seglayerPlot = plotDensityV1[0:nNeuronsPerSegLayer]
        seglayerPlot = np.reshape(seglayerPlot, (nOri,numPixelRows,numPixelColumns))
        # Set up an image to plot for V1 oriented neurons activity
        dataV1 = np.zeros((numPixelRows, numPixelColumns,3), dtype=np.uint8)
        for i in range(numPixelRows):         # Rows
            for j in range(numPixelColumns):  # Columns
                if nOri==2:          # Vertical and horizontal
                    dataV1[i][j] = [20*seglayerPlot[0][i][j], 20*seglayerPlot[1][i][j], 0]
                if nOri==4:          # Vertical, horizontal, both diagonals
                    diagV1 = max(seglayerPlot[0][i][j], seglayerPlot[2][i][j])
                    dataV1[i][j] = [seglayerPlot[1][i][j], seglayerPlot[3][i][j], diagV1]
        msg_frame = CvBridge().cv2_to_imgmsg(dataV1, 'rgb8')
        output_segmented[seglayer].send_message(msg_frame)
        # Publish the V1 activity density map
        # csv_recorder.record_entry(*dataV1)
