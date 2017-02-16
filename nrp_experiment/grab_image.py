# Imported Python Transfer Function
# Input transfer function that transforms the camera input into a DC current source fed to LGN neurons
import numpy as np, sensor_msgs.msg
from cv_bridge import CvBridge
@nrp.MapRobotSubscriber("camera", Topic("/icub_model/left_eye_camera/image_raw", sensor_msgs.msg.Image))
@nrp.MapSpikeSource("LGNBrightInput", nrp.map_neurons(range(0, nrp.config.brain_root.ImageNumPixelRows * nrp.config.brain_root.ImageNumPixelColumns), lambda i: nrp.brain.LGNBright[i]), nrp.dc_source)
@nrp.MapSpikeSource("LGNDarkInput",   nrp.map_neurons(range(0, nrp.config.brain_root.ImageNumPixelRows * nrp.config.brain_root.ImageNumPixelColumns), lambda i: nrp.brain.LGNDark[i]),   nrp.dc_source)
@nrp.Robot2Neuron()
def grab_image(t, camera, LGNBrightInput, LGNDarkInput):
    image = camera.value

    if image is not None:
        nRows = nrp.config.brain_root.ImageNumPixelRows
        nCols = nrp.config.brain_root.ImageNumPixelColumns

        # Read the image into an array, mean over 3 colors, resize it to the dimensions of the network and flatten the result
        import scipy.ndimage.interpolation
        imgIn = np.mean(CvBridge().imgmsg_to_cv2(image, "rgb8"), axis=2)
        resizeFactor = (float(nRows)/imgIn.shape[0], float(nCols)/imgIn.shape[1])
        imgResized = scipy.ndimage.interpolation.zoom(imgIn, resizeFactor, order=3).flatten()
        # Give the pre-processed image to the LGN (bright and dark inputs)
        LGNBrightInput.amplitude = 10.0*np.maximum(0.0, (imgResized/127.0-1.0))
        LGNDarkInput  .amplitude = 10.0*np.maximum(0.0, 1.0-(imgResized/127.0))
