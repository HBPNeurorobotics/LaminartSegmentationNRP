@nrp.MapVariable("params", scope=nrp.GLOBAL, initial_value=None)
@nrp.Robot2Neuron()
def init_params(t, params):
#    params.value = {
#               'nCols': nrp.brain.ImageNumPixelColumns,
#            'nRows': nrp.brain.ImageNumPixelRows,
#            'nOri': nrp.brain.numOrientations,
#        }
        if params.value is None:
                params.value = {
                        'nCols': 10,
                        'nRows': 10,
                        'nOri': 2
                }
