# Imports
import numpy

# Function 1: Take filter parameters and build 2 oriented filters with different polarities for connection pattern from the LGN to V1
# Usage : filters1, filters2 = createFilters(numOrientations=8, size=4, sigma2=0.75, Olambda=4)
def createFilters(numOrientations, size, sigma2, Olambda):

    # Initialize the filters
    filters1 = numpy.zeros((numOrientations, size, size))
    filters2 = numpy.zeros((numOrientations, size, size))

    # Fill them with gabors
    midSize = (size-1.)/2.
    maxValue = -1
    for k in range(0, numOrientations):
        theta = numpy.pi * (k + 1) / numOrientations
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


# Function 2: Take filter parameters and build connection pooling and connection filters arrays
# Usage (for V1 e.g.) : V1poolingfilters, V1poolingconnections1, V1poolingconnections2 = createPoolConnAndFilters(numOrientations=8, VPoolSize=3, sigma2=4.0, Olambda=5)
# Usage (for V2 e.g.) : V2poolingfilters, V2poolingconnections1, V2poolingconnections2 = createPoolConnAndFilters(numOrientations=8, VPoolSize=7, sigma2=26.0, Olambda=9)
def createPoolingConnectionsAndFilters(numOrientations, VPoolSize, sigma2, Olambda):

    # Set up layer23 pooling filters
    # Set up orientation kernels
    midSize = (VPoolSize - 1.) / 2.
    Vpoolingfilters = numpy.zeros((numOrientations, VPoolSize, VPoolSize))
    for k in range(0, numOrientations):
        theta = numpy.pi*(k+1)/numOrientations
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
        theta = 180*(k+1)/numOrientations
        if min(abs(theta-0),abs(theta-180)) < min(abs(theta-90),abs(theta-270)): # more horizontally oriented filter
            for i in range(0, VPoolSize):
                check = Vpoolingfilters[k,i:min(i+2,VPoolSize),:]
                Vpoolingfilters[k,min(i+1,VPoolSize-1),numpy.where(numpy.sum(check,axis=0) == 2.0)] = 0
        else:                                                                    # more vertically oriented filter
            for j in range(0, VPoolSize):
                check = Vpoolingfilters[k,:,j:min(j+2,VPoolSize)]
                Vpoolingfilters[k,numpy.where(numpy.sum(check,axis=1)==2.0),min(j+1,VPoolSize-1),] = 0

    # ==================================
    # Set up layer23 pooling cell connections (connect to points at either extreme of pooling line)
    # Set up orientation kernels

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