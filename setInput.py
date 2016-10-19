# Several function loading the stimulus from any BMP into the simulation of LaminartWithSegmentation.py

######################################################################################
### SEGMENTATION SIGNAL LOCATIONS GO IN : def chooseSegmentationSignal (see below) ###
### ADDITIONAL CROP DEFINITIONS GO IN : def chooseAddCrop (see below)              ###
######################################################################################

# Imports:
import os
import shutil
import numpy
from PIL import Image


# Function made to create a directory (/RetinaInput) and put retina input images there, wrt the stimulus
# Pixels is created by the function below (this could be done better)
def createRetinaInput(pixels, numTrials, stimulusTimeSteps, InterTrialInterval):

    # Create a blank (all black) input for times when stimulus does not appear
    blankInput = numpy.zeros(numpy.shape(pixels), dtype=numpy.uint8) # used for inter trial intervals or just when stimulus stops

    # Find the current path and create a new folder to host the retina input
    dirPath = os.getcwd()+'/RetinaInput'
    if os.path.exists(dirPath):
        shutil.rmtree(dirPath) # delete the retina input of the previous simulation (if there)
    os.makedirs(dirPath)

    for trial in range(numTrials):

        # Fill the directory with input images
        for time in range(0, int(stimulusTimeSteps)):
            img = Image.fromarray(numpy.uint8(pixels)) # if time < 3 else Image.fromarray(blankInput)
            img.save(dirPath+"/input"+str(trial)+"A"+str(time)+".png")

        # Fill the directory with blank images (inter trial interval)
        for time in range(0, int(InterTrialInterval)):
            img = Image.fromarray(blankInput)
            img.save(dirPath+"/input"+str(trial)+"B"+str(time)+".png")


# Function to read any BMP file as an image and transform it into a numpy array ; crops the white background
# Intended to be used by the script LaminartWithSegmentation.py (and consor)
# onlyZerosAndOnes --> 0 if you want aliasing, 1 if you want no aliasing, 2 if you want (aliased+notaliased)/2
def readAndCropBMP(thisConditionName, onlyZerosAndOnes=0):

    # Read the image and transforms it into a [0 to 254] array
    dirPath = os.getcwd() + "/Stimuli/"
    im = Image.open(dirPath+thisConditionName+".bmp")
    im = numpy.array(im)
    if len(numpy.shape(im)) > 2:                # if the pixels are [r,g,b] and not simple integers
        im = numpy.mean(im,2)                   # mean each pixel to integer (r+g+b)/3
    if onlyZerosAndOnes == 1:
        im = numpy.round(im/im.max())*im.max()  # either 0 or im.max()
    if onlyZerosAndOnes == 2:
        im = 0.5*(im + numpy.round(im/im.max())*im.max())  # less aliasing ...
    im *= 254.0/im.max()                        # array of floats between 0.0 and 254.0 (more useful for the Laminart script)
    white = im.max()                            # to detect the parts to crop

    ## Cropping:

    # Remove the upper white background (delete white rows above)
    usefulImg = numpy.where(im[:,int(len(im[0,:])/2)] != white)[0]
    indexesToRemove = range(usefulImg[0])
    im = numpy.delete(im,indexesToRemove,axis=0)

    # Remove the left and right white background (delete white columns left and right)
    usefulImg = numpy.where(im[0,:] != white)[0]
    indexesToRemove = numpy.append(numpy.array(range(usefulImg[0])),numpy.array(range(usefulImg[-1]+1,len(im[0,:]))))
    im = numpy.delete(im,indexesToRemove,axis=1)

    # Remove the bottom white background (delete white rows below)
    usefulImg = numpy.where(im[:,0] != white)[0]
    indexesToRemove = range(usefulImg[-1]+1,len(im[:,0]))
    im = numpy.delete(im,indexesToRemove,axis=0)

    # Crops the image even more, if wanted (for computation time)
    addCropX, addCropY = chooseAddCrop(thisConditionName)
    if addCropX != 0:
        indexesToRemove = numpy.append(numpy.array(range(addCropX)),numpy.array(range(len(im[0,:])-addCropX,len(im[0,:]))))
        im = numpy.delete(im,indexesToRemove,axis=1) # horizontal crop (remove columns on both sides)
    if addCropY != 0:
        indexesToRemove = numpy.append(numpy.array(range(addCropY)),numpy.array(range(len(im[:,0])-addCropY,len(im[:,0]))))
        im = numpy.delete(im,indexesToRemove,axis=0) # vertical crop (remove lines on both sides)

    return im, numpy.shape(im)[0], numpy.shape(im)[1]


# Find the vernier location into pixels, and create a template to compute a template match score
def findVernierAndCreateTemplates(pixels, TemplateSize, vSide, vbarlength, vbarwidth, barshift):

    # V4 Brightness template for a vernier (left or right is for bottom line)
    ImageNumPixelRows, ImageNumPixelColumns = numpy.shape(pixels)
    RightVernierTemplate = numpy.zeros((ImageNumPixelRows, ImageNumPixelColumns))
    LeftVernierTemplate = numpy.zeros((ImageNumPixelRows, ImageNumPixelColumns))

    # Randomly choose the vernier side (1/2 chance for each side)
    vSide = numpy.int(vSide/2 + 0.5)  # 1 for right bottom bar, 0 for left (from 1 to 1 and -1 to 0)

    # Create the corresponding vernier filter
    [Vx, Vy] = [2 * vbarlength, 2 * vbarwidth + barshift]  # usually [2*10,3]
    vernierFilter = -numpy.ones([Vx, Vy])
    vernierFilter[0:vbarlength, 0 * vSide + 2 * (1 - vSide)] = 1.0
    vernierFilter[vbarlength:2 * vbarlength, 2 * vSide + 0 * (1 - vSide)] = 1.0

    # # Create a Vernier filter
    # [Vx, Vy] = [2 * vbarlength, 2 * vbarwidth + barshift]  # usually [2*10,3]
    # vernierFilter = -numpy.ones([Vx, Vy])
    # vernierFilter[0:vbarlength, 0] = 1.0  # takes 0 but not vbarlength
    # vernierFilter[vbarlength:2 * vbarlength, 2] = 1.0  # takes vbarlength but not 2*vbarlength

    # Check for the Vernier location on the stimulus
    matchMax = 0
    [vernierLocX, vernierLocY] = [0, 0]  # location of the top-left corner of the vernier on the stimulus
    for i in range(ImageNumPixelRows - Vx + 1):
        for j in range(ImageNumPixelColumns - Vy + 1):
            match = numpy.sum(vernierFilter * pixels[i:i + Vx, j:j + Vy])
            if match > matchMax:
                matchMax = match
                [vernierLocY, vernierLocX] = [i, j]  # LocX corresponds to a column and LocY to a row

    # Bottom right
    firstRow = max(vernierLocY + vbarlength, 0)
    lastRow = min(firstRow + TemplateSize, ImageNumPixelRows)
    firstColumn = max(vernierLocX + 2, 0)
    lastColumn = min(firstColumn + TemplateSize, ImageNumPixelColumns)
    RightVernierTemplate[firstRow:lastRow, firstColumn:lastColumn] = 1.0

    # Top left
    lastRow = min(vernierLocY + vbarlength, ImageNumPixelRows)
    firstRow = max(lastRow - TemplateSize, 0)
    lastColumn = min(vernierLocX + 1, ImageNumPixelColumns)
    firstColumn = max(lastColumn - TemplateSize, 0)
    RightVernierTemplate[firstRow:lastRow, firstColumn:lastColumn] = 1.0

    # Bottom left
    firstRow = max(vernierLocY + vbarlength, 0)
    lastRow = min(firstRow + TemplateSize, ImageNumPixelRows)
    lastColumn = min(vernierLocX + 1, ImageNumPixelColumns)
    firstColumn = max(lastColumn - TemplateSize, 0)
    LeftVernierTemplate[firstRow:lastRow, firstColumn:lastColumn] = 1.0

    # Top right
    lastRow = min(vernierLocY + vbarlength, ImageNumPixelRows)
    firstRow = max(lastRow - TemplateSize, 0)
    firstColumn = max(vernierLocX + 2, 0)
    lastColumn = min(firstColumn + TemplateSize, ImageNumPixelColumns)
    LeftVernierTemplate[firstRow:lastRow, firstColumn:lastColumn] = 1.0

    # Return the position of the vernier and the created template
    return vernierLocX, vernierLocY, RightVernierTemplate, LeftVernierTemplate


# Randomize the side of the vernier inside the stimulus, useful in case of asymmetric flankers
def randomizeVernierSide(pixels, vernierLocX, vernierLocY, vbarlength, vbarwidth, barshift):

    # Find background and target colors
    white = pixels.max()  # target
    dark = pixels.min()   # background

    # Randomly choose the vernier side (1/2 chance for each side)
    vSide = numpy.int(numpy.round(numpy.random.rand()))  # 1.0 for right bottom bar, 0.0 for left

    # Create the corresponding vernier filter
    [Vx, Vy] = [2 * vbarlength, 2 * vbarwidth + barshift]  # usually [2*10,3]
    vernierFilter = numpy.zeros([Vx, Vy]) + dark
    vernierFilter[0:vbarlength, 0*vSide + 2*(1-vSide)] = white
    vernierFilter[vbarlength:2 * vbarlength, 2*vSide + 0*(1-vSide)] = white

    # Replace the genuine vernier by the chosen one and return the result
    pixels[vernierLocY:vernierLocY+Vx,vernierLocX:vernierLocX+Vy] = vernierFilter
    return pixels, 2*(vSide-0.5)  # 1.0 for right, -1.0 for left


# Choose the optimal segmentation signal location (x,y) in function of the stimulus name
def chooseSegmentationSignal(thisConditionName, segmentSignalSize):

    # Default values
    segmentationTargetLocationX = 25  # distance from target vernier (at center)
    segmentationTargetLocationY = 0
    # Vernier only
    if thisConditionName in ["vernier", "circles 1", "patterns2 1", "stars 1", "pattern stars 1", "hexagons 1",
                             "octagons 1", "irreg1 1", "irreg2 1", "pattern irregular 1"]:
        segmentationTargetLocationX = 30 + segmentSignalSize  # far enough away that we do not care about it = 50
    # Only 1 flanking shape around vernier
    if thisConditionName in ["squares 1", "circles 2", "patterns2 2", "stars 2", "stars 6", "pattern stars 2",
                             "hexagons 2", "hexagons 7", "octagons 2", "octagons 7", "irreg1 2", "irreg2 2",
                             "pattern irregular 2"]:
        segmentationTargetLocationX = 16 + segmentSignalSize  # aiming for outer edge of square
    # Flanking shape around vernier + 1 on each side (RE-DESTRIBUTE STIMULI!)
    if thisConditionName in ["cuboids", "scrambled cuboids", "squares 2",
                             "circles 3", "circles 4", "circles 5", "circles 6", "triangles 2", "triangles 3",
                             "pattern2 3", "pattern2 4", "pattern2 5", "pattern2 6", "pattern2 7", "pattern2 8",
                             "pattern2 9", "pattern2 10",
                             "stars 3", "stars 4", "stars 5", "stars 7", "stars 8", "stars 9",
                             "pattern stars 3", "pattern stars 4", "pattern stars 5", "pattern stars 6",
                             "pattern stars 7", "pattern stars 8",
                             "pattern stars 9", "pattern stars 10", "pattern stars 11", "pattern stars 13",
                             "pattern stars 14",
                             "hexagons 3", "hexagons 4", "hexagons 5", "hexagons 6", "hexagons 8", "hexagons 9",
                             "hexagons 10", "hexagons 11",
                             "octagons 3", "octagons 4", "octagons 5", "octagons 6", "octagons 8", "octagons 9",
                             "octagons 10", "octagons 11",
                             "irreg1 3", "irreg1 4", "irreg1 5", "irreg1 6", "irreg1 7", "irreg1 8", "irreg1 9",
                             "irreg1 10",
                             "irreg2 3", "irreg2 4", "irreg2 5",
                             "pattern irregular 3", "pattern irregular 4", "pattern irregular 5", "pattern irregular 6",
                             "pattern irregular 7",
                             "pattern irregular 8", "pattern irregular 9", "pattern irregular 10",
                             "pattern irregular 11"]:
        segmentationTargetLocationX = 25 + segmentSignalSize  # aiming for outer edge of outer square
    # Flanking shape around vernier + more than 1 on each side
    if thisConditionName in ["squares 3", "squares 4"]:
        segmentationTargetLocationX = 30 + segmentSignalSize
    # Same as previous condition, but vertically shaped
    if thisConditionName == "pattern stars 12":
        segmentationTargetLocationX = 0
        segmentationTargetLocationY = 32 + 8 + segmentSignalSize  # aiming for outer edge of outer square ("2 *" originally) = 60
    # Box at both sides of the vernier
    if thisConditionName == "boxes":
        segmentationTargetLocationX = int(numpy.sqrt(2.0)*10) + 8 + segmentSignalSize  # aiming for outer edge of rectangle = 42
    # Cross at both sides of the vernier
    if thisConditionName == "crosses":
        segmentationTargetLocationX = int(numpy.sqrt(2.0)*10) + segmentSignalSize  # aiming for a bit off center of rectangle (does not make much difference) = 34
    # Box with a cross within at both sides of the vernier
    if thisConditionName == "boxes and crosses":
        segmentationTargetLocationX = int(numpy.sqrt(2.0)*10) + 4 + segmentSignalSize  # aiming so edge of seg signal just covers region next to target = 38
    # One flanking line
    if thisConditionName in ["HalfLineFlanks1", "malania short 1", "malania equal 1", "malania long 1",
                             "malania equal assymR 1", "malania equal assymL 1", "malania short assymR 1", "malania short assymL 1", "malania long assymR 1", "malania long assymL 1"]:
        segmentationTargetLocationX = 8 + segmentSignalSize  # aiming to just touch the flanker = 27
    # Two flanking lines
    if thisConditionName in ["HalfLineFlanks2", "malania short 2", "malania equal 2", "malania long 2", "malania equal assymR 2", "malania equal assymL 2"]:
        segmentationTargetLocationX = 16 + segmentSignalSize - 2  # aiming to just touch the outer flanker = 34
    # More than two flanking lines
    if thisConditionName in ["HalfLineFlanks3", "HalfLineFlanks4", "HalfLineFlanks5", "HalfLineFlanks6",
                             "HalfLineFlanks7", "HalfLineFlanks8", "HalfLineFlanks10",
                             "malania short 4", "malania short 8", "malania short 16",
                             "malania equal 4", "malania equal 8", "malania equal 16",
                             "malania long 4", "malania long 8", "malania long 16"]:
        segmentationTargetLocationX = 24 + segmentSignalSize - 2  # aiming to just touch the outer flanker = 42
    # Harrisson C's
    if thisConditionName == "Harrisson":
        segmentationTargetLocationX = 50

    return segmentationTargetLocationX, segmentationTargetLocationY


# Additional crop definitions go there !
def chooseAddCrop(thisConditionName):

    [addCropX, addCropY] = [0, 0]
    if thisConditionName in ["vernier"]:
        [addCropX, addCropY] = [50, 15]
    if thisConditionName in ["malania short 1", "malania short 2", "malania short 4", "malania short 8", "malania short 16",
                             "malania equal 1", "malania equal 2", "malania equal 4", "malania equal 8", "malania equal 16"]:
        [addCropX, addCropY] = [10, 15]
    if thisConditionName in ["malania long 1", "malania long 2", "malania long 4", "malania long 8", "malania long 16",
                             "malania long assymR 1", "malania long assymL 1", "cuboids"]:
        [addCropX, addCropY] = [10, 10]
    if thisConditionName in ["boxes", "crosses", "boxes and crosses"]:
        [addCropX, addCropY] = [20, 15]
    if thisConditionName in ["scrambled cuboids", "malania short assymR 1", "malania short assymL 1", "malania equal assymR 1", "malania equal assymL 1"]:
        [addCropX, addCropY] = [15, 20]
    if thisConditionName in ["malania equal assymR 2", "malania equal assymL 2"]:
        [addCropX, addCropY] = [25, 20]
    if thisConditionName in ["squares 1", "circles 1", "circles 2", "pattern2 1", "pattern2 2", "stars 1", "stars 2",
                             "stars 6",
                             "pattern stars 1", "pattern stars 2", "hexagons 1", "hexagons 2", "hexagons 7",
                             "octagons 1", "octagons 2",
                             "octagons 7", "irreg1 1", "irreg1 2", "irreg2 1", "irreg2 2", "pattern irregular 1",
                             "pattern irregular 2"]:
        [addCropX, addCropY] = [80, 75] # put back 180, 75
    if thisConditionName in ["circles 3", "stars 3"]:
        [addCropX, addCropY] = [110, 75]
    if thisConditionName in ["circles 4", "hexagons 3", "hexagons 4", "hexagons 5", "hexagons 6", "hexagons 8",
                             "hexagons 9", "hexagons 10",
                             "hexagons 11", "octagons 3", "octagons 4", "octagons 5", "octagons 6", "octagons 8",
                             "octagons 9", "octagons 10",
                             "octagons 11", "stars 4", "stars 7", "irreg1 5", "irreg1 6", "irreg1 7", "irreg1 9"]:
        [addCropX, addCropY] = [100, 75]
    if thisConditionName in ["circles 5", "stars 8", "pattern2 3", "pattern2 4", "irreg2 3"]:
        [addCropX, addCropY] = [90, 75]
    if thisConditionName in ["circles 6", "stars 5", "stars 9", "pattern irregular 3", "pattern irregular 5"]:
        [addCropX, addCropY] = [80, 75]
    if thisConditionName == "irreg2 4":
        [addCropX, addCropY] = [70, 75]
    if thisConditionName == "irreg2 5":
        [addCropX, addCropY] = [55, 75]
    if thisConditionName in ["irreg1 8"]:
        [addCropX, addCropY] = [130, 75]
    if thisConditionName in ["squares 2", "triangles 2"]:
        [addCropX, addCropY] = [160, 75]
    if thisConditionName in ["irreg1 3"]:
        [addCropX, addCropY] = [120, 75]
    if thisConditionName in ["irreg1 4", "irreg1 10"]:
        [addCropX, addCropY] = [110, 75]
    if thisConditionName == "pattern stars 12":
        [addCropX, addCropY] = [180, 50]
    if thisConditionName == "pattern stars 13":
        [addCropX, addCropY] = [130, 50]
    if thisConditionName in ["squares 3", "triangles 3"]:
        [addCropX, addCropY] = [130, 85]
    if thisConditionName == "squares 4":
        [addCropX, addCropY] = [95, 85]
    if thisConditionName in ["pattern2 5", "pattern2 6", "pattern2 7", "pattern2 8", "pattern2 9", "pattern2 10",
                             "pattern stars 5",
                             "pattern stars 6", "pattern stars 7", "pattern stars 8", "pattern stars 9",
                             "pattern stars 14"]:
        [addCropX, addCropY] = [95, 50]
    if thisConditionName in ["pattern irregular 4", "pattern irregular 6", "pattern irregular 7", "pattern irregular 8",
                             "pattern irregular 9",
                             "pattern irregular 10"]:
        [addCropX, addCropY] = [90, 40]
    if thisConditionName in ["pattern stars 10", "pattern stars 11"]:
        [addCropX, addCropY] = [95, 5]
    if thisConditionName in ["pattern irregular 11"]:
        [addCropX, addCropY] = [92, 5]

    return addCropX, addCropY