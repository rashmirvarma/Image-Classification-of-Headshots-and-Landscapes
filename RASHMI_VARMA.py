##Author: Rashmi Varma
##Date Created: October 17, 2017
##Image Classification of Headshots and Landscapes


##Refer to read me file
try:
    import numpy as np
except:
    ImportError
try:
    import os
except:
    ImportError
try:
    from PIL import Image
except:
    ImportError
try:
    import random
except:
    ImportError
try:
    import matplotlib.pyplot as pl
except:
    ImportError
try:
    import pickle
except:
    ImportError
try:
    import operator
except:
    ImportError

##Function that normalizes the RGB counts extracted per pixel
def normalizeRGB(x, pixels):
    x = float(x)/float(pixels)
    return x

##Function that divides RGB into 4 parts each to enhance accuracy
def computeRGB(r,g,b,r1,g1,b1,r2,g2,b2,r3,g3,b3,r4,g4,b4):
    if r>=0 and r<=64:
        r1 = r1 + 1
    elif r>=65 and r<=128:
        r2 = r2 + 1
    elif r>=129 and r<=192:
        r3 = r3 + 1
    else:
        r4 = r4 + 1        
    if g>=0 and g<=64:
        g1 = g1 + 1
    elif g>=65 and g<=128:
        g2 = g2 + 1
    elif g>=129 and g<=192:
        g3 = g3 + 1
    else:
        g4 = g4 + 1       
    if b>=0 and b<=64:
        b1 = b1 + 1
    elif b>=65 and b<=128:
        b2 = b2 + 1
    elif b>=129 and b<=192:
        b3 = b3 + 1
    else:
        b4 = b4 + 1
      
    return (r1,g1,b1,r2,g2,b2,r3,g3,b3,r4,g4,b4)

def computeRGB11(r,g,b,r1,g1,b1,r2,g2,b2,r3,g3,b3,r4,g4,b4):
    if r>=0 and r<=64:
        r1 = r1 + r
    elif r>=65 and r<=128:
        r2 = r2 + r
    elif r>=129 and r<=192:
        r3 = r3 + r
    else:
        r4 = r4 + r        
    if g>=0 and g<=64:
        g1 = g1 + g
    elif g>=65 and g<=128:
        g2 = g2 + g
    elif g>=129 and g<=192:
        g3 = g3 + g
    else:
        g4 = g4 + g       
    if b>=0 and b<=64:
        b1 = b1 + b
    elif b>=65 and b<=128:
        b2 = b2 + b
    elif b>=129 and b<=192:
        b3 = b3 + b
    else:
        b4 = b4 + b
      
    return (r1,g1,b1,r2,g2,b2,r3,g3,b3,r4,g4,b4)

##dumps dictionary into a pickle file
def checkPrep(file_name,table):
    
    fileObject = open(file_name,'wb')
    pickle.dump(table,fileObject)
    fileObject.close()
    fileObject = open(file_name,'r')  
    table = pickle.load(fileObject)
    
##Extracts and prepares table of Flags Dataset
def flagPic(pathFlag):
    countryName = []
    r1 = 0
    r2 = 0
    r3 = 0
    r4 = 0
    g1 = 0
    g2 = 0
    g3 = 0
    g4 = 0
    b1 = 0
    b2 = 0
    b3 = 0
    b4 = 0
    R1 = []
    R2 = []
    R3 = []
    R4 = []
    G1 = []
    G2 = []
    G3 = []
    G4 = []
    B1 = []
    B2 = []
    B3 = []
    B4 = []
    table = {}
    flagList = os.listdir(pathFlag)
    for flags in flagList:
        if not flags.startswith('.'):
            countryName.append(flags)
            fs = Image.open(pathFlag + flags)
            rgbIm = fs.convert('RGB')

            rgbPix = rgbIm.load()
            height, width = fs.size
            arr = np.array(rgbIm)
            for x in range(0, width):
                for y in range(0,height):
                    r, g, b = arr[x,y]
                    r1,g1,b1,r2,g2,b2,r3,g3,b3,r4,g4,b4 = computeRGB(r,g,b,r1,g1,b1,r2,g2,b2,r3,g3,b3,r4,g4,b4)
                    
            pixels = height * width
            
            r1 = normalizeRGB(r1, pixels)
            r1 = round(r1, 3)
            r2 = normalizeRGB(r2, pixels)
            r2 = round(r2, 3)
            r3 = normalizeRGB(r3, pixels)
            r3 = round(r3, 3)
            r4 = normalizeRGB(r4, pixels)
            r4 = round(r4, 3)
            g1 = normalizeRGB(g1, pixels)
            g1 = round(g1, 3)
            g2 = normalizeRGB(g2, pixels)
            g2 = round(g2, 3)
            g3 = normalizeRGB(g3, pixels)
            g3 = round(g3, 3)
            g4 = normalizeRGB(g4, pixels)
            g4 = round(g4, 3)
            b1 = normalizeRGB(b1, pixels)
            b1 = round(b1, 3)
            b2 = normalizeRGB(b2, pixels)
            b2 = round(b2, 3)
            b3 = normalizeRGB(b3, pixels)
            b3 = round(b3, 3)
            b4 = normalizeRGB(b4, pixels)
            b4 = round(b4, 3)

            R1.append(r1)
            R2.append(r2)
            R3.append(r3)
            R4.append(r4)
            G1.append(g1)
            G2.append(g2)
            G3.append(g3)
            G4.append(g4)
            B1.append(b1)
            B2.append(b2)
            B3.append(b3)
            B4.append(b4)           
    table["File"] = countryName
    table["R1"] = R1
    table["R2"] = R2
    table["R3"] = R3
    table["R4"] = R4
    table["G1"] = G1
    table["G2"] = G2
    table["G3"] = G3
    table["G4"] = G4
    table["B1"] = B1
    table["B2"] = B2
    table["B3"] = B3
    table["B4"] = B4
    checkPrep(file2,flagtable)

    return table

##Function that creates table of Landscapes and Headshots
def datasetPrep(file_name,table, fileName, R1, R2, R3, R4, B1, B2, B3, B4, G1, G2, G3, G4, label,pathLandscape, pathHeadshots):
    r1 = 0
    r2 = 0
    r3 = 0
    r4 = 0
    g1 = 0
    g2 = 0
    g3 = 0
    g4 = 0
    b1 = 0
    b2 = 0
    b3 = 0
    b4 = 0
    
    landscapeList = os.listdir(pathLandscape)
    headshotsList = os.listdir(pathHeadshots)

    for lands in landscapeList:
        if not lands.startswith('.'):
            fileName.append(lands)
            ls = Image.open(pathLandscape + lands)
            rgbIm = ls.convert('RGB')
            rgbPix = rgbIm.load()
            height, width = ls.size
            arr = np.array(rgbIm)
            for x in range(0, width):
                for y in range(0,height):
                    r, g, b = arr[x,y]
                    r1,g1,b1,r2,g2,b2,r3,g3,b3,r4,g4,b4 = computeRGB(r,g,b,r1,g1,b1,r2,g2,b2,r3,g3,b3,r4,g4,b4)
                    
            pixels = height * width
            
            r1 = normalizeRGB(r1, pixels)
            r1 = round(r1, 3)
            r2 = normalizeRGB(r2, pixels)
            r2 = round(r2, 3)
            r3 = normalizeRGB(r3, pixels)
            r3 = round(r3, 3)
            r4 = normalizeRGB(r4, pixels)
            r4 = round(r4, 3)
            g1 = normalizeRGB(g1, pixels)
            g1 = round(g1, 3)
            g2 = normalizeRGB(g2, pixels)
            g2 = round(g2, 3)
            g3 = normalizeRGB(g3, pixels)
            g3 = round(g3, 3)
            g4 = normalizeRGB(g4, pixels)
            g4 = round(g4, 3)
            b1 = normalizeRGB(b1, pixels)
            b1 = round(b1, 3)
            b2 = normalizeRGB(b2, pixels)
            b2 = round(b2, 3)
            b3 = normalizeRGB(b3, pixels)
            b3 = round(b3, 3)
            b4 = normalizeRGB(b4, pixels)
            b4 = round(b4, 3)

            R1.append(r1)
            R2.append(r2)
            R3.append(r3)
            R4.append(r4)
            G1.append(g1)
            G2.append(g2)
            G3.append(g3)
            G4.append(g4)
            B1.append(b1)
            B2.append(b2)
            B3.append(b3)
            B4.append(b4)
            label.append("landscape")

    for heads in headshotsList:
        if not heads.startswith('.'):
            fileName.append(heads)
            hs = Image.open(pathHeadshots + heads)
            rgbIm = hs.convert('RGB')
            rgbPix = rgbIm.load()
            height, width = hs.size
            arr = np.array(rgbIm)
            for x in range(0, width):
                for y in range(0,height):
                    r, g, b = arr[x,y]
                    r1,g1,b1,r2,g2,b2,r3,g3,b3,r4,g4,b4 = computeRGB(r,g,b,r1,g1,b1,r2,g2,b2,r3,g3,b3,r4,g4,b4)
                    
            pixels = height * width
            
            r1 = normalizeRGB(r1, pixels)
            r1 = round(r1, 3)
            r2 = normalizeRGB(r2, pixels)
            r2 = round(r2, 3)
            r3 = normalizeRGB(r3, pixels)
            r3 = round(r3, 3)
            r4 = normalizeRGB(r4, pixels)
            r4 = round(r4, 3)
            g1 = normalizeRGB(g1, pixels)
            g1 = round(g1, 3)
            g2 = normalizeRGB(g2, pixels)
            g2 = round(g2, 3)
            g3 = normalizeRGB(g3, pixels)
            g3 = round(g3, 3)
            g4 = normalizeRGB(g4, pixels)
            g4 = round(g4, 3)
            b1 = normalizeRGB(b1, pixels)
            b1 = round(b1, 3)
            b2 = normalizeRGB(b2, pixels)
            b2 = round(b2, 3)
            b3 = normalizeRGB(b3, pixels)
            b3 = round(b3, 3)
            b4 = normalizeRGB(b4, pixels)
            b4 = round(b4, 3)

            R1.append(r1)
            R2.append(r2)
            R3.append(r3)
            R4.append(r4)
            G1.append(g1)
            G2.append(g2)
            G3.append(g3)
            G4.append(g4)
            B1.append(b1)
            B2.append(b2)
            B3.append(b3)
            B4.append(b4)
            label.append("headshots")

    table["File"] = fileName
    table["R1"] = R1
    table["R2"] = R2
    table["R3"] = R3
    table["R4"] = R4
    table["G1"] = G1
    table["G2"] = G2
    table["G3"] = G3
    table["G4"] = G4
    table["B1"] = B1
    table["B2"] = B2
    table["B3"] = B3
    table["B4"] = B4

    table["Label"] = label
    checkPrep(file_name,table)
    return table

##Calculating Euclidean distance between RGB of two images during cross validation
def compareValues1(trainR1, trainR2, trainR3, trainR4, trainB1, trainB2, trainB3, trainB4, trainG1, trainG2, trainG3, trainG4, valR1, valR2, valR3, valR4, valB1, valB2, valB3, valB4, valG1, valG2, valG3, valG4):
    distance2 = []
    distance = []
    for j in range(0,len(valR1)):
        distance1 = []

        for i in range(0,len(trainR1)):

            diff = (((trainR1[i] - valR1[j])**2) + ((trainR2[i] - valR2[j])**2) + ((trainR3[i] - valR3[j])**2) + ((trainR4[i] - valR4[j])**2)+ ((trainG1[i] - valG1[j])**2) + ((trainG2[i] - valG2[j])**2) + ((trainG3[i] - valG3[j])**2) + ((trainG4[i] - valG4[j])**2) + ((trainB1[i] - valB1[j])**2)+ ((trainB2[i] - valB2[j])**2) + ((trainB3[i] - valB3[j])**2) + ((trainB4[i] - valB4[j])**2))
            dist = diff**(1/2.0)
            dist = round(dist, 3)
            distance1.append(dist)

        distance.append(distance1)


    return distance
##Calculating Euclidean distance between RGB of two images during Knn

def compareValues(tempRed,tempBlue,tempGreen, red1,green1, blue1, index):
    distance = []
    totalLength = len(tempRed)
    for x in range(0,totalLength):
        diff = (((tempRed[x] - red1)**2) + ((tempBlue[x] - blue1)**2) + ((tempGreen[x] - green1)**2))
        dist = diff**(1/2.0)
        dist = round(dist, 3)
        distance.append(dist)
    return distance

##Calculating Euclidean distance between RGB of two images during cross validation
       
def computeDistance(table, red1, red2, red3, red4, blue1, blue2, blue3, blue4, green1, green2, green3, green4, index):
    distance = []
    distance1 = []
    distance2 = []
    distance3 = []
    distance4 = []

    tempRed = []
    tempBlue = []
    tempGreen = []
    tempRed1 = table["R1"]
    tempBlue1 = table["G1"]
    tempGreen1 = table["B1"]
    tempRed2 = table["R2"]
    tempBlue2 = table["G2"]
    tempGreen2 = table["B2"]
    tempRed3 = table["R3"]
    tempBlue3 = table["G3"]
    tempGreen3 = table["B3"]
    tempRed4 = table["R4"]
    tempBlue4 = table["G4"]
    tempGreen4 = table["B4"]

    distance1 = compareValues(tempRed1,tempBlue1,tempGreen1, red1,green1, blue1, index)
    distance2 = compareValues(tempRed2,tempBlue2,tempGreen2, red2,green2, blue2, index)
    distance3 = compareValues(tempRed3,tempBlue3,tempGreen3, red3,green3, blue3, index)
    distance4 = compareValues(tempRed4,tempBlue4,tempGreen4, red4,green4, blue4, index)

    for k in range(0,len(distance1)):
        dist = distance1[k] + distance2[k] + distance3[k] + distance4[k]
        dist = round(dist, 3)
        distance.append(dist)
    return (distance)

##Agent accepts k and image from user to find knn
def agent(table, distance, index):
    minimumDistance = []
    distanceAsPerK = []
    freqLandscape = 0
    freqHeadshots = 0
    equal = 0
    k = int(raw_input("\nPlease enter the value of k for computing KNN:"))

    minimumDistance = sorted(distance)

    distanceAsPerK = minimumDistance[:k]

    for i in range(0, len(distance)):
        for j in range(0, len(distanceAsPerK)):
            if distance[i] == distanceAsPerK[j]:
                index.append(i)
    for ind in index:
        temp = table["Label"][ind]
        if temp == "landscape":
            freqLandscape = freqLandscape + 1
        if temp == "headshots":
            freqHeadshots = freqHeadshots + 1

    if freqLandscape > freqHeadshots:
        return "Landscape"
    elif freqHeadshots > freqLandscape:
        return "Headshots"
    else:
        return "None"
    
##Extracting features of new image. Extension of agent
def newImageCleaner(table, red1, red2, red3, red4, blue1, blue2, blue3, blue4, green1, green2, green3, green4):

    newImage = raw_input("Please enter path of Image you wish to classify:")
    nI = Image.open(newImage)
    rgbIm = nI.convert('RGB')
    height, width = nI.size
    arr = np.array(rgbIm)
    for x in range(0, width):
        for y in range(0,height):
            r, g, b = arr[x,y]
            pixels = height * width

            red1,green1,blue1,red2,green2,blue2,red3,green3,blue3,r4,green4,blue4 = computeRGB(r,g,b,red1,green1,blue1,red2,green2,blue2,red3,green3,blue3,red4,green4,blue4,pixels)
                   
    red1,red2,red3,red4,green1,green2,green3,green4,blue1,blue2,blue3,blue4 = normalize(red1,red2,red3,red4,green1,green2,green3,green4,blue1,blue2,blue3,blue4)
    return(red1,red2,red3,red4,green1,green2,green3,green4,blue1,blue2,blue3,blue4)

##Environment building our lookup table
def environment(table):

    red1 = 0
    red2 = 0
    red3 = 0
    red4 = 0
    green1 = 0
    green2 = 0
    green3 = 0
    green4 = 0
    blue1 = 0
    blue2 = 0
    blue3 = 0
    blue4 = 0
    index = []
    distance = []
    
    newImage = raw_input("Please enter path of Image you wish to classify:")
    nI = Image.open(newImage)
    rgbIm = nI.convert('RGB')
    height, width = nI.size
    arr = np.array(rgbIm)
    for x in range(0, width):
        for y in range(0,height):
            r, g, b = arr[x,y]
            red1,green1,blue1,red2,green2,blue2,red3,green3,blue3,r4,green4,blue4 = computeRGB(r,g,b,red1,green1,blue1,red2,green2,blue2,red3,green3,blue3,red4,green4,blue4)
                   
    pixels = height * width

    red1 = normalizeRGB(red1, pixels)
    red1 = round(red1, 3)
    red2 = normalizeRGB(red2, pixels)
    red2 = round(red2, 3)
    red3 = normalizeRGB(red3, pixels)
    red3 = round(red3, 3)
    red4 = normalizeRGB(red4, pixels)
    red4 = round(red4, 3)
    green1 = normalizeRGB(green1, pixels)
    green1 = round(green1, 3)
    green2 = normalizeRGB(green2, pixels)
    green2 = round(green2, 3)
    green3 = normalizeRGB(green3, pixels)
    green3 = round(green3, 3)
    green4 = normalizeRGB(green4, pixels)
    green4 = round(green4, 3)
    blue1 = normalizeRGB(blue1, pixels)
    blue1 = round(blue1, 3)
    blue2 = normalizeRGB(blue2, pixels)
    blue2 = round(blue2, 3)
    blue3 = normalizeRGB(blue3, pixels)
    blue3 = round(blue3, 3)
    blue4 = normalizeRGB(blue4, pixels)
    blue4 = round(blue4, 3)

    distance = computeDistance(table, red1, red2, red3, red4, blue1, blue2, blue3, blue4, green1, green2, green3, green4, index)
    try:
        frequency = agent(table, distance, index)
        if frequency == "None":
            print("\nThe algorithm could not decide what classification the image is.There was a tie.Please try again with a different value of k")
        if frequency == "Landscape":
            print("\nThe new image entered is a Landscape")
        elif frequency == "Headshots":
            print("\nThe image entered is a Headshot")

    except:
        print("Recheck input to agent")
        
##Function to automatically split into 3 folds
def splittingFunction(listPic, lenList):
    number = lenList / 3
    splitList = np.array_split(listPic,3)
    return splitList

##Function to normalize pixels
def normalize(r1,r2,r3,r4,g1,g2,g3,g4,b1,b2,b3,b4,pixels):
    r1 = normalizeRGB(r1, pixels)
    r1 = round(r1, 3)
    r2 = normalizeRGB(r2, pixels)
    r2 = round(r2, 3)
    r3 = normalizeRGB(r3, pixels)
    r3 = round(r3, 3)
    r4 = normalizeRGB(r4, pixels)
    r4 = round(r4, 3)
    g1 = normalizeRGB(g1, pixels)
    g1 = round(g1, 3)
    g2 = normalizeRGB(g2, pixels)
    g2 = round(g2, 3)
    g3 = normalizeRGB(g3, pixels)
    g3 = round(g3, 3)
    g4 = normalizeRGB(g4, pixels)
    g4 = round(g4, 3)
    b1 = normalizeRGB(b1, pixels)
    b1 = round(b1, 3)
    b2 = normalizeRGB(b2, pixels)
    b2 = round(b2, 3)
    b3 = normalizeRGB(b3, pixels)
    b3 = round(b3, 3)
    b4 = normalizeRGB(b4, pixels)
    b4 = round(b4, 3)

    return(r1,r2,r3,r4,g1,g2,g3,g4,b1,b2,b3,b4)

## Function to extract rows from look up table for clustering   
def extract(table,lists,fileName,R1, R2, R3, R4, B1, B2, B3, B4, G1, G2, G3, G4, label):
    r1 = 0
    r2 = 0
    r3 = 0
    r4 = 0
    g1 = 0
    g2 = 0
    g3 = 0
    g4 = 0
    b1 = 0
    b2 = 0
    b3 = 0
    b4 = 0

    label = []
    fileName = []
    for files in lists:
        if files.startswith("l"):
            fileName.append(files)
            ls = Image.open(pathLandscape + files)
            totalLandLen = len(lists)
            tempLabels = files[:-7]
            rgbIm = ls.convert('RGB')
            rgbPix = rgbIm.load()
            height, width = ls.size
            arr = np.array(rgbIm)
            for x in range(0,width):
                for y in range(0, height):
                    r,g,b = arr[x,y]
                    r1,g1,b1,r2,g2,b2,r3,g3,b3,r4,g4,b4 = computeRGB(r,g,b,r1,g1,b1,r2,g2,b2,r3,g3,b3,r4,g4,b4)
            pixels = height * width

            r1,r2,r3,r4,g1,g2,g3,g4,b1,b2,b3,b4 = normalize(r1,r2,r3,r4,g1,g2,g3,g4,b1,b2,b3,b4,pixels)

            R1.append(r1)
            R2.append(r2)
            R3.append(r3)
            R4.append(r4)
            G1.append(g1)
            G2.append(g2)
            G3.append(g3)
            G4.append(g4)
            B1.append(b1)
            B2.append(b2)
            B3.append(b3)
            B4.append(b4)
            label.append(files[:-7])

        if files.startswith("h"):
            fileName.append(files)
            hs = Image.open(pathHeadshots + files)
            totalLandLen = len(lists)
            tempLabels = files[:-7]
            rgbIm = hs.convert('RGB')
            rgbPix = rgbIm.load()
            height, width = hs.size
            arr = np.array(rgbIm)
            for x in range(0,width):
                for y in range(0, height):
                    r,g,b = arr[x,y]
                    r1,g1,b1,r2,g2,b2,r3,g3,b3,r4,g4,b4 = computeRGB(r,g,b,r1,g1,b1,r2,g2,b2,r3,g3,b3,r4,g4,b4)
            pixels = height * width
            
            r1,r2,r3,r4,g1,g2,g3,g4,b1,b2,b3,b4 = normalize(r1,r2,r3,r4,g1,g2,g3,g4,b1,b2,b3,b4,pixels)


            R1.append(r1)
            R2.append(r2)
            R3.append(r3)
            R4.append(r4)
            G1.append(g1)
            G2.append(g2)
            G3.append(g3)
            G4.append(g4)
            B1.append(b1)
            B2.append(b2)
            B3.append(b3)
            B4.append(b4)
            label.append(files[:-7])


    table["File"] = fileName
    table["R1"] = R1
    table["R2"] = R2
    table["R3"] = R3
    table["R4"] = R4
    table["G1"] = G1
    table["G2"] = G2
    table["G3"] = G3
    table["G4"] = G4
    table["B1"] = B1
    table["B2"] = B2
    table["B3"] = B3
    table["B4"] = B4

    table["Label"] = label

    return table
            
def extract1(table,table1,lists,fileName,R1, R2, R3, R4, B1, B2, B3, B4, G1, G2, G3, G4, label):
    r1 = 0
    r2 = 0
    r3 = 0
    r4 = 0
    g1 = 0
    g2 = 0
    g3 = 0
    g4 = 0
    b1 = 0
    b2 = 0
    b3 = 0
    b4 = 0
    ind = 0

    label = []
    fileName = []
    index = []
    for files in lists:
        if files.startswith("l"):
            fileName.append(files)
##            ls = Image.open(pathLandscape + files)
            totalLandLen = len(lists)
            tempLabels = files[:-7]
##            rgbIm = ls.convert('RGB')
##            rgbPix = rgbIm.load()
            
            for i in range(0,len(table["File"])):
                if files == table["File"][i]:
                    ind = i
            r1 = table["R1"][ind]
            r2 = table["R2"][ind]
            r3 = table["R3"][ind]
            r4 = table["R4"][ind]
            g1 = table["G1"][ind]
            g2 = table["G2"][ind]
            g3 = table["G3"][ind]
            g4 = table["G4"][ind]
            b1 = table["B1"][ind]
            b2 = table["B2"][ind]
            b3 = table["B3"][ind]
            b4 = table["B4"][ind]

            R1.append(r1)
            R2.append(r2)
            R3.append(r3)
            R4.append(r4)
            G1.append(g1)
            G2.append(g2)
            G3.append(g3)
            G4.append(g4)
            B1.append(b1)
            B2.append(b2)
            B3.append(b3)
            B4.append(b4)
            label.append(files[:-7])

        if files.startswith("h"):
            fileName.append(files)
##            hs = Image.open(pathHeadshots + files)
            totalHeadLen = len(lists)
            tempLabels = files[:-7]
##            rgbIm = hs.convert('RGB')
##            rgbPix = rgbIm.load()
            for j in range(0,len(table["File"])):
                if files == table["File"][j]:
                    ind = j
            r1 = table["R1"][ind]
            r2 = table["R2"][ind]
            r3 = table["R3"][ind]
            r4 = table["R4"][ind]
            g1 = table["G1"][ind]
            g2 = table["G2"][ind]
            g3 = table["G3"][ind]
            g4 = table["G4"][ind]
            b1 = table["B1"][ind]
            b2 = table["B2"][ind]
            b3 = table["B3"][ind]
            b4 = table["B4"][ind]

            R1.append(r1)
            R2.append(r2)
            R3.append(r3)
            R4.append(r4)
            G1.append(g1)
            G2.append(g2)
            G3.append(g3)
            G4.append(g4)
            B1.append(b1)
            B2.append(b2)
            B3.append(b3)
            B4.append(b4)
            label.append(files[:-7])

    table1["File"] = fileName
    table1["R1"] = R1
    table1["R2"] = R2
    table1["R3"] = R3
    table1["R4"] = R4
    table1["G1"] = G1
    table1["G2"] = G2
    table1["G3"] = G3
    table1["G4"] = G4
    table1["B1"] = B1
    table1["B2"] = B2
    table1["B3"] = B3
    table1["B4"] = B4

    table1["Label"] = label

    return table1
                                         
##splitting the datasets into 3 folds
def partition(table,validation,training, fileName, R1, R2, R3, R4, B1, B2, B3, B4, G1, G2, G3, G4, Red1, Red2, Red3, Red4, Blue1, Blue2, Blue3, Blue4, Green1, Green2, Green3, Green4, label):
    r1 = 0
    r2 = 0
    r3 = 0
    r4 = 0
    g1 = 0
    g2 = 0
    g3 = 0
    g4 = 0
    b1 = 0
    b2 = 0
    b3 = 0
    b4 = 0
    temp = []
    totalLandscape = []
    totalHeadshots = []
    landscapeList = []
    headshotsList = []
    landIndex = []
    headIndex = []
    l1 = len(table["Label"])
    for i in range(0, len(table["Label"])):
        if table["Label"][i] == "landscape":
            landIndex.append(i)
        if table["Label"][i] == "headshots":
            headIndex.append(i)
    for ind in landIndex:
        totalLandscape.append(table["File"][ind])
    for ind in headIndex:
        totalHeadshots.append(table["File"][ind])   
    landscapeList = totalLandscape
    headshotsList = totalHeadshots

    np.random.shuffle(landscapeList)
    np.random.shuffle(headshotsList)
    
    lenLand = len(totalLandscape)
    lenHead = len(totalHeadshots)

    splitLand = splittingFunction(landscapeList, lenLand)
    landFold1 = splitLand[0]
    landFold2 = splitLand[1]
    landFold3 = splitLand[2]

    splitHead = splittingFunction(headshotsList, lenHead)
    headFold1 = splitHead[0]
    headFold2 = splitHead[1]
    headFold3 = splitHead[2]

    train1 = np.append(landFold1, headFold1)
    np.random.shuffle(train1)

    train2 = np.append(landFold2, headFold2)
    np.random.shuffle(train2)

    validate = np.append(landFold3, headFold3)
    np.random.shuffle(validate)

    train = np.append(train1, train2)

    training = extract1(table,training, train,fileName,R1, R2, R3, R4, B1, B2, B3, B4, G1, G2, G3, G4, label)
    validation = extract1(table,validation, validate,fileName,Red1, Red2, Red3, Red4, Blue1, Blue2, Blue3, Blue4, Green1, Green2, Green3, Green4, label)

    return (training, validation)

##Extracting RGB features from table for calculating distance for clustering
def extractRGB(dataset, R1, R2, R3, R4, B1, B2, B3, B4, G1, G2, G3, G4):
    R1 = dataset["R1"]
    R2 = dataset["R2"]
    R3 = dataset["R3"]
    R4 = dataset["R4"]
    G1 = dataset["G1"]
    G2 = dataset["G2"]
    G3 = dataset["G3"]
    G4 = dataset["G4"]
    B1 = dataset["B1"]
    B2 = dataset["B2"]
    B3 = dataset["B3"]
    B4 = dataset["B4"]
    G1 = dataset["G1"]
    G2 = dataset["G2"]
    G3 = dataset["G3"]
    G4 = dataset["G4"]
    return (R1, R2, R3, R4, B1, B2, B3, B4, G1, G2, G3, G4)

##initialisation function which calls functions to compare distances
def computeDistances(train, validation):

    distance = []
    
    trainR1 = []
    trainR2 = []
    trainR3 = []
    trainR4 = []
    trainG1 = []
    trainG2 = []
    trainG3 = []
    trainG4 = []
    trainB1 = []
    trainB2 = []
    trainB3 = []
    trainB4 = []
    valR1 = []
    valR2 = []
    valR3 = []
    valR4 = []
    valB1 = []
    valB2 = []
    valB3 = []
    valB4 = []
    valG1 = []
    valG2 = []
    valG3 = []
    valG4 = []

    trainR1, trainR2, trainR3, trainR4, trainB1, trainB2, trainB3, trainB4, trainG1, trainG2, trainG3, trainG4 = extractRGB(train, trainR1, trainR2, trainR3, trainR4, trainB1, trainB2, trainB3, trainB4, trainG1, trainG2, trainG3, trainG4)
    valR1, valR2, valR3, valR4, valB1, valB2, valB3, valB4, valG1, valG2, valG3, valG4 = extractRGB(validation,valR1, valR2, valR3, valR4, valB1, valB2, valB3, valB4, valG1, valG2, valG3, valG4)
    
    distance = compareValues1(trainR1, trainR2, trainR3, trainR4, trainB1, trainB2, trainB3, trainB4, trainG1, trainG2, trainG3, trainG4, valR1, valR2, valR3, valR4, valB1, valB2, valB3, valB4, valG1, valG2, valG3, valG4)
    return distance

##Function that calculates accuracy of knn
def computeAccuracy(train, validation, distance,k):

    temp = []
    minimumDistance = []
    distanceAsPerK = []
    Label_train = []
    Label_val = []
    index = []
    freqLandscape = 0
    freqHeadshots = 0
    equal = 0
    newLabel = []
    oldLabel = []
    correct = 0
    incorrect = 0
    temp1 = []
    lengthDistance = distance[0]

    for i in range(0, len(distance)):
        temp = distance[i]

        minimumDistance = sorted(temp)
        distanceAsPerK = minimumDistance[:k]
        for m in range(0, len(temp)):
            for j in range(0, len(distanceAsPerK)):
                if temp[m] == distanceAsPerK[j]:

                    index.append(m)

        for ind in index:
            temp1 = train["Label"][ind]
            if temp1 == "landscape":
                freqLandscape = freqLandscape + 1
            if temp1 == "headshots":
                freqHeadshots = freqHeadshots + 1
        if freqLandscape > freqHeadshots:
            newLabel.append("landscape")
        elif freqHeadshots > freqLandscape:
            newLabel.append("headshot")
        else:
            newLabel.append("None")
    oldLabel = validation["Label"]
    
    for x in range(0, len(newLabel)):
        if oldLabel[x] == newLabel[x]:
            correct = correct + 1
        else:
            incorrect = incorrect+1

    total = correct+incorrect
    accuracy = (float(correct)/float(total))*100
    accuracy = round(accuracy, 2)
    return accuracy

##Main cross val function
def crossVal(table,validation, train, fileName, R1, R2, R3, R4, B1, B2, B3, B4, G1, G2, G3, G4, Red1, Red2, Red3, Red4, Blue1, Blue2, Blue3, Blue4, Green1, Green2, Green3, Green4, label,avgAcc, distance,k):

    train, validation = partition(table,validation, train, fileName, R1, R2, R3, R4, B1, B2, B3, B4, G1, G2, G3, G4, Red1, Red2, Red3, Red4, Blue1, Blue2, Blue3, Blue4, Green1, Green2, Green3, Green4, label)
    
    ac= []
    a = 0
    distance = computeDistances(train, validation)
##    accuracy =  computeAccuracy(train, validation, distance,k)
    for i in range(0,3):
        accuracy =  computeAccuracy(train, validation, distance,k)
        a = a + accuracy
    avg = a/3

##    x = max(acc)
##    for t in range(1,10):
##        if x == acc[t]:
##            check = t
##    pl.plot(acc, 'go')
##    pl.xlabel("Accuracy")
##    pl.plot(indk, 'ro')
##    pl.ylabel("Number of neighbours")
##    pl.title("Accuracy wrt k for cross validation")
##    pl.show()
       
    return avg

##Function to make rows for flag clustering
def sepFlag(table):
    fs = []
    fsName = []
    sizeTable = len(table["R1"])
    for i in range(0,sizeTable):
        RGB = []
        r1 = table["R1"][i]
        r2 = table["R2"][i]
        r3 = table["R3"][i]
        r4 = table["R4"][i]
        g1 = table["G1"][i]
        g2 = table["G2"][i]
        g3 = table["G3"][i]
        g4 = table["G4"][i]
        b1 = table["B1"][i]
        b2 = table["B2"][i]
        b3 = table["B3"][i]
        b4 = table["B4"][i]
        ffile = table["File"][i]
        RGB.append(r1)
        RGB.append(r2)
        RGB.append(r3)
        RGB.append(r4)
        RGB.append(g1)
        RGB.append(g2)
        RGB.append(g3)
        RGB.append(g4)
        RGB.append(b1)
        RGB.append(b2)
        RGB.append(b3)
        RGB.append(b4)
        fs.append(RGB)
        fsName.append(ffile)
    return fs,fsName
    
##Function to make rows for flag clustering
def separation(table):
    
    lsTuple = []
    hsTuple = []
    labelLsTuple = []
    labelHsTuple = []
    lsfile = []
    hsfile = []
    sizeTable = len(table["R1"])

##    lab = table["Label"][0]

    for i in range(0,sizeTable):
        RGB = []
        RGB1 = []
        
        if table["Label"][i] == "landscape":
            
            r1 = table["R1"][i]
            r2 = table["R2"][i]
            r3 = table["R3"][i]
            r4 = table["R4"][i]
            g1 = table["G1"][i]
            g2 = table["G2"][i]
            g3 = table["G3"][i]
            g4 = table["G4"][i]
            b1 = table["B1"][i]
            b2 = table["B2"][i]
            b3 = table["B3"][i]
            b4 = table["B4"][i]
            lab = table["Label"][i]
            lfile = table["File"][i]
            RGB.append(r1)
            RGB.append(r2)
            RGB.append(r3)
            RGB.append(r4)
            RGB.append(g1)
            RGB.append(g2)
            RGB.append(g3)
            RGB.append(g4)
            RGB.append(b1)
            RGB.append(b2)
            RGB.append(b3)
            RGB.append(b4)
            labelLsTuple.append(lab)
            lsTuple.append(RGB)
            lsfile.append(lfile)
        if table["Label"][i] == "headshots":
            r1 = table["R1"][i]

            r2 = table["R2"][i]
            r3 = table["R3"][i]
            r4 = table["R4"][i]
            g1 = table["G1"][i]
            g2 = table["G2"][i]
            g3 = table["G3"][i]
            g4 = table["G4"][i]
            b1 = table["B1"][i]
            b2 = table["B2"][i]
            b3 = table["B3"][i]
            b4 = table["B4"][i]
            lab = table["Label"][i]
            hfile = table["File"][i]
            RGB1.append(r1)
            RGB1.append(r2)
            RGB1.append(r3)
            RGB1.append(r4)
            RGB1.append(g1)
            RGB1.append(g2)
            RGB1.append(g3)
            RGB1.append(g4)
            RGB1.append(b1)
            RGB1.append(b2)
            RGB1.append(b3)
            RGB1.append(b4)
            hsfile.append(hfile)
            hsTuple.append(RGB1)
            labelHsTuple.append(lab)


    return(lsTuple, hsTuple, labelLsTuple, labelHsTuple,lsfile, hsfile)

##Computing distances between centres
def computedClusterDis(ps, x):

    temp = []
    dis = 0
    for i in range(0,len(ps)):
        d = ((ps[i] - x[i])**2)
        temp.append(d)
    for j in range(0,len(temp)):
        dis = dis + temp[j]
    distance = dis**(1/2.0)
    distance = round(distance, 2)
    return distance
           
def distanceCalc(hsPoint,lsPoint,ls,hs):
    distance1 = []
    distance2 = []

    temp = []

    c1Center = []
    c2Center = []

    ts = ls + hs
    for i in range(0, len(ts)):
        k = computedClusterDis(ts[hsPoint], ts[i])
        distance1.append(k)
    for j in range(0, len(ts)):
        k = computedClusterDis(ts[lsPoint], ts[i])
        distance2.append(k)       

    for m in range(0, len(distance1)):
        if distance1[m] < distance2[m]:
            c1Center.append(m)
        else:
            c2Center.append(m)

    return (c1Center, c2Center)

##Kmeans function
def kmeansCluster(lsPoint,hsPoint,index_ls, index_hs,index_ts, ls, hs,ts,cen1,cen2,flag):
    c1 = []
    c2 = []
    c1,c2 = distanceCalc(hsPoint,lsPoint,ls,hs)
    return (c1, c2)

def doK(lsPoint,hsPoint,ls, hs,ts, index_ls, index_hs, index_ts, cen1, cen2,kflag):
    c1 = []
    c2 = []
    ls_dist = []
    hs_dist = []
    ls_dist, hs_dist = kmeansCluster(lsPoint,hsPoint,index_ls, index_hs,index_ts, ls, hs,ts,cen1,cen2,kflag)
    for i in range(0,len(ls_dist)):
        for j in range(0, len(ts)):
            if ls_dist[i] == ts[j]:
               c1.append[j]
    for i in range(0,len(hs_dist)):
        for j in range(0, len(ts)):
            if hs_dist[i] == ts[j]:
                c2.append[j]


    return (kflag,ls_dist,hs_dist)

            
def kmeans(table):

    ls = []
    hs = []
    ts = []
    total = []
    labelLs = []
    labelHs = []
    index_ls = []
    index_hs = []
    index_ts = []
    labelTs = []
    cen1 = []
    cen2 = []
    hsPoint = 0
    accuracy = 0
    lsPoint = 0
    c1 = []
    c2 = []
    lsfile = []
    hsfile = []
    ls,hs,labelLs,labelHs,lsfile, hsfile = separation(table)

    labelTs = labelLs + labelHs
    kflag = 0

    ts = ls + hs
    for i in range(0,len(labelTs)):
        if labelTs[i] == "landscape":
            index_ls.append(i)
        else:
            index_hs.append(i)
    while kflag == 0:
        lsPoint = random.choice(index_ls)
        hsPoint = random.choice(index_hs)
        kflag,index_ls,index_hs = doK(lsPoint,hsPoint,ls, hs,ts, index_ls, index_hs, index_ts, cen1, cen2,kflag)
    
        cen2.append(hsPoint)
        cen1.append(lsPoint)
        print(cen2)

        accuracy = findAccuracyKmeans(index_ls,index_hs,ts,labelTs)

        if accuracy > 50:
            kflag = 1
            pl.plot(index_ls, 'go')
            pl.plot(index_hs, 'ro')
            pl.title("KNN")
            pl.xlabel("Images") 
            pl.ylabel("Index")
            pl.show()
            c1 = cluster(table, index_ls)
            c2 = cluster(table, index_hs)
            return accuracy, c1, c2
        
##Substitutes indexes with appropriate file names in the cluster       
def cluster(table, centers):
    lists = []
    for c in centers:
        fileName = table["File"][c]
        lists.append(fileName)
    return lists

##Function that calculates the accuracy of k-means
def findAccuracyKmeans(index_ls,index_hs,ts,labelTs):
    correctC1 = 0
    correctC2 = 0
    for i in index_ls:
        if labelTs[i]=="landscape":
            correctC1 = correctC1 + 1

    for j in index_hs:
        if labelTs[j]=="headshots":
            correctC2 = correctC2 + 1
    accuracy = (float(correctC1 + correctC2))/len(labelTs)
    accuracy = round(accuracy, 2)
    accuracy = accuracy * 100

    return accuracy

##Function that calculates Euclidean distance between 2 lists
def disMat(x1, x2):
    d = 0
    for i in range(0, len(x1)):
        d1 = (x2[i]-x1[i])**2
        d = d + d1
    dist = d**(1/2.0)
    dist = round(dist, 2)
    return dist

##def findMin(Matrix, minArray, Index,lenTs):
##    points = []
##
##    minArray.append(mini)
##    for k in range(0,lenTs):
##        for j in range(0,lenTs):
##            mini = min(i for i in Matrix[k][j] if i > 0)
##
##            if mini == Matrix[k][j]:
##                points.append(k)
##                points.append(j)
##                x = k;
##                y = j;
##    return minArray, points
            
    
## Cluster function used from rflynn's Github
##Credits: https://github.com/rflynn/python-examples/blob/master/src/stats/cluster/agglomerative.py

class Cluster:
	def __init__(self):
		pass
	def __repr__(self):
		return '(%s,%s)' % (self.left, self.right)
	def add(self, clusters, grid, lefti, righti):
		self.left = clusters[lefti]
		self.right = clusters[righti]
		for r in grid:
			r[lefti] = min(r[lefti], r.pop(righti))
		grid[lefti] = map(min, zip(grid[lefti], grid.pop(righti)))
		clusters.pop(righti)
		return (clusters, grid)

##GitHub function tweaked to calculate single linkage agglomerative clustering
def agglomerate(labels, grid):
	clusters = labels
	cc = []
	while len(clusters) > 1:
		distances = [(1, 0, grid[1][0])]
		for i,row in enumerate(grid[2:]):
			distances += [(i+2, j, c) for j,c in enumerate(row[:i+2])]
		j,i,_ = min(distances, key=lambda x:x[2])
		# merge i<-j
		c = Cluster()
		clusters, grid = c.add(clusters, grid, i, j)
		clusters[i] = c
	cc = clusters.pop()
	return cc
    
##Main hierarchical clustering function. We first compute distance matrix    
def hier(table):
    ls = []
    hs = []
    ts = []
    total = []
    labelLs = []
    labelHs = []
    index_ls = []
    index_hs = []
    index_ts = []
    labelTs = []
    cen1 = []
    cen2 = []
    points = []
    hsPoint = 0
    accuracy = 0
    lsPoint = 0
    c1 = []
    c = []
    minArray = []
    minIndex = []
    lsfile = []
    hsfile = []
    files = []
    ls,hs,labelLs,labelHs,lsfile, hsfile = separation(table)
    files = lsfile+hsfile
    ts = ls + hs
    lenTs = len(ts)
    Matrix = [[0 for x in range(lenTs)] for y in range(lenTs)]
    Index = [[0 for x in range(lenTs)] for y in range(lenTs)] 

    labelTs = labelLs + labelHs
    for l in range(0, lenTs):
        for h in range(0, lenTs):
            Matrix[l][h] = disMat(ts[l], ts[h])
##    minArray, points = findMin(Matrix, minArray, Index,lenTs)           
    c = agglomerate(files, Matrix)
    return c

##Calculate table for flags
def fp(ile2,flagtable,pathFlags):
    flagtable = flagPic(pathFlags)
    return flagtable

##Calculate RGB and distance matrix for flag table. Then cluster using hierarchical
def flagProc(file2,flagtable):
    fs = []
    flagLabel = []
    flagNames = []

    fs, flagNames = sepFlag(flagtable)
    lenFs = len(fs)
    flagMatrix = [[0 for x in range(lenFs)] for y in range(lenFs)]
    for l in range(0, lenFs):
        for h in range(0, lenFs):
            flagMatrix[l][h] = disMat(fs[l], fs[h])
    c = agglomerate(flagNames, flagMatrix)
    return c
  
def Threefold(table, validation, train, fileName1, R11, R21, R31, R41, B11, B21, B31, B41, G11, G21, G31, G41, Red1, Red2, Red3, Red4, Blue1, Blue2, Blue3, Blue4, Green1, Green2, Green3, Green4, label1,avgAcc, distance,k):
    validation = {}
    train = {}
    fileName1 = []
    R11 = []
    R21 = []
    R31 = []
    R41 = []
    G11 = []
    G21 = []
    G31 = []
    G41 = []
    B11 = []
    B21 = []
    B31 = []
    B41 = []
    Red1 = []
    Red2 = []
    Red3 = []
    Red4 = []
    Green1 = []
    Green2 = []
    Green3 = []
    Green4 = []
    Blue1 = []
    Blue2 = []
    Blue3 = []
    Blue4 = []
    label1 = []
    distance = []
    accuracy1 = crossVal(table, validation, train, fileName1, R11, R21, R31, R41, B11, B21, B31, B41, G11, G21, G31, G41, Red1, Red2, Red3, Red4, Blue1, Blue2, Blue3, Blue4, Green1, Green2, Green3, Green4, label1,avgAcc, distance,k)
    validation = {}
    train = {}
    fileName1 = []
    R11 = []
    R21 = []
    R31 = []
    R41 = []
    G11 = []
    G21 = []
    G31 = []
    G41 = []
    B11 = []
    B21 = []
    B31 = []
    B41 = []
    Red1 = []
    Red2 = []
    Red3 = []
    Red4 = []
    Green1 = []
    Green2 = []
    Green3 = []
    Green4 = []
    Blue1 = []
    Blue2 = []
    Blue3 = []
    Blue4 = []
    label1 = []
    distance = []

    accuracy2 = crossVal(table, validation, train, fileName1, R11, R21, R31, R41, B11, B21, B31, B41, G11, G21, G31, G41, Red1, Red2, Red3, Red4, Blue1, Blue2, Blue3, Blue4, Green1, Green2, Green3, Green4, label1,avgAcc, distance,k)
    validation = {}
    train = {}
    fileName1 = []
    R11 = []
    R21 = []
    R31 = []
    R41 = []
    G11 = []
    G21 = []
    G31 = []
    G41 = []
    B11 = []
    B21 = []
    B31 = []
    B41 = []
    Red1 = []
    Red2 = []
    Red3 = []
    Red4 = []
    Green1 = []
    Green2 = []
    Green3 = []
    Green4 = []
    Blue1 = []
    Blue2 = []
    Blue3 = []
    Blue4 = []
    label1 = []
    distance = []
    accuracy3 = crossVal(table, validation, train, fileName1, R11, R21, R31, R41, B11, B21, B31, B41, G11, G21, G31, G41, Red1, Red2, Red3, Red4, Blue1, Blue2, Blue3, Blue4, Green1, Green2, Green3, Green4, label1,avgAcc, distance,k)
    validation = {}
    train = {}
    fileName1 = []
    R11 = []
    R21 = []
    R31 = []
    R41 = []
    G11 = []
    G21 = []
    G31 = []
    G41 = []
    B11 = []
    B21 = []
    B31 = []
    B41 = []
    Red1 = []
    Red2 = []
    Red3 = []
    Red4 = []
    Green1 = []
    Green2 = []
    Green3 = []
    Green4 = []
    Blue1 = []
    Blue2 = []
    Blue3 = []
    Blue4 = []
    label1 = []
    distance = []
    avgAcc.append(accuracy1)
    avgAcc.append(accuracy2)
    avgAcc.append(accuracy3)
    
    pl.xlabel("Fold number")
    pl.plot(avgAcc, 'ro')
    pl.ylabel("Accuracy")
    pl.title("Accuracy wrt folds for cross validation")
    pl.show()
    return avgAcc
def accComp(k,avgAcc):
    tot = 0 
    for e in avgAcc:
        tot = tot + e
    accuracy = float(tot)/3
    return accuracy
    
def kval():
    kvalue = []
    for i in range(0,9):
        kvalue.append(i+1)
    accu = []
    accu.append(50)
    accu.append(49)
    accu.append(49)
    accu.append(50)
    accu.append(50)
    accu.append(50)
    accu.append(50)
    accu.append(50)
    accu.append(50)
    accu.append(50)

    pl.plot(accu, 'go')
    pl.ylabel("Accuracy")
    pl.xlabel("Number of neighbours")
    pl.title("Accuracy wrt k for cross validation")
    pl.show()
def Ga(flagtable):
    fs = []
    flagname = []
    fs, flagNames = sepFlag(flagtable)
    flag1 = fs[3]
    name1 = flagNames[3]
    flag2 = fs[9]
    name2 = flagNames[9]
    source = flag1
    target = flag2
    print(flag1)
    print(flag2)
    print(name1)
    print(name2)
    
    l = []
    x = source
    y = target

    while y!=x:
        x = aaa(x,y)
        print(x)
        if(y == x):
            print(y)
            print(x)
            print("RGB of first flag is same as RGB of second Flag")
            
def Ga1():
    pathFlag = raw_input("Path of Flags folder:")
    r1 = 0
    r2 = 0
    r3 = 0
    r4 = 0
    g1 = 0
    g2 = 0
    g3 = 0
    g4 = 0
    b1 = 0
    b2 = 0
    b3 = 0
    b4 = 0
    r11 = 0
    r21 = 0
    r31 = 0
    r41 = 0
    g11 = 0
    g21 = 0
    g31 = 0
    g41 = 0
    b11 = 0
    b21 = 0
    b31 = 0
    b41 = 0
    flagList = os.listdir(pathFlag)
    for flags in flagList:
        if not flags.startswith('.'):
                if flags=="Cyprus.jpg":
                    fs = Image.open(pathFlag + flags)
                    fs2 = Image.open(pathFlag + flags,'r')
                    rgbIm = fs.convert('RGB')
    
                    rgbPix = rgbIm.load()
                    height, width = fs.size
                    arr = np.array(rgbIm)
                    for x in range(0, width):
                        for y in range(0,height):
                            r, g, b = arr[x,y]
                            r1,g1,b1,r2,g2,b2,r3,g3,b3,r4,g4,b4 = computeRGB11(r,g,b,r1,g1,b1,r2,g2,b2,r3,g3,b3,r4,g4,b4)

                            pixels = height * width
                if flags=="England.jpg":
                    fs1 = Image.open(pathFlag + flags)
                    rgbIm1 = fs1.convert('RGB')
    
                    rgbPix1 = rgbIm1.load()
                    height1, width1 = fs1.size
                    arr1 = np.array(rgbIm1)
                    for x1 in range(0, width):
                        for y1 in range(0,height):
                            r1, g1, b1 = arr[x,y]
                            r11,g11,b11,r21,g21,b21,r31,g31,b31,r41,g41,b41 = computeRGB11(r1,g1,b1,r11,g11,b11,r21,g21,b21,r31,g31,b31,r41,g41,b41)

                            pixels1 = height1 * width1
    RGB = []
    RGB1 = []
    RGB.append(r1)
    RGB.append(r2)
    RGB.append(r3)
    RGB.append(r4)
    RGB.append(g1)
    RGB.append(g2)
    RGB.append(g3)
    RGB.append(g4)
    RGB.append(b1)
    RGB.append(b2)
    RGB.append(b3)
    RGB.append(b4)
    
    RGB1.append(r11)
    RGB1.append(r21)
    RGB1.append(r31)
    RGB1.append(r41)
    RGB1.append(g11)
    RGB1.append(g21)
    RGB1.append(g31)
    RGB1.append(g41)
    RGB1.append(b11)
    RGB1.append(b21)

    RGB1.append(b31)
    RGB1.append(b41)
    print("==========RGB of flag1==========")
    print(RGB)
    print("==========RGB of flag2===========")
    print(RGB1)
    print("\n\nPerforming GA to change RGB of flag1 to flag2")
    while RGB1!=RGB:
        RGB = aaa(RGB,RGB1)
        print(RGB)

        if(RGB1==RGB):
            print("GA stopped. RGB of flag1 = RGB of flag2")
 
def aaa(x,y):
    x = mutation(x,0.5,y)
    return x

def mutation(x, muta,y):
    for i in range(len(x)):
        if random.random() < muta:
            x[i] = type(x[i])(y[i])
    return x
def main():
    file_name="lookup.csv"
    cluster1 = []
    cluster2 = []
    
    table = {}
    train = {}
    exitFlag = 0
    fileName = []
    validation = {}
    distance = []
    R1 = []
    R2 = []
    R3 = []
    R4 = []
    G1 = []
    G2 = []
    G3 = []
    G4 = []
    B1 = []
    B2 = []
    B3 = []
    B4 = []
    R11 = []
    R21 = []
    R31 = []
    R41 = []
    G11 = []
    G21 = []
    G31 = []
    G41 = []
    B11 = []
    B21 = []
    B31 = []
    B41 = []
    Red1 = []
    Red2 = []
    Red3 = []
    Red4 = []
    Green1 = []
    Green2 = []
    Green3 = []
    Green4 = []
    Blue1 = []
    Blue2 = []
    Blue3 = []
    Blue4 = []
    label = []
    label1 = []
    avgAcc = []
    fileName1 = []
    flagtable = {}
    file2 = "flag.csv"
    flagC = []
    acs = 0
    path_file = raw_input("Please enter the path of folder where lookup file is:")
 
    if os.path.isfile(path_file+file_name):
        table = pickle.load( open( file_name, "rb" ) )
    else:
        try:
            pathLandscape = raw_input("Please enter the path of Landscape folder:")
            pathHeadshots = raw_input("Please enter the path of Headshot folder:")
            table = datasetPrep(file_name,table, fileName, R1, R2, R3, R4, B1, B2, B3, B4, G1, G2, G3, G4, label, pathLandscape, pathHeadshots)
        except:

            print("Something went wrong while reading the folder path. Please try again")
    if os.path.isfile(path_file+file2):
        flagtable = pickle.load( open( file2, "rb" ) )

    else:
        try:
            pathFlags = raw_input("Please enter the path of Flags folder:")
            flagtable = fp(file2,flagtable,pathFlags)

        except:

            print("Something went wrong while reading the folder path. Please try again")

    while exitFlag == 0:
        try:
            inputInt = int(raw_input("\nWhat would you like to do? \n\t0. Quit the program \n\t1. Upload image to know if it's a Landscape or Headshot \n\t2. See average accuracy of 3 fold cross validation \n\t3.K-means \n\t4.Hierarchical Clustering \n\t5.Hierarchical for Flags\n\t6. Cross validation for k = 1 to 10 \n\t7.Genetic Algorithm\n\tEnter your choice:"))
            if inputInt == 0:
                exitFlag = 1
                quit
            if inputInt == 1:
                environment(table)
                exitFlag = 0
            if inputInt == 2:
                k = int(raw_input("Please enter a value of k:"))
                avgAcc = []
                avgAcc = Threefold(table, validation, train, fileName1, R11, R21, R31, R41, B11, B21, B31, B41, G11, G21, G31, G41, Red1, Red2, Red3, Red4, Blue1, Blue2, Blue3, Blue4, Green1, Green2, Green3, Green4, label1,avgAcc, distance,k)
                acs = accComp(k,avgAcc)
                print("The average accuracy is:%d"% acs+"%")

##                accuracy = crossVal(table, validation, train, fileName1, R11, R21, R31, R41, B11, B21, B31, B41, G11, G21, G31, G41, Red1, Red2, Red3, Red4, Blue1, Blue2, Blue3, Blue4, Green1, Green2, Green3, Green4, label1,avgAcc, distance,k)
            if inputInt == 3:
                accuracy,cluster1, cluster2 = kmeans(table)
                print("The accuracy is:%d"% accuracy+"%")
                print("\n========Cluster 1==========")
                print cluster1
                print("\n========Cluster 2==========")
                print cluster2
                print("The accuracy is:%d"% accuracy+"%")

            if inputInt ==4:
                 cluster1 = hier(table)
                 print(cluster1)
            if inputInt ==5:
                flagC = flagProc(file2,flagtable)
                print(flagC)
            if inputInt ==6:
                kval()
            if inputInt ==7:
##                print("No result to show. Could not complete algorithm")
##                Ga(flagtable)
                Ga1()

        except:
            print("\nThere was an error encountered. Please try again")
            quit
        

main()
