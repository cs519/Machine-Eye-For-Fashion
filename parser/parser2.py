import cv2
import shutil, os
import os.path as path
import numpy as np 

class Parsing:
    def __init__(self, ifilename, ofilenames):
        with open(ifilename) as infile:
            data = infile.readlines()
        data.pop(0) #removing number of images line
        data.pop(0) #removing line of descriptors
        List1 = []
        List2 = []
        List3 = []
        #separate each line into its respective list
        for line in data: 
            tlist = line.split()
            if tlist[1] == '1': 
                List1.append(tlist) 
            elif tlist[1] == '2':
                List2.append(tlist)
            else:
                List3.append(tlist)
        #running separate function to create cvs file and separate images
        #into their respective directories
        separate(ofilenames[0], '1', List1)
        separate(ofilenames[1], '2', List2)
        separate(ofilenames[2], '3', List3)

def separate(filename, clothesType, myList):
    with open(filename, mode = 'w') as outfile:
        #counter that keeps track of which image number we are writing
        counter = 1
        while(myList):
            tlist = myList.pop(0)
            #obtaining path and name of image so we can copy it later
            name = tlist.pop(0)
            #creating cvs line
            nline = [counter, tlist.pop(0), tlist.pop(0)]
            vVals = []
            while(tlist):
                visibility = tlist.pop(0)
                nline.append(tlist.pop(0))
                nline.append(tlist.pop(0))
                if visibility == '0':
                    vVals.append('1')
                    vVals.append('0')
                    vVals.append('0')
                elif visibility == '1':
                    vVals.append('0')
                    vVals.append('1')
                    vVals.append('0')
                else:
                    vVals.append('0')
                    vVals.append('0')
                    vVals.append('1')
            nline.extend(vVals)
            tstr = ' '.join(nline) + '\n'
            outfile.write(tstr)
            #creating path and name for destination of image to be copied
            tstr = "output/" + clothesType + "/" + str(counter) + ".jpg"
            #copying image from source name, to destination tstr
            shutil.copy(name, tstr)
            counter += 1


if __name__ == "__main__":
    import os
    from os import path

    infile = "list_landmarks.txt"
    outfiles = ["1_landmarks.csv", "2_landmarks.csv", "3_landmarks.csv"] 
    
    for i in range(1, 4, 1):
        if not path.exists('output/{}'.format(i)):
            os.makedirs('output/{}'.format(i), mode=0o777, exist_ok=False)
    
    parser = Parsing(infile, outfiles)
