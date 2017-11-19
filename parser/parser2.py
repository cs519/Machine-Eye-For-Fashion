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
            tstr = ' '.join(tlist) + '\n'
            outfile.write(tstr)
            #creating path and name for destination of image to be copied
            tstr = "output/" + clothesType + "/" + str(counter) + ".jpg"
            #copying image from source name, to destination tstr
            shutil.copy(name, tstr)
            counter += 1


if __name__ == "__main__":
    infile = "test_landmarks.txt"
    outfiles = ["1_landmarks.csv", "2_landmarks.csv", "3_landmarks.csv"] 
    parser = Parsing(infile, outfiles)