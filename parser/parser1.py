import cv2
import numpy as np 

class Parsing:
    def __init__(self, ifilename, ofilename):
        with open(ifilename) as infile:
            data = infile.readlines()
        data.pop(0) #removing number of images line
        data.pop(0) #removing line of descriptors
        lList = []
        with open(ofilename, mode = 'w') as outfile:
            for line in data:
                tlist = line.split()
                name = tlist.pop(0) #removing name of image as we don't need it
                #checking if the list called line has less than 26 list elements
                #if it does, then we fill with non existing landmark points
                while (len(tlist) < 26):
                    tlist.append('2')
                    tlist.append('000')
                    tlist.append('000')
                tstr = ' '.join(tlist) + '\n'
                outfile.write(tstr)

if __name__ == "__main__":
    infile = "list_landmarks.txt"
    outfile = "list_landmarks.csv"
    parser = Parsing(infile, outfile)