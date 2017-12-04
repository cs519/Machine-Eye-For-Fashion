from landmarks import Landmarks


class LandmarkParser:
    """
    Class to parse the landmarks txt file and pass the information to 
    landmarks class

    :param: filename: a string containing the landmarks txt file to be parsed
    """
    def __init__(self, filename):
        with open(filename) as infile: #open file to be parsed
            data = infile.readlines() #reach each line to a list
        n = int(data.pop(0)) #get total number of images
        data.pop(0) #getting rid of data definitions
        for line in data:
            tlist = line.split() #split the string to a list
            name = tlist.pop(0) #get name of img
            tlist.pop(0) #get rid of clothes type
            tlist.pop(0) #get rid of varyation type
            lList = [] #create new list of landmark coordinates
            while (tlist):
                if int(tlist.pop(0)) == 2: #landmark not visible
                    tlist.pop(0) #getting rid of x y coordinates at (0,0)
                    tlist.pop(0)
                else: #add xy coordinates to landmark coordinates list
                    lList.append((int(tlist.pop(0)), int(tlist.pop(0))))
            landmarks = Landmarks(name, lList) #create landmarks class
            landmarks.showLand() #write landmarks on top of image
