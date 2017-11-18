from landmarks import Landmarks


class LandmarkParser:

    def __init__(self, filename):
        with open(filename) as infile:
            data = infile.readlines()
        n = int(data.pop(0))
        data.pop(0)
        for line in data:
            tlist = line.split()
            name = tlist.pop(0)
            tlist.pop(0)
            tlist.pop(0)
            lList = []
            while (tlist):
                if int(tlist.pop(0)) == 2:
                    tlist.pop(0)
                    tlist.pop(0)
                else:
                    lList.append((int(tlist.pop(0)), int(tlist.pop(0))))
            landmarks = Landmarks(name, lList)
            landmarks.showLand()
