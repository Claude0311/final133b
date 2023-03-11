from smoothcaRRT_trailer import Node, Visualization
from smoothcaRRT_trailer import *
from math import pi
import csv
import sys
sys.setrecursionlimit(10000)
datas = []
path = []

with open('datas/trailer.csv', encoding='utf-8') as f:
    spamreader = csv.reader(f, delimiter=' ', quotechar=',')
    for s in spamreader:
        print(s)
        # if len(s)!=6:continue
        [x, y, t0, t1, forw, steer] = s[0].split(',')
        if x.startswith('\ufeff'): x = x[1:]
        
        x = float(x)
        y = float(y)
        t0 = float(t0)
        t1 = float(t1)
        forw = float(forw)
        steer = float(steer)
        t0 = t0/180*pi
        t1 = t1/180*pi
        steer = steer/180*pi
        tmpn = Node(x, y, t0, t0+t1)
        tmpn.steer = steer
        tmpn.forward = forw
        path.append(tmpn)

visual = Visualization()

######################################################################
#
#   Main Code
#
def main():
    # Create the figure.
    visual = Visualization()
    global path


    print("-"*50)
    print("RRT based Planner for Non-Holonomic Mobile Robots (Car+Trailer)")
    print("-"*50)



    # Create the start/goal nodes.
    startnode = Node(startx, starty, starttheta)
    goalnode  = Node(goalx,  goaly,  goaltheta)
    # internode = Node(interx, intery, intertheta)

    # print(startnode.distance(goalnode))

    # Show the start/goal nodes.
    visual.drawNode(startnode, color='c', linewidth=2)
    visual.drawNode(goalnode,  color='m', linewidth=2)
    # visual.drawNode(internode, color='m', linewidth=2)


    visual.show("Showing basic world", 0.1)

    

    # Print/Show the path.
    print("PATH found after sampling %d nodes" % nodecounter)


    while True:
        visual.drawPath(path, color='r', linewidth=2)
        visual.show("Showing the Raw PATH", delay=0)
        q = input()
        if q=='q': break
        if q=='p':
            visual.drawPath(path, show_track=True, color='r', linewidth=2)
            visual.show("Showing the Raw PATH")
            break

    post1 = time.time()

    global nodecount
    nodecount = False
    smooth_path = PostProcess_smooth(path)


    smooth_time = round(time.time() - post1,2)


    while True:
        visual.drawPath(smooth_path, color='b', linewidth=2)
        visual.show("Showing the Smoothed PATH", delay=0)
        q = input()
        if q=='q': break
        if q=='p':
            visual.drawPath(smooth_path, show_track=True, color='b', linewidth=2)
            visual.show("Showing the Smoothed PATH")
            break


    

if __name__== "__main__":
    main()

        