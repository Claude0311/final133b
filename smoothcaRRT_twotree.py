#!/usr/bin/env python3
#
#   carplanner_skeleton.py
#
#   Skeleton code to plan car movements.
#
import bisect
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from scipy.optimize import fsolve, least_squares

from copy import deepcopy
import time

from math          import pi, sin, cos, tan, sqrt, ceil, floor, atan #, atan2, asin
from planarutils   import *
from numpy import arctan2 as atan2, arcsin as asin
import cupy as cp

from test import distance as cp_distance, update_data

######################################################################
# SECTION 1: DEFINITIONS
######################################################################

#
# scenario = 'parallel parking'
# scenario = 'u turn'
scenario = 'move over'
# scenario = 'narrow park'
# scenario = 'blank'
# scenario = 'garage'
scenario = 'narrow garage'
scenario = 'load map'
scenario = 'dual parallel parking'

######################################################################
#
#   Car Definitions
#
#   Define the car dimensions and parameters.
#

# Car outline.
(lcar, wcar) = (4, 2)           # Length/width

# Center of rotation (center of rear axle).
lb           = 0.5              # Center of rotation to back bumper
lf           = lcar - lb        # Center of rotation to front bumper
wc           = wcar/2           # Center of rotation to left/right
wheelbase    = 3                # Center of rotation to front wheels

PHIMAX = np.pi / 6
DPHI = PHIMAX / 3
RMAX = wheelbase / tan(PHIMAX)
RMAX_SQUARE_4 = 4 * RMAX ** 2

######################################################################
#
#   World Defintions
#
#   This should select:
#
#   (xmin, xmax, ymin, ymax)        # Overall dimensions
#   (walls)                         # List of walls
#   (startx, starty, starttheta)    # Starting pose
#   (startx, starty, starttheta)    # Goal pose
#

### Parallel Parking:
if scenario == 'parallel parking':

    # General numbers.
    (wroad)                  = 6                # Road width
    (xspace, lspace, wspace) = (3, 6, 2.5)      # Parking Space pos/len/width

    # Overall size.
    (xmin, xmax) = (0, 14)
    (ymin, ymax) = (0, wroad+wspace)

    # Construct the walls.
    walls = (((xmin         , ymin        ), (xmax         , ymin        )),
             ((xmax         , ymin        ), (xmax         , wroad       )),
             ((xmax         , wroad       ), (xspace+lspace, wroad       )),
             ((xspace+lspace, wroad       ), (xspace+lspace, wroad+wspace)),
             ((xspace+lspace, wroad+wspace), (xspace       , wroad+wspace)),
             ((xspace       , wroad+wspace), (xspace       , wroad       )),
             ((xspace       , wroad       ), (xmin         , wroad       )),
             ((xmin         , wroad       ), (xmin         , ymin        )))

    # Pick your start and goal locations.
    (startx, starty, starttheta) = (1.0, 4.0, 0.0)
    (goalx,  goaly,  goaltheta)  = (xspace+(lspace-lcar)/2+lb, wroad+wc, 0.0)

### U Turn (3 point turn).
elif scenario == 'u turn':

    # General numbers.
    (xroad, yroad, wlane) = (5, 10, 3)  # road len, pos, lane width

    # Overall size.
    (xmin, xmax) = (0, 22)
    (ymin, ymax) = (0, 20)

    # Construct the walls.
    walls = (((xmin,  yroad - wlane), (xroad, yroad - wlane)),
             ((xroad, yroad - wlane), (xroad, ymin         )),
             ((xroad, ymin         ), (xmax,  ymin         )),
             ((xmax , ymin         ), (xmax,  ymax         )),
             ((xmax , ymax         ), (xroad, ymax         )),
             ((xroad, ymax         ), (xroad, yroad + wlane)),
             ((xroad, yroad + wlane), (xmin,  yroad + wlane)),
             ((xmin,  yroad + wlane), (xmin,  yroad - wlane)))

    # Pick your start and goal locations.
    (startx, starty, starttheta) = (1.0, yroad - 1.5, 0)
    (goalx,  goaly,  goaltheta)  = (4.0, yroad + 1.5, pi)

### Shift over
elif scenario == 'move over':

    # General numbers.
    (xrail, yrail, lrail) = (7.5, 4, 5)          # Guard rail pos/len

    # Overall size.
    (xmin, xmax) = (0, 20)
    (ymin, ymax) = (0, 20)

    # Construct the walls.
    walls = (((xrail, yrail), (xrail+lrail, yrail)),
             ((xmin , ymin ), (xmax       , ymin )),
             ((xmax , ymin ), (xmax       , ymax )),
             ((xmax , ymax ), (xmin       , ymax )),
             ((xmin , ymax ), (xmin       , ymin )))
    
    # Pick your start and goal locations.
    (startx, starty, starttheta) = (8.5, 2.0, 0.0)
    (goalx,  goaly,  goaltheta)  = (8.5, 6.0, 0.0)

elif scenario == 'garage':
    (xmin, xmax) = (0, 30)
    (ymin, ymax) = (0, 30)

    walls = [((xmin, ymin), (xmax, ymin)),
            ((xmax, ymin), (xmax, ymax)),
            ((xmax, ymax), (xmin, ymax)),
            ((xmin, ymax), (xmin, ymin)),
            ]

    walls_extend = [
        ((5,30),(0,25)),
        ((0,18),(5,18)),
        ((5,20),(5,18)),
        ((5,20),(0,20)),
        ((0,5),(5,0)),
        ((10,15),(10,22)),
        ((10,22),(12,22)),
        ((12,22),(12,17)),
        ((12,17),(15,17)),
        ((15,17),(15,13)),
        ((15,13),(13,13)),
        ((13,13),(13,15)),
        ((13,15),(10,15)),
        ((16,0),(16,13)),
        ((16,13),(18,13)),
        ((18,13),(18,0)),
        ((25,0),(30,5)),
        ((30,10),(20,10)),
        ((20,10),(20,12)),
        ((20,12),(30,12)),
        ((20,30),(20,20)),
        ((20,20),(23,20)),
        ((23,20),(23,28)),
        ((23,28),(27,28)),
        ((27,28),(27,18)),
        ((27,18),(30,18))
    ]
    walls.extend(walls_extend)

    (startx, starty, starttheta) = ( 7, 7, 0.25*pi)
    (goalx,  goaly,  goaltheta)  = (25,  22, 0.5*pi)


elif scenario == 'narrow garage':
    (xmin, xmax) = (0, 30)
    (ymin, ymax) = (0, 30)

    walls = [((xmin, ymin), (xmax, ymin)),
            ((xmax, ymin), (xmax, ymax)),
            ((xmax, ymax), (xmin, ymax)),
            ((xmin, ymax), (xmin, ymin)),
            ]

    walls_extend = [
        ((5,30),(0,25)),
        ((0,18),(5,18)),
        ((5,20),(5,18)),
        ((5,20),(0,20)),
        ((0,5),(5,0)),
        ((10,15),(10,22)),
        ((10,22),(12,22)),
        ((12,22),(12,17)),
        ((12,17),(15,17)),
        ((15,17),(15,13)),
        ((15,13),(13,13)),
        ((13,13),(13,15)),
        ((13,15),(10,15)),
        ((16,0),(16,13)),
        ((16,13),(18,13)),
        ((18,13),(18,0)),
        ((25,0),(30,5)),
        ((30,10),(20,10)),
        ((20,10),(20,12)),
        ((20,12),(30,12)),
        ((20,30),(20,20)),
        ((20,20),(23,20)),
        ((23,20),(23,28)),
        ((23,28),(27,28)),
        ((27,28),(27,18)),
        ((27,18),(30,18)),
        ((7,30),(7,25)),
        ((7,25),(18,25)),
        ((18,25),(18,30))
    ]
    walls.extend(walls_extend)

    (startx, starty, starttheta) = ( 7, 7, 0.25*pi)
    (goalx,  goaly,  goaltheta)  = (25,  22, 0.5*pi)

elif scenario=='blank':
    (xmin, xmax) = (0, 30)
    (ymin, ymax) = (0, 30)

    walls = [((xmin, ymin), (xmax, ymin)),
            ((xmax, ymin), (xmax, ymax)),
            ((xmax, ymax), (xmin, ymax)),
            ((xmin, ymax), (xmin, ymin)),
            ]

    def ranpos():
        return random.uniform(xmin+5, xmax-5), random.uniform(ymin+5, ymax-5), random.uniform(-np.pi, np.pi)

    (startx, starty, starttheta) = ranpos() #(25, 5, 0.5*pi) # 
    (goalx,  goaly,  goaltheta)  = ranpos() #(5,  25, 0.5*pi) # 
    # (startx, starty, starttheta) = (25, 5, 0.5*pi) # 
    # (goalx,  goaly,  goaltheta)  = (5,  25, 0.5*pi) # 

elif scenario=='load map':
    (xmin, xmax) = (0, 30)
    (ymin, ymax) = (0, 30)

    walls = [((xmin, ymin), (xmax, ymin)),
            ((xmax, ymin), (xmax, ymax)),
            ((xmax, ymax), (xmin, ymax)),
            ((xmin, ymax), (xmin, ymin)),
            ]

    f = np.load('map2.npy')
    for [[x0,y0],[x1,y1]] in  f:
        walls.append(((x0/20,ymax-y0/20),(x1/20,ymax-y1/20)))
    
    (startx, starty, starttheta) = ( 7, 7, 0.25*pi)
    (goalx,  goaly,  goaltheta)  = (25,  22, 0.5*pi)

elif scenario=='dual parallel parking':
    (xmin, xmax) = (0, 30)
    (ymin, ymax) = (0, 30)

    walls = [((xmin, ymin), (xmax, ymin)),
            ((xmax, ymin), (xmax, ymax)),
            ((xmax, ymax), (xmin, ymax)),
            ((xmin, ymax), (xmin, ymin)),
            ]

    f = np.load('map3.npy')
    for [[x0,y0],[x1,y1]] in  f:
        walls.append(((x0/20,ymax-y0/20),(x1/20,ymax-y1/20)))
    
    (startx, starty, starttheta) = ( 4, 12, 0.0*pi)
    (goalx,  goaly,  goaltheta)  = (18,  12, 0.0*pi)


elif scenario=='narrow park':
    (xmin, xmax) = (0, 30)
    (ymin, ymax) = (0, 30)

    walls = [((xmin, ymin), (xmax, ymin)),
            ((xmax, ymin), (xmax, ymax)),
            ((xmax, ymax), (xmin, ymax)),
            ((xmin, ymax), (xmin, ymin)),
            ]

    walls_extend = [((0,25),(21.58,25)),
                    ((21.58,25),(21.58,20)),
                    ((21.58,20),(0,20)),
                    ((30,20),(18,12)),
                    ((18,12),(18,0)),
                    ((5.23,15.16),(11.28,13.85)),
                    ((11.28,13.85),(13.17,7.95)),
                    ((13.17,7.95),(9,3.37)),
                    ((9,3.37),(2.95,4.68)),
                    ((2.95,4.68),(1.07,10.58)),
                    ((1.07,10.58),(5.23,15.16))
                    ]

    walls.extend(walls_extend)
    
    (startx, starty, starttheta) = ( 1, 27.5, 0)
    (goalx,  goaly,  goaltheta)  = (5, 1.5 , pi)

### Unknown.
else:
    raise ValueError("Unknown Scenario")


######################################################################
#
#   Algorithm Parameters
#
#   Grid:
#     dstep             Distance (meters) traveled per move
#     thetastep         Angle (radian) rotated per non-straight move
#
#   Costs:
#     cstep             Cost per each step
#     csteer            Cost for each change in steering angle
#     creverse          Cost for each change in fworward/back direction
#
#   TODO: Please change the costs as you explore
#

cstep    =   1
csteer   =  0
creverse = 0
dt = 1

if scenario == 'parallel parking':
    dstep      = 0.4            # Distrance driven per move
    thetastep = pi/36           # Angle rotated per non-straight move

elif scenario == 'u turn':
    dstep     = 0.5             # Distrance driven per move
    thetastep = pi/36           # Angle rotated per non-straight move

elif scenario == 'move over':
    dstep     = 0.5             # Distrance driven per move
    thetastep = pi/36           # Angle rotated per non-straight move

elif scenario == 'garage':
    dstep     = 0.5             # Distrance driven per move
    thetastep = pi/36 

elif scenario == 'narrow park':
    dstep     = 0.5             # Distrance driven per move
    thetastep = pi/36 

elif scenario == 'narrow garage':
    dstep     = 0.5             # Distrance driven per move
    thetastep = pi/36 

elif scenario == 'blank':
    dstep      = 0.4            # Distrance driven per move
    thetastep = pi/36

elif scenario == 'load map':
    dstep      = 0.4            # Distrance driven per move
    thetastep = pi/36

else:
    dstep      = 0.4            # Distrance driven per move
    thetastep = pi/36
    # raise ValueError("Unknown Scenario")


def wrap360(theta):
    # return 0~360
    return theta - floor(theta/(np.pi*2))*np.pi*2
def wrap180(theta):
    # return -180~180
    return theta - round(theta/(np.pi*2))*np.pi*2

######################################################################
# SECTION 2: NODE AND VISUALIZATION
######################################################################

######################################################################
#
#   Node Definition
#
# Initialize the counters (for diagnostics only)
nodecounter = 0
donecounter = 0
nodecount = True

# Make sure thetastep is an integer fraction of 2pi, so the grid wraps nicely.
#thetastep = dstep * tan(steerangle) / wheelbase
thetastep  = 2*pi / round(2*pi/thetastep)
steerangle = np.arctan(wheelbase * thetastep / dstep)

# Node Class
class Node:
    def __init__(self, x, y, theta):
        # Setup the basic A* node.
        super().__init__()

        # Remember the state (x,y,theta).
        self.x     = x
        self.y     = y
        self.theta = theta

        # Precompute/save the trigonometry
        self.s = np.sin(theta)
        self.c = np.cos(theta)

        # circle center
        self.rc = [
            (x + RMAX*self.s, y - RMAX*self.c),
            (x - RMAX*self.s, y + RMAX*self.c)
        ]

        # Box = 4 corners: frontleft, backleft, backright, frontright
        self.box = ((x + self.c*lf - self.s*wc, y + self.s*lf + self.c*wc),
                    (x - self.c*lb - self.s*wc, y - self.s*lb + self.c*wc),
                    (x - self.c*lb + self.s*wc, y - self.s*lb - self.c*wc),
                    (x + self.c*lf + self.s*wc, y + self.s*lf - self.c*wc))

        # Tree connectivity.  Define how we got here (set defaults for now).
        self.parent  = None
        self.forward = 1        # +1 = drove forward, -1 = backward
        self.steer   = 0        # +1 = turned left, 0 = straight, -1 = right

        # Cost/status.
        self.cost = 0           # Cost to get here.
        self.done = False       # The path here is optimal.

        # Count the node - for diagnostics only.
        global nodecount
        if nodecount:
            global nodecounter
            nodecounter += 1
            if nodecounter % 1000 == 0:
                print("Sampled %d nodes... " %
                    (nodecounter))
            
    def distance(self, other, additionalInfo = False):
        min_d = float('inf')

        for s in [0, 1]:
            for o in [0, 1]:
                base_d = (self.rc[s][0]-other.rc[o][0])**2 + (self.rc[s][1]-other.rc[o][1])**2
                theta_cc = atan2( other.rc[o][1]-self.rc[s][1], other.rc[o][0]-self.rc[s][0] )
                
                if s-o!=0: 
                    if base_d < RMAX_SQUARE_4: continue
                    dtheta = asin( 2*RMAX/sqrt(base_d) )
                    base_d -= RMAX_SQUARE_4
                    


                base_d = sqrt(base_d)
                

                for forward in [1, -1]:
                    if base_d < 2*dt and s==o:
                        delta = wrap360(other.theta-self.theta) if (s, forward) in [(1,1),(0,-1)] else wrap360(self.theta-other.theta)
                        tmp_d = RMAX * delta + base_d
                    else:
                        (t_from, t_to) = (self.theta, other.theta) if forward == 1 else (other.theta, self.theta)
                        (orientation_from, orientation_to) = (s, o) if forward == 1 else (o, s)
                        tc = theta_cc if forward==1 else (theta_cc+np.pi) if theta_cc<0 else (theta_cc-np.pi)

                        if s-o!=0:
                            if orientation_from==0: tc = tc - dtheta
                            else: tc = tc + dtheta

                        dt1 = wrap360(tc-t_from) if orientation_from==1 else wrap360(t_from-tc)
                        dt2 = wrap360(t_to-tc) if orientation_to==1 else wrap360(tc-t_to)

                        tmp_d = base_d + RMAX * (dt1+dt2)
                        # print(dt1, dt2)

                    if tmp_d<min_d:
                        min_d = tmp_d

                    # print(s, o, forward, tmp_d, base_d, dt1, dt2, tc)
        return min_d
        if not additionalInfo:
            return min(min_d, Node.distance(other, self, True))
        else:
            return min_d

        # return 
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2 +
                    wheelbase**2 * (self.s - other.s)**2 + wheelbase**2 * (self.c - other.c)**2)
    
    def goal_distance(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + wheelbase * ((self.s - other.s)**2 + (self.c - other.c)**2))

    ############
    # Utilities:
    # In case we want to print the node.
    def __repr__(self):
        return ("<XY %5.2f,%5.2f @ %5.1f deg> (fwd %2d, str %5.1f deg)" %
                (self.x, self.y, np.degrees(self.theta),
                 self.forward, np.degrees(self.steer)))
    
    # Define the "less-than" to enable sorting by cost!
    def __lt__(self, other):
        return (self.cost < other.cost)

    # Determine the grid indices based on the coordinates.  The x/y
    # coordinates are regular numbers, the theta maps to 0..360deg!
    def indices(self):
        return (round((self.x - xmin)/dstep),
                round((self.y - ymin)/dstep),
                round(self.theta/thetastep) % round(2*pi/thetastep))


    #####################
    # Forward Simulations
    # Instantiate a new (child) node, based on driving from this node.
    def nextNode(self, forward, steer):
        # TODO - Problem 1, write the simulation.
        # Assume forward = +1  or -1,   to be scaled by dstep
        #        steer   = +1, 0, -1,   to be scaled by thetastep
        if steer == 0:
            xnext = self.x + (dstep*forward) * cos(self.theta)
            ynext = self.y + (dstep*forward) * sin(self.theta)
            thetanext = self.theta
        else:
            dtheta = thetastep * steer * forward
            r = dstep / thetastep * steer
            thetanext = self.theta + dtheta
            xnext = self.x + r * (sin(thetanext) - sin(self.theta))
            ynext = self.y - r * (cos(thetanext) - cos(self.theta))

        child = Node(xnext, ynext, thetanext)

        # Set the parent relationship.
        child.parent  = self
        child.forward = forward
        child.steer   = steer

        child.cost = self.cost + cstep
        if self.forward!=child.forward: child.cost += creverse
        if self.steer != child.steer: child.cost += csteer

        # Return
        return child
    

    ######################
    # Collision functions:
    # Check whether in free space.
    def inFreespace(self):
        for wall in walls:
            if SegmentCrossBox(wall, self.box):
                return False
        return True

    # Check the local planner - whether this connects to another node.
    def connectsTo(self, other):
        for wall in walls:
            if (SegmentCrossSegment(wall, (self.box[0], other.box[0])) or
                SegmentCrossSegment(wall, (self.box[1], other.box[1])) or
                SegmentCrossSegment(wall, (self.box[2], other.box[2])) or
                SegmentCrossSegment(wall, (self.box[3], other.box[3]))):
                return False
        return True


######################################################################
#
#   Visualization
#
class Visualization:
    def __init__(self):
        # Clear the current, or create a new figure.
        plt.clf()

        # Create a new axes, enable the grid, and set axis limits.
        plt.axes()
        plt.grid(True)
        plt.gca().axis('on')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)
        plt.gca().set_aspect('equal')

        # Show the walls.
        for wall in walls:
            plt.plot([wall[0][0], wall[1][0]],
                     [wall[0][1], wall[1][1]], 'k', linewidth=2)

        # Mark the locations.
        plt.gca().set_xticks(list(set([wall[0][0] for wall in walls]+
                                      [wall[1][0] for wall in walls]+
                                      [startx, goalx])))
        plt.gca().set_yticks(list(set([wall[0][1] for wall in walls]+
                                      [wall[1][1] for wall in walls]+
                                      [starty, goaly])))

        # Show.
        self.show()

    def show(self, text = '', delay = 0.001):
        # Show the plot.
        plt.pause(max(0.001, delay))
        # If text is specified, print and maybe wait for confirmation.
        if len(text)>0:
            if delay>0:
                input(text + ' (hit return to continue)')
            else:
                print(text)

    def drawNode(self, node, *args, **kwargs):
        b = node.box
        tmp_plot = []
        # Box
        tmp_plot.append(plt.plot((b[0][0], b[1][0]), (b[0][1], b[1][1]), *args, **kwargs))
        tmp_plot.append(plt.plot((b[1][0], b[2][0]), (b[1][1], b[2][1]), *args, **kwargs))
        tmp_plot.append(plt.plot((b[2][0], b[3][0]), (b[2][1], b[3][1]), *args, **kwargs))
        tmp_plot.append(plt.plot((b[3][0], b[0][0]), (b[3][1], b[0][1]), *args, **kwargs))
        # Headlights
        tmp_plot.append(plt.plot(0.9*b[3][0]+0.1*b[0][0], 0.9*b[3][1]+0.1*b[0][1],
                 *args, **kwargs, marker='o'))
        tmp_plot.append(plt.plot(0.1*b[3][0]+0.9*b[0][0], 0.1*b[3][1]+0.9*b[0][1],
                 *args, **kwargs, marker='o'))
        
        # for s in [0,1]:
        #     tmp_plot.append(plt.plot(node.rc[s][0], node.rc[s][1], 
        #             *args, **kwargs, marker='o'))
        
        kwargs.update(linewidth=5)
        for head in [0.125, 0.875]:
            for left in [0.1, 0.9]:
                theta = node.theta
                x = head * (left * b[3][0]+ (1-left) * b[0][0]) + (1-head) * (left * b[2][0]+ (1-left) * b[1][0])
                y = head * (left * b[3][1]+ (1-left) * b[0][1]) + (1-head) * (left * b[2][1]+ (1-left) * b[1][1])
                if head==0.875: 
                    theta += node.steer
                tmp_plot.append(plt.plot( (x + 0.25*cos(theta), x - 0.25*cos(theta)), (y + 0.25*sin(theta), y - 0.25*sin(theta)), 
                                        *args, **kwargs))
        return tmp_plot

    def drawEdge(self, head, tail, *args, **kwargs):
        plt.plot([head.x, tail.x], [head.y, tail.y], *args, **kwargs)

    def drawPath(self, path, show_track = False, *args, **kwargs):
        for node in path:
            tmp_plot = self.drawNode(node, *args, **kwargs)
            self.show(delay = 0.1)
            if not show_track:
                for tmp in tmp_plot:
                    t = tmp.pop(0)
                    t.remove()

######################################################################
#
#   goal by bias
#
TOLERANCE_SAMPLE = wheelbase/2
TOLERANCE_TOGOAL = 0.5
Nmax  = float('inf')

def get_target(tree):
    p = random.random()
    if p<0.05:
        return tree[0]
    else:
        targetnode = Node( 
                    random.uniform(xmin, xmax),
                    random.uniform(ymin, ymax),
                    random.uniform(-np.pi, np.pi)
            )
    return targetnode

######################################################################
#
#   optimize input collision detection
#
cp.linspace(-PHIMAX, PHIMAX, int(PHIMAX//DPHI) * 2 + 1)
s_all, phi_all = cp.meshgrid(cp.array([-1, 1]), cp.linspace(-PHIMAX, PHIMAX, int(PHIMAX//DPHI) * 2 + 1))
s_all = s_all.flatten()
phi_all = phi_all.flatten()

def omp_input_coldet( nearnode, targetnode):
    theta = nearnode.theta
    x_now = nearnode.x
    y_now = nearnode.y

    s_opt = 0
    phi_opt = 0
    distance_min = float('inf')
    opt_node = None

    for s in [-1, 1]:
        tmp_phi = -PHIMAX
        while tmp_phi<=PHIMAX:
            xn = x_now + s * dt * cos(theta)
            yn = y_now + s * dt * sin(theta)
            theta_n = theta + s * tan(tmp_phi)/wheelbase * dt
            if tmp_phi == nearnode.steer and s == -nearnode.forward: # skip the case that return to near
                tmp_phi += DPHI
                continue
            pre_next_node = Node(xn, yn, theta_n)
            if pre_next_node.inFreespace():
                tmp_d = pre_next_node.distance(targetnode)
                if distance_min>tmp_d:
                    distance_min = tmp_d
                    s_opt = s
                    phi_opt = tmp_phi
                    opt_node = pre_next_node
                    # print(phi_opt, s_opt, tmp_d, theta_n, yn)
            tmp_phi += DPHI
    if opt_node is None or not opt_node.inFreespace(): return None
    opt_node.steer = phi_opt
    opt_node.forward = s_opt

    return opt_node

  

######################################################################
#
#   RRT Functions
#
K_RATIO_OF_R_N_D = 2

def rrt(startnode, goalnode, visual):
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    tree = [startnode]
    tree2 = [goalnode]

    # Function to attach a new node to an existing node: attach the
    # parent, add to the tree, and show in the figure.
    def addtotree(oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)
        visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
        visual.show()
    
    def addtotree2(oldnode, newnode):
        newnode.parent = oldnode
        tree2.append(newnode)
        visual.drawEdge(oldnode, newnode, color='orange', linewidth=1)
        visual.show()

    # Loop - keep growing the tree.
    r = float('inf')
    tmp_min = float('inf')

    datas = cp.zeros((10000000,7))
    data2 = cp.zeros((10000000,7))
    update_data(datas, 0, startnode)
    update_data(data2, 0, goalnode)

    swap = True

    while True:
        # Determine the target state.
        if swap:
            ##########################################################################
            targetnode = get_target(tree2)

            # Directly determine the distances to the target node.
            if True:#len(tree)>10000:
                index = cp_distance(datas[:len(tree),], targetnode)
            else:
                distances = np.array([node.distance(targetnode) for node in tree])
                index     = np.argmin(distances)
            nearnode  = tree[index]

            # Determine the next node.
            nextnode1 = omp_input_coldet(nearnode, targetnode)
            
            if nextnode1 is None: continue
            update_data(datas, len(tree), nextnode1)

            # Check whether to attach.
            addtotree(nearnode, nextnode1)
            ##########################################################################
            targetnode = nextnode1

            # Directly determine the distances to the target node.
            if True:#len(tree)>10000:
                index = cp_distance(data2[:len(tree2),], targetnode)
            else:
                distances = np.array([node.distance(targetnode) for node in tree])
                index     = np.argmin(distances)
            nearnode  = tree2[index]

            # Determine the next node.
            nextnode2 = omp_input_coldet(nearnode, targetnode)
            
            if nextnode2 is None: continue
            update_data(data2, len(tree2), nextnode2)

            # Check whether to attach.
            addtotree2(nearnode, nextnode2)

            swap = not swap
        else:
            ##########################################################################
            targetnode = get_target(tree)

            # Directly determine the distances to the target node.
            if True:#len(tree)>10000:
                index = cp_distance(data2[:len(tree2),], targetnode)
            else:
                distances = np.array([node.distance(targetnode) for node in tree])
                index     = np.argmin(distances)
            nearnode  = tree2[index]

            # Determine the next node.
            nextnode2 = omp_input_coldet(nearnode, targetnode)
            
            if nextnode2 is None: continue
            update_data(data2, len(tree2), nextnode2)

            # Check whether to attach.
            addtotree2(nearnode, nextnode2)

            ##########################################################################
            targetnode = nextnode2

            # Directly determine the distances to the target node.
            if True:#len(tree)>10000:
                index = cp_distance(datas[:len(tree),], targetnode)
            else:
                distances = np.array([node.distance(targetnode) for node in tree])
                index     = np.argmin(distances)
            nearnode  = tree[index]

            # Determine the next node.
            nextnode1 = omp_input_coldet(nearnode, targetnode)
            
            if nextnode1 is None: continue
            update_data(datas, len(tree), nextnode1)

            # Check whether to attach.
            addtotree(nearnode, nextnode1)
            ##########################################################################
            swap = not swap

            # If within dstep, also try connecting to the goal.  If
            # the connection is made, break the loop to stop growing.
            # TODO:
        d_next_to_goal = nextnode1.goal_distance(nextnode2)

        if d_next_to_goal < TOLERANCE_TOGOAL: # and nextnode.connectsTo(goalnode):
            # addtotree(nextnode, goalnode)
            break

        # Check whether we should abort (tree has gotten too large).
        if (len(tree) >= Nmax) or (len(tree2) >= Nmax):
            return (None, len(tree) + len(tree2))

    # Build and return the path.
    path1 = [nextnode1]
    path2 = [nextnode2]
    while path1[0].parent is not None:
        path1.insert(0, path1[0].parent)

    while path2[-1].parent is not None:
        path2.append(path2[-1].parent)
    
    path1.extend(path2)
    return path1, len(tree) + len(tree2)


def PostProcess_smooth(path):
    if len(path)==1: return path
    new_path = [path[0]]
    while len(new_path)<len(path):
        nextnode = omp_input_coldet(new_path[-1],path[-1])
        if nextnode is None: break
        new_path.append(nextnode)
        if nextnode.goal_distance(path[-1]) < TOLERANCE_TOGOAL:
            return new_path
    path1 = path[0:len(path)//2+1]
    path2 = path[len(path)//2:]
    path1 = PostProcess_smooth(path1)
    path2 = PostProcess_smooth(path2)
    path1.pop()
    path1.extend(path2)
    return path1



def disp_metrics(path,plan_time,tree_size,smooth_time=None):

    path_length = 0
    smoothness = 0

    for i in range(len(path)):
        print(path[i])

        if i<len(path)-1:
            phi = path[i+1].steer
            smoothness += np.tan(phi)**2
            if abs(phi)>1e-4:
                path_length += abs((wheelbase/np.tan(phi))*(path[i+1].theta - path[i].theta))
            else:
                path_length += np.sqrt((path[i+1].x-path[i].x)**2 + (path[i+1].y-path[i].y)**2)

    path_length = round(path_length,3)
    smoothness = round(smoothness,3)

    print("="*40)
    print("Metrics for Raw Path") if smooth_time is None else print("Metrics for Smoothed Path")
    print("-"*40)

    print("Planning Time           : %s s"%(plan_time))
    if smooth_time is not None:
        print("Smoothing Time           : %s s"%(smooth_time))
    print("# Nodes Sampled         : "+str(nodecounter))
    print("# Nodes in Path         : "+str(len(path)))
    print("Tree Size               : "+str(tree_size))
    print("Path Length             : "+str(path_length))
    print("Path Smoothness         : "+str(smoothness))
    print("Outcome                 : Success")

    print("="*40)







######################################################################
#
#   Main Code
#
def main():
    # Create the figure.
    visual = Visualization()


    print("-"*50)
    print("RRT based Planner for Non-Holonomic Mobile Robots (Car)")
    print("-"*50)



    # Create the start/goal nodes.
    startnode = Node(startx, starty, starttheta)
    goalnode  = Node(goalx,  goaly,  goaltheta)

    # print(startnode.distance(goalnode))

    # Show the start/goal nodes.
    visual.drawNode(startnode, color='c', linewidth=2)
    visual.drawNode(goalnode,  color='m', linewidth=2)


    visual.show("Showing basic world", 0.1)

    start_time = time.time()
    # Run the planner.
    print("Running planner...")
    path,tree_size = rrt(startnode, goalnode, visual)

    # If unable to connect, show what we have.
    if not path:
        visual.show("UNABLE TO FIND A PATH")
        print("Outcome                 : Failure")
        return

    plan_time = round(time.time() - start_time,2)


    # Print/Show the path.
    print("PATH found after sampling %d nodes" % nodecounter)

    disp_metrics(path,plan_time,tree_size)

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


    disp_metrics(smooth_path,plan_time,tree_size,smooth_time)

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
