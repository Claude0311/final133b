import numpy as np
import cupy as cp
# print(cuda.gpus)

from smoothcaRRT_module import Node, RMAX_SQUARE_4, RMAX, dt


def wrap360(theta):
    # return 0~360
    return theta - cp.floor(theta/(cp.pi*2))*cp.pi*2

def flip(theta):
    # return 0~360
    return theta - cp.floor(theta/(cp.pi*2))*cp.pi*2

def update_data(datas, index, node):
    datas[index, 0] =     node.x
    datas[index, 1] =     node.y
    datas[index, 2] =     node.theta
    datas[index, 3] =     node.rc[0][0]
    datas[index, 4] =     node.rc[0][1]
    datas[index, 5] =     node.rc[1][0]
    datas[index, 6] =     node.rc[1][1]
    # datas[index, :] = cp.array([
    #     node.x,
    #     node.y,
    #     node.theta,
    #     node.rc[0][0],
    #     node.rc[0][1],
    #     node.rc[1][0],
    #     node.rc[1][1]
    # ])

def distance(datas, target_node :Node):
    # 0: x
    # 1: y
    # 2: theta
    # 3: rc[0] x
    # 4: rc[0] y
    # 5: rc[1] x
    # 6: rc[1] y
    node_num = len(datas)
    glob_distance = cp.full(node_num, cp.inf)
    for s in [0,1]:
        for o in [0,1]:
            based = (datas[:, 3+s*2]-target_node.rc[o][0])**2 + (datas[:, 4+s*2]-target_node.rc[o][1])**2
            theta_cc = cp.arctan2( target_node.rc[o][1]-datas[:, 4+s*2], target_node.rc[o][0]-datas[:, 3+s*2])
            distance_filter = cp.ones((node_num), dtype=bool)
            dtheta = cp.zeros((node_num), dtype=float)
            if s-o!=0:
                distance_filter[based<RMAX_SQUARE_4] = False
                dtheta[distance_filter] = cp.arcsin( 2*RMAX/cp.sqrt(based[distance_filter]) )
                based -= RMAX_SQUARE_4
            based[distance_filter] = cp.sqrt(based[distance_filter])

            close_filter = cp.logical_and((based < 2*dt) , s==o)
            far_filter = cp.logical_not(close_filter)
            close_filter = cp.logical_and(close_filter, distance_filter)
            far_filter = cp.logical_and(far_filter, distance_filter)

            tmp_d = cp.full((node_num, 2), cp.inf)
            for forward in [1, -1]:
                # handle close case
                delta = wrap360(target_node.theta-datas[close_filter,2]) if (s, forward) in [(1,1),(0,-1)] else wrap360(datas[close_filter,2]-target_node.theta)
                tmp_d[close_filter, (forward+1)//2] = delta

                # handle basic case
                (t_from, t_to) = (datas[far_filter, 2], target_node.theta) if forward == 1 else (target_node.theta, datas[far_filter, 2])
                (orientation_from, orientation_to) = (s, o) if forward == 1 else (o, s)
                tc = theta_cc[far_filter]
                if forward==-1:
                    tc[theta_cc[far_filter]<0] += np.pi
                    tc[theta_cc[far_filter]>0] -= np.pi
                if s-o!=0:
                    if orientation_from==0: tc -= dtheta[far_filter]
                    else: tc += dtheta[far_filter]

                dt1 = wrap360(tc-t_from) if orientation_from==1 else wrap360(t_from-tc)
                dt2 = wrap360(t_to-tc) if orientation_to==1 else wrap360(tc-t_to)

                tmp_d[far_filter, (forward+1)//2] = dt1+dt2

            glob_distance = cp.minimum(glob_distance, based + RMAX * cp.minimum(tmp_d[:,0], tmp_d[:,1]))
    return int(cp.argmin(glob_distance))



# distance(None)
# distance(None)

