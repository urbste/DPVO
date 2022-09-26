import numpy as np
from scipy.spatial.transform import Rotation as R

def norm_vec(vec):
    return vec / np.linalg.norm(vec)

# find rotation to world frame for first trajectory
def rot_between_vectors(a,b):
    # rotates a -> b
    def skew(vector):
        return np.array([[0, -vector[2], vector[1]], 
                        [vector[2], 0, -vector[0]], 
                        [-vector[1], vector[0], 0]])

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a,b)
    c = np.dot(a,b)
    s = np.linalg.norm(v)

    R = np.eye(3) + skew(v) + np.linalg.matrix_power(skew(v),2)*((1-c)/s**2)

    return R

def get_rot_to_worldframe(gravity_vecs, q_w_c, world_vec=np.array([0,0,-1])):
    mean_vec = []
    for i in range(gravity_vecs.shape[0]):
        R_cam_to_world = rot_between_vectors(gravity_vecs[i], world_vec)
        Rij = R_cam_to_world @ R.from_quat(q_w_c[i,:]).as_matrix().T
        mean_vec.append(R.from_matrix(Rij).as_rotvec())

    return R.from_rotvec(np.median(mean_vec,0)).as_matrix()

def get_heading_rot(position_data):
    traj_direction = position_data[-1]- position_data[0]
    traj_direction[2] = 0.0
    traj_direction /= np.linalg.norm(traj_direction)

    angle_to_y_axis = np.arcsin(traj_direction[0]/np.sqrt(
        traj_direction[1]**2 + traj_direction[0]**2))
    return R.from_rotvec([0,0,angle_to_y_axis]).as_matrix()

def get_vis_scaler(vis_pos, gps_pos):
    # scale trajectory with gps first and last pose
    d_vis = np.linalg.norm(vis_pos[-1]- vis_pos[0])
    d_gps = np.linalg.norm(gps_pos[-1]- gps_pos[0])
    return d_gps/d_vis


def get_heading_angle_diff(norm_vis, norm_gps):
    # angle between vectors aroun z-axis
    dir_gps = norm_vec(norm_gps[-1] - norm_gps[0])
    dir_vis = norm_vec(norm_vis[-1] - norm_vis[0])
    return np.arccos(np.dot(dir_gps[:2],dir_vis[:2]))
