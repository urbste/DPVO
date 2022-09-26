

import numpy as np
import cv2
import os
import pickle
from scipy.spatial.transform import Rotation as R
import pyvisfm as pvi
import math
import natsort
import numpy as np
import cv2

import Ogre
import Ogre.Bites
import Ogre.RTShader

from utils import load_dataset

import time

def get_cam_pose_from_spline_at_time(spline, time_ns):
    pose_vec = spline.GetCameraPose(int(time_ns))
    R_w_c = R.from_quat([pose_vec[0],pose_vec[1],pose_vec[2],pose_vec[3]]).as_matrix()
    p_w_c = np.expand_dims(pose_vec[4:],1)
    R_c_w = R_w_c.T
    t_c_w = -R_c_w@p_w_c

    return R_c_w, t_c_w, p_w_c

def set_camera_intrinsics(cam, K, imsize):
    cam.setAspectRatio(imsize[0]/imsize[1])

    zNear = cam.getNearClipDistance()
    top = zNear * K[1, 2] / K[1, 1]
    left = -zNear * K[0, 2] / K[0, 0]
    right = zNear * (imsize[0] - K[0, 2]) / K[0, 0]
    bottom = -zNear * (imsize[1] - K[1, 2]) / K[1, 1]

    cam.setFrustumExtents(left, right, top, bottom)

    fovy = math.atan2(K[1, 2], K[1, 1]) + math.atan2(imsize[1] - K[1, 2], K[1, 1])
    cam.setFOVy(fovy)

def create_traj_line(name, mat_name, scn_mgr, pos, color):

    line_obj = scn_mgr.createManualObject(name)
    line_node = scn_mgr.getRootSceneNode().createChildSceneNode()
    line_mat = Ogre.MaterialManager.getSingleton().create(mat_name, "General")
    line_mat.setReceiveShadows(False)
    line_mat.getTechnique(0).getPass(0).setDiffuse(0,0,1,0)
    line_mat.getTechnique(0).getPass(0).setAmbient(0.5,0.5,0.5)
    line_mat.getTechnique(0).getPass(0).setSelfIllumination(color[0],color[1],color[2])
    line_mat.getTechnique(0).getPass(0).setPointSize(5.)
    line_mat.getTechnique(0).getPass(0).setLineWidth(10.)

    line_obj.begin(mat_name, Ogre.RenderOperation.OT_POINT_LIST)
    for i in range(pos.shape[0]):
       line_obj.position(pos[i,:])
    line_obj.end()

    line_node.attachObject(line_obj)
    return line_node

def create_image_background(scn_mgr):
    tex = Ogre.TextureManager.getSingleton().create("bgtex", Ogre.RGN_DEFAULT, True)
    tex.setNumMipmaps(0)

    mat = Ogre.MaterialManager.getSingleton().create("bgmat", Ogre.RGN_DEFAULT)
    mat.getTechnique(0).getPass(0).createTextureUnitState().setTexture(tex)
    mat.getTechnique(0).getPass(0).setDepthWriteEnabled(False)
    mat.getTechnique(0).getPass(0).setLightingEnabled(False)

    rect = scn_mgr.createScreenSpaceRect(True)
    rect.setMaterial(mat)
    rect.setRenderQueueGroup(Ogre.RENDER_QUEUE_BACKGROUND)
    scn_mgr.getRootSceneNode().attachObject(rect)

    return tex

def main(ctx):

    base_path = "/media/Data/Sparsenet/Ammerbach/Links"
    path_traj1 = os.path.join(base_path,"run1")
    path_traj2 = os.path.join(base_path,"run2")

    imsize = (640, 480)

    K = np.diag([297.4347120333558, 297.4347120333558, 1.]) 
    K[0,2] =  323.609635667
    K[1,2] =  237.52771880186
    K = cv2.getDefaultNewCameraMatrix(K, imsize, True)

    #a_file = open(os.path.join(base_path,"aligned_trajectories.dict"), "rb")
    #traj = pickle.load(a_file)
    #a_file.close()

    
    spline = pvi.SplineTrajectoryEstimator()
    pvi.ReadSpline(spline, 
        os.path.join(base_path,"spline_recon_run1.spline"))
    # generate image poses
    image_folder = os.listdir(os.path.join(base_path,"run1"))
    timestamps1 = natsort.natsorted(
        [os.path.splitext(os.path.basename(p))[0] for p in image_folder])
    p1 = []
    q1 = []
    for t_ns in timestamps1:
        R_w_c, _, p_w_c = get_cam_pose_from_spline_at_time(spline, t_ns)
        p1.append(p_w_c.squeeze())
        q1.append(R.from_matrix(R_w_c.T).as_quat())
    q1 = np.array(q1)
    p1 = np.array(p1)

    cfg = Ogre.ConfigFile()
    cfg.loadDirect("ogre_resources.cfg")
    rgm = Ogre.ResourceGroupManager.getSingleton()

    for sec, settings in cfg.getSettingsBySection().items():
        for kind, loc in settings.items():
            rgm.addResourceLocation(loc, kind, sec)
    rgm.initialiseAllResourceGroups()

    ## setup Ogre for AR
    scn_mgr = ctx.getRoot().createSceneManager()
    Ogre.RTShader.ShaderGenerator.getSingleton().addSceneManager(scn_mgr)

    cam= scn_mgr.createCamera("camera1")
    cam.setNearClipDistance(0.01)
    cam.setFarClipDistance(4.0)
    win = ctx.getRenderWindow().addViewport(cam)

    line1_node = create_traj_line("line1", "linemat1", scn_mgr, p1+np.array([0,0,0.5]), [1,0,0])
    line1_node.setVisible(True)

    #line2_node = create_traj_line("line2", "linemat2", scn_mgr, p2+np.array([0,0,-1.25]), [0,1,0])
    #line2_node.setVisible(True)

    camnode = scn_mgr.getRootSceneNode().createChildSceneNode()

    # convert OpenCV to OGRE coordinate system
    # camnode.rotate((1, 0, 0), math.pi)
    camnode.attachObject(cam)

    set_camera_intrinsics(cam, K, imsize)

    bgtex = create_image_background(scn_mgr)

    ## setup 3D scene
    scn_mgr.setAmbientLight((.1, .1, .1))
    scn_mgr.getRootSceneNode().createChildSceneNode().attachObject(scn_mgr.createLight())

    marker_node = scn_mgr.getRootSceneNode().createChildSceneNode()
    mesh_node = marker_node.createChildSceneNode()
    mesh_node.attachObject(scn_mgr.createEntity("Sinbad.mesh"))
    mesh_node.rotate((1, 0, 0), math.pi/2)
    mesh_node.translate((0, 0, 0))
    mesh_node.setVisible(False)
    mesh_node.scale(np.array([0.5,0.5,0.5]))

    for i in range(len(timestamps1)):
        img = cv2.imread(os.path.join(path_traj1,str(timestamps1[i])+".png"))
        T_w_c = np.eye(4)
        T_w_c[:3,3] = p1[i,:]
        T_w_c[:3,:3] = R.from_quat(q1[i,:]).as_matrix()
        T_w_c = T_w_c @ np.diag([1,-1,-1, 1])
        rvec = R.from_matrix(T_w_c[:3,:3]).as_quat()[[3,0,1,2]]
        tvec = T_w_c[:3,3]
        
        im = Ogre.Image(Ogre.PF_BYTE_BGR, img.shape[1], img.shape[0], 1, img, False)
        if bgtex.getBuffer():
            bgtex.getBuffer().blitFromMemory(im.getPixelBox())
        else:
            bgtex.loadImage(im)

        #ax = Ogre.Vector3(*rvec)
        #ang = ax.normalise()
        #marker_node.setOrientation(Ogre.Quaternion(ang, ax))
        marker_node.setPosition(p1[1000,:]+np.array([0,2,-1.0]))
        mesh_node.setVisible(True)

        camnode.setOrientation(Ogre.Quaternion(*rvec))
        camnode.setPosition(tvec)

        ctx.getRoot().renderOneFrame()

        # time.sleep(0.1)
ctx = Ogre.Bites.ApplicationContext()
ctx.initApp()
main(ctx)
ctx.closeApp()