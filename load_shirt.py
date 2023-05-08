import os
import torch 
import imageio
import numpy as np 
import json 
import cv2
from scipy.spatial.transform import Rotation

from load_blender import pose_spherical

def pose_from_rq(pos, quat):
    '''
        TODO: Verify. May need to transpose rotation
        Gets the camera-to-world matrix from the position and quaternion.
        Allegedly: 
        https://stackoverflow.com/questions/695043/how-does-one-convert-world-coordinates-to-camera-coordinates
    '''
    # Change from scalar first to scalar last quaternion
    rot = np.roll(quat, -1)

    # Rotation matrix
    rot = Rotation.from_quat(quat).as_matrix()

    # Pose
    pose = np.vstack([np.hstack([rot, np.expand_dims(pos, axis=1)]), np.array([0,0,0,1])])

    distance = np.linalg.norm(pos)

    return pose, distance


def load_shirt_data(datadir, half_res=False):
    # Load in all needed images and the relevant information
    # We will want 150 images in train, 20 images in test, & 20 images in validate
    # These will all be sequential images within the same time frame
    # So, let's collect the first 190 images every 5 images

    # We'll need images, poses and times
    imgs = []
    poses = []

    # Load in appropiate roe$NUM$.json
    # TODO: For now, assume roe2
    with open(os.path.join(datadir, 'roe2.json'), 'r') as f:
        meta = json.load(f)

    # Get data from JSON
    near = np.inf
    far = 0
    for i, frame in enumerate(meta):
        # Only consider every 5 images and up to 190 images
        if i % 5:
            continue
        elif i >= 190*5:
            break

        # Get image
        # TODO: Include setting for choosing which image set to use
        # TODO: Is SHIRT grayscale??? Converting to RGB for now
        image_fname = os.path.join(datadir, "synthetic","images", frame["filename"])
        gray = imageio.imread(image_fname)
        rgb  = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.
        imgs.append(rgb)

        # Get pose
        pose, distance = pose_from_rq(frame["r_Vo2To_vbs_true"], frame["q_vbs2tango_true"])
        poses.append(np.array(pose))

        if near > distance:
            near = distance
        elif far < distance:
            far = distance

    # Re-organize data
    # imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
    imgs = np.array(imgs)
    
    # Camera parameters
    H, W = imgs[0].shape[:2]
    focal = 3020.0062181662161
    # focal = .5 * W / np.tan(.5 * focal)

    # TODO: Check Render locations
    if os.path.exists(os.path.join(datadir, 'transforms_{}.json'.format('render'))):
        with open(os.path.join(datadir, 'transforms_{}.json'.format('render')), 'r') as fp:
            meta = json.load(fp)
        render_poses = []
        for frame in meta['frames']:
            render_poses.append(np.array(frame['transform_matrix']))
        render_poses = np.array(render_poses).astype(np.float32)
    else:
        render_poses = torch.stack([pose_spherical(angle, -30.0, 6.0) for angle in np.linspace(-90,90,40+1)[:-1]], 0)
    render_times = torch.linspace(0., 1., render_poses.shape[0])

    # Test of the render_poses
    # render_poses = np.repeat(poses[0][:, :, np.newaxis], 20, axis=2)
    # render_times = torch.linspace(0., 1., render_poses.shape[0])

    # Half res work
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
    
    # Build i_split
    all_idxs = np.arange(0, 190, 1)

    val_idxs  = np.sort(np.random.choice(all_idxs, 20, replace=False))
    all_idxs  = [i for i in all_idxs if i not in val_idxs]
    test_idxs = np.sort(np.random.choice(all_idxs, 20, replace=False))
    train_idxs = np.sort(np.array([i for i in all_idxs if i not in test_idxs]))

    i_split = [train_idxs, val_idxs, test_idxs]

    # Build normalized times
    train_times = np.linspace(0,1,150)
    val_times   = np.linspace(0,1,20)
    test_times  = np.linspace(0,1,20)

    times = np.concatenate([train_times, val_times, test_times])
    times = np.array(times).astype(np.float32)

    assert times[0] == 0, "Time must start at 0"

    return imgs, poses, times, render_poses, render_times, [H, W, focal], i_split, near, far
