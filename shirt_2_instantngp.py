import os
import imageio
import numpy as np 
import json 
import cv2
from scipy.spatial.transform import Rotation

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm

def qvec2rotmat(qvec):
    # return np.array([
    #     [
    #         2 * qvec[0]**2 + 2 * qvec[1]**2 - 1,
    #         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
    #         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
    #     ], [
    #         2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
    #         2 * qvec[0]**2 + 2 * qvec[2]**2 - 1,
    #         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
    #     ], [
    #         2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
    #         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
    #         2 * qvec[0]**2 + 2 * qvec[3]**2 - 1
    #     ]
    # ])
    return np.array([
        [
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ]
    ])

def qvec2rotmat2(qvec):
    # Change from scalar first to scalar last quaternion
    quat = np.roll(qvec, -1)

    # Rotation matrix
    rot = Rotation.from_quat(quat).as_matrix()

    return rot

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

def get_camera_intrinsics(filepath, aabb_scale):
    '''
        Gets camera information from JSON
        Transforms to instant NGP version
    '''
    with open(filepath, 'r') as f:
        shirt_camera = json.load(f)
    
    # From data
    h = shirt_camera["Nu"]
    w = shirt_camera["Nv"]

    cx = shirt_camera["ccx"]
    cy = shirt_camera["ccy"]

    # From camera matrix
    fl_x = shirt_camera["cameraMatrix"][0][0]
    fl_y = shirt_camera["cameraMatrix"][1][1]

    # From math
    angle_x = np.arctan(w / (fl_x * 2)) * 2
    angle_y = np.arctan(h / (fl_y * 2)) * 2

    # Pinhole camera
    camera = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "is_fisheye": False,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "aabb_scale": aabb_scale,
    }

    return camera

def get_frame_transforms(roe_json_fp, metadata_json_fp, img_dir):
    # Load JSON
    with open(roe_json_fp, 'r') as f:
        roe = json.load(f)
    
    with open(metadata_json_fp, "r") as f:
        meta = json.load(f)

    # Get rotation & translation from servicer principal to camera axes
    r_spri2cam_spri = meta["pMdl"]["x1"]["r_pri2cam_pri"]
    q_spri2cam     = meta["pMdl"]["x1"]["q_pri2cam"]
    # Note q_pri2cam = [1,0,0,0]
    R_spri2cam     = qvec2rotmat(q_spri2cam)
    DCM_spri2cam   = R_spri2cam.T

    # Get data for each frame
    frames = []
    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    up = np.zeros(3)
    use_colmap_coords = True

    for idx, (frame, rv, qvec) in enumerate(zip(roe, meta["tRelState"]["rv_scom2tcom_spri"], meta["tRelState"]["q_spri2tpri"])):
        # Only take first 190 images
        if idx > 190:
            break

        # Image sharpness
        img_fname = os.path.join(img_dir, frame["filename"])
        sharp = sharpness(img_fname)

        # Target pose in servicer principal frame
        r_scom2tcom_spri = np.array(rv[0:3])
        q_spri2tpri = np.array(qvec)
        R_spri2tpri = qvec2rotmat(q_spri2tpri)
        DCM_spri2tpri = R_spri2tpri.T

        # Target pose in servicer camera frame
        r_cam2tcom_cam = DCM_spri2cam@(r_scom2tcom_spri - r_spri2cam_spri)
        # R_cam2tpri = R_spri2cam.T@R_spri2tpri
        # Numerically confirmed R_cam2tpri = DCM_cam2tpri.T
        R_cam2tpri = R_spri2tpri@R_spri2cam.T
        DCM_cam2tpri = DCM_spri2tpri@DCM_spri2cam.T

        # World to camera matrix
        tvec = r_cam2tcom_cam.reshape([3,1])
        m = np.concatenate([np.concatenate([R_cam2tpri, tvec], 1), bottom], 0)
        # m = np.concatenate([np.concatenate([R_cam2tpri.T, tvec], 1), bottom], 0)

        # Get camera to world matrix
        c2w = np.linalg.inv(m)
        if not use_colmap_coords:
            c2w[0:3,2] *= -1 # flip the y and z axis
            c2w[0:3,1] *= -1
            c2w = c2w[[1,0,2,3],:]
            c2w[2,:] *= -1 # flip whole world upside down

            up += c2w[0:3,1]
        
        # Append results
        frames.append({
            "file_path": os.path.join("images/", frame["filename"]),
            "sharpness": sharp,
            "transform_matrix": c2w
        })
    
    # Stolen from intant-ngp/colmap2nerf.py
    nframes = len(frames)
    if use_colmap_coords:
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        for f in frames:
            f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat) # flip cameras (it just works)
    else:
        # don't keep colmap coords - reorient the scene to be easier to work with

        up = up / np.linalg.norm(up)
        print("up vector was", up)
        R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
        R = np.pad(R,[0,1])
        R[-1, -1] = 1

        for f in frames:
            f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

        # find a central point they are all looking at
        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in frames:
            mf = f["transform_matrix"][0:3,:]
            for g in frames:
                mg = g["transform_matrix"][0:3,:]
                p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                if w > 0.00001:
                    totp += p*w
                    totw += w
        if totw > 0.0:
            totp /= totw
        print(totp) # the cameras are looking at totp
        for f in frames:
            f["transform_matrix"][0:3,3] -= totp

        avglen = 0.
        for f in frames:
            avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)
        for f in frames:
            f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

    # To list so they can be stored as a JSON
    for f in frames:
        f["transform_matrix"] = f["transform_matrix"].tolist()

    return frames

def get_frame_transforms2(roe_json_fp, metadata_json_fp, img_dir):
    # Load JSON
    with open(roe_json_fp, 'r') as f:
        roe = json.load(f)
    
    with open(metadata_json_fp, "r") as f:
        meta = json.load(f)

    # Get data for each frame
    frames = []
    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    up = np.zeros(3)
    use_colmap_coords = False

    for idx, (frame, rv, qvec) in enumerate(zip(roe, meta["tRelState"]["rv_scom2tcom_spri"], meta["tRelState"]["q_spri2tpri"])):
        # Only take first 150 images
        if idx > 150:
            break

        # Image sharpness
        img_fname = os.path.join(img_dir, frame["filename"])
        sharp = sharpness(img_fname)
        
        # Tango position in servicer camera frame
        tvec = np.array(frame["r_Vo2To_vbs_true"])

        # Orientation of tango wrt camera in camera frame
        qvec = np.array(frame["q_vbs2tango_true"])
        R    = qvec2rotmat2(qvec)

        # COLMAP: World to camera
        R    = np.transpose(R)

        # 
        tvec = tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, tvec], 1), bottom], 0)

        ################# FROM INSTANT-NGP CODE NOW #################
        # Get camera to world matrix
        c2w = np.linalg.inv(m)
        if not use_colmap_coords:
            c2w[0:3,2] *= -1 # flip the y and z axis
            c2w[0:3,1] *= -1
            c2w = c2w[[1,0,2,3],:]
            c2w[2,:] *= -1 # flip whole world upside down

            up += c2w[0:3,1]
        
        # Append results
        frames.append({
            "file_path": os.path.join("images/", frame["filename"]),
            "sharpness": sharp,
            "transform_matrix": c2w
        })
    
    nframes = len(frames)
    if use_colmap_coords:
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        for f in frames:
            f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat) # flip cameras (it just works)
    else:
        # don't keep colmap coords - reorient the scene to be easier to work with

        up = up / np.linalg.norm(up)
        print("up vector was", up)
        R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
        R = np.pad(R,[0,1])
        R[-1, -1] = 1

        for f in frames:
            f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

        # find a central point they are all looking at
        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in frames:
            mf = f["transform_matrix"][0:3,:]
            for g in frames:
                mg = g["transform_matrix"][0:3,:]
                p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                if w > 0.00001:
                    totp += p*w
                    totw += w
        if totw > 0.0:
            totp /= totw
        print(totp) # the cameras are looking at totp
        for f in frames:
            f["transform_matrix"][0:3,3] -= totp

        avglen = 0.
        for f in frames:
            avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)
        for f in frames:
            f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

    # To list so they can be stored as a JSON
    for f in frames:
        f["transform_matrix"] = f["transform_matrix"].tolist()

    return frames

def get_frame_transforms_eci(roe_json_fp, metadata_json_fp, img_dir):
    # Load JSON
    with open(roe_json_fp, 'r') as f:
        roe = json.load(f)
    
    with open(metadata_json_fp, "r") as f:
        meta = json.load(f)

    # Get rotation & translation from servicer principal to camera axes
    r_spri2cam_spri = meta["pMdl"]["x1"]["r_pri2cam_pri"]
    q_spri2cam     = meta["pMdl"]["x1"]["q_pri2cam"]
    R_spri2cam     = qvec2rotmat(q_spri2cam)

    # Get data for each frame
    frames = []
    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    up = np.zeros(3)
    use_colmap_coords = False

    for idx, (frame, rv, qvec) in enumerate(zip(roe, meta["sAbsState"]["rv_eci2com_eci"], meta["sAbsState"]["q_eci2pri"])):
        # Only take first 190 images
        if idx > 180:
            break

        # Image sharpness
        img_fname = os.path.join(img_dir, frame["filename"])
        sharp = sharpness(img_fname)

        # Servicer pose in ECI frame
        r_eci2scom_eci = np.array(rv[0:3])
        q_eci2spri = np.array(qvec)
        R_eci2spri = qvec2rotmat(q_eci2spri)

        # Camera pose in ECI frame
        r_eci2cam_eci = r_eci2scom_eci - R_eci2spri.T@r_spri2cam_spri
        R_cam2eci     = (R_eci2spri@R_spri2cam).T

        # ECI to camera matrix
        tvec = -r_eci2cam_eci.reshape([3,1])
        m = np.concatenate([np.concatenate([R_cam2eci.T, tvec], 1), bottom], 0)

        # Get camera to world matrix
        c2w = np.linalg.inv(m)
        if not use_colmap_coords:
            c2w[0:3,2] *= -1 # flip the y and z axis
            c2w[0:3,1] *= -1
            c2w = c2w[[1,0,2,3],:]
            c2w[2,:] *= -1 # flip whole world upside down

            up += c2w[0:3,1]
        
        # Append results
        frames.append({
            "file_path": os.path.join("images/", frame["filename"]),
            "sharpness": sharp,
            "transform_matrix": c2w
        })
    
    # Stolen from intant-ngp/colmap2nerf.py
    nframes = len(frames)
    if use_colmap_coords:
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        for f in frames:
            f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat) # flip cameras (it just works)
    else:
        # don't keep colmap coords - reorient the scene to be easier to work with

        up = up / np.linalg.norm(up)
        print("up vector was", up)
        R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
        R = np.pad(R,[0,1])
        R[-1, -1] = 1

        for f in frames:
            f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

        # find a central point they are all looking at
        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in frames:
            mf = f["transform_matrix"][0:3,:]
            for g in frames:
                mg = g["transform_matrix"][0:3,:]
                p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                if w > 0.00001:
                    totp += p*w
                    totw += w
        if totw > 0.0:
            totp /= totw
        print(totp) # the cameras are looking at totp
        for f in frames:
            f["transform_matrix"][0:3,3] -= totp

        avglen = 0.
        for f in frames:
            avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)
        for f in frames:
            f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

    # To list so they can be stored as a JSON
    for f in frames:
        f["transform_matrix"] = f["transform_matrix"].tolist()

    return frames

def shirt_json_2_instant_ngp_json(camera_json_fp, metadata_json_fp, roe_json_fp, img_dir, aabb_scale):
    '''
        Gets the relevant JSON's and transforms then to the
        format expect by intant NGP
    '''
    camera_json = get_camera_intrinsics(camera_json_fp, aabb_scale)
    frames      = get_frame_transforms(roe_json_fp, metadata_json_fp, img_dir)
    # frames = get_frame_transforms_eci(roe_json_fp, metadata_json_fp, img_dir)

    out = camera_json
    out["frames"] = frames

    return out


if __name__ == "__main__":
    aabb_scale = 4

    camera_json_fp = '/home/pol/Documents/Stanford/stanford/AA290/D-NeRF/data/shirtv1/camera.json'
    metadata_json_fp = '/home/pol/Documents/Stanford/stanford/AA290/D-NeRF/data/shirtv1/roe1/metadata.json'
    roe_json_fp = '/home/pol/Documents/Stanford/stanford/AA290/D-NeRF/data/shirtv1/roe1/roe1.json'
    img_dir     = '/home/pol/Documents/Stanford/stanford/AA290/D-NeRF/data/shirtv1/roe1/synthetic/images'
    save_fp = '/home/pol/Documents/Stanford/stanford/AA290/D-NeRF/data/shirtv1/roe1/synthetic/transforms.json'

    transforms = shirt_json_2_instant_ngp_json(camera_json_fp, metadata_json_fp, roe_json_fp, img_dir, aabb_scale)

    with open(save_fp, "w") as f:
        json.dump(transforms, f, indent=2)