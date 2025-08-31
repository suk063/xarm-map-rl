"""
3D Pose estimation demo for RealSense camera
Based on https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/pyglet_pointcloud_viewer.py
"""

USAGE = ''' 
Usage:
------
Mouse:
    Drag with left button to rotate around pivot (thick small axes),
    with right button to translate and the wheel to zoom.

Keyboard:
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [x]     Toggle point distance attenuation
    [c]     Toggle color source
    [l]     Toggle lighting
    [f]     Toggle depth post-processing
    [s]     Save PNG (./out.png)
    [e]     Export points to ply (./scene.ply)
    [o]     Export object pointcloud to ply (./object_realtime_color.ply or ./object_realtime_depthcolor.ply)
    [m]     Model scan mode - once started, take scans using 'O' and press 'm' again when enough scans are taken
    [k]     Toggle use_color for model scan multiway registration
    [h]     Cycle between model to be used for registration [Cube / Small Cube / Can / Rabbit]
    [j]     Perform registration of last saved object scan (object_realtime_color.ply) with model
    [g]     Perform continuous registration of realtime object scan with model. On ending continuous registration, euler angles history is plotted
    [b]     Switch between best fit vs most frequent pose estimation for continuous registration
    [n]     Plot euler angles corresponding to high fitness scores after continuous registration
    [q/ESC] Quit

'''


import math
import ctypes
import pyglet
import pyglet.gl as gl
import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d
from utils.registration_utils import get_multiway_registered_pcd, draw_registration_result, perform_registration
import transforms3d as t3d
import matplotlib.pyplot as plt
from scipy.stats import mode

model_types_paths = [("cube", "/home/erl-tianyu/xArm_robosuite/printed_cube_model.ply"),
                     ("cube_xyz", "/home/erl-tianyu/xArm_robosuite/printed_cube_xyz_model.ply"),
                     ("can", "/home/erl-tianyu/xArm_robosuite/printed_can_model.ply"),
                     ("rabbit", "/home/erl-tianyu/xArm_robosuite/printed_rabbit_model.ply")]


# https://stackoverflow.com/a/6802723
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


class AppState:

    def __init__(self, *args, **kwargs):
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, 0.1], np.float32)
        self.distance = 2
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 0
        self.scale = True
        self.attenuation = False
        self.color = True
        self.lighting = False
        self.postprocessing = False
        self.model_scan_mode = False
        self.model_scan_count = 0
        self.model_scan_use_color = False
        self.continuous_registration = False
        self.model_id = 0
        self.model_pcd = None
        self.use_bestfit_estimate = True
        self.plot_only_bestfits = False

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, 1

    @property
    def rotation(self):
        Rx = rotation_matrix((1, 0, 0), math.radians(-self.pitch))
        Ry = rotation_matrix((0, 1, 0), math.radians(-self.yaw))
        return np.dot(Ry, Rx).astype(np.float32)

state = AppState()

# Configure streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
other_stream, other_format = rs.stream.color, rs.format.rgb8
config.enable_stream(other_stream, other_format, 30)

# Start streaming
pipeline.start(config)
profile = pipeline.get_active_profile()

depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()
filters = [rs.disparity_transform(),
           rs.spatial_filter(),
           rs.temporal_filter(),
           rs.disparity_transform(False)]


# pyglet
window = pyglet.window.Window(
    config=gl.Config(
        double_buffer=True,
        samples=8  # MSAA
    ),
    resizable=True, vsync=True)
keys = pyglet.window.key.KeyStateHandler()
window.push_handlers(keys)


def convert_fmt(fmt):
    """rs.format to pyglet format string"""
    return {
        rs.format.rgb8: 'RGB',
        rs.format.bgr8: 'BGR',
        rs.format.rgba8: 'RGBA',
        rs.format.bgra8: 'BGRA',
        rs.format.y8: 'L',
    }[fmt]


# Create a VertexList to hold pointcloud data
# Will pre-allocates memory according to the attributes below
vertex_list = pyglet.graphics.vertex_list(
    w * h, 'v3f/stream', 't2f/stream', 'n3f/stream')
# Create and allocate memory for our color data
other_profile = rs.video_stream_profile(profile.get_stream(other_stream))

image_w, image_h = w, h
color_intrinsics = other_profile.get_intrinsics()
color_w, color_h = color_intrinsics.width, color_intrinsics.height

if state.color:
    image_w, image_h = color_w, color_h

image_data = pyglet.image.ImageData(image_w, image_h, convert_fmt(
other_profile.format()), (gl.GLubyte * (image_w * image_h * 3))())

if (pyglet.version <  '1.4' ):
    # pyglet.clock.ClockDisplay has be removed in 1.4
    fps_display = pyglet.clock.ClockDisplay()
else:
    fps_display = pyglet.window.FPSDisplay(window)


@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    w, h = map(float, window.get_size())

    if buttons & pyglet.window.mouse.LEFT:
        state.yaw -= dx * 0.5
        state.pitch -= dy * 0.5

    if buttons & pyglet.window.mouse.RIGHT:
        dp = np.array((dx / w, -dy / h, 0), np.float32)
        state.translation += np.dot(state.rotation, dp)

    if buttons & pyglet.window.mouse.MIDDLE:
        dz = dy * 0.01
        state.translation -= (0, 0, dz)
        state.distance -= dz


def handle_mouse_btns(x, y, button, modifiers):
    state.mouse_btns[0] ^= (button & pyglet.window.mouse.LEFT)
    state.mouse_btns[1] ^= (button & pyglet.window.mouse.RIGHT)
    state.mouse_btns[2] ^= (button & pyglet.window.mouse.MIDDLE)


window.on_mouse_press = window.on_mouse_release = handle_mouse_btns


@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    dz = scroll_y * 0.1
    state.translation -= (0, 0, dz)
    state.distance -= dz


def on_key_press(symbol, modifiers):
    global angles_translations_history

    if symbol == pyglet.window.key.R:
        state.reset()

    if symbol == pyglet.window.key.P:
        state.paused ^= True

    if symbol == pyglet.window.key.D:
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

    if symbol == pyglet.window.key.C:
        state.color ^= True

    if symbol == pyglet.window.key.Z:
        state.scale ^= True

    if symbol == pyglet.window.key.X:
        state.attenuation ^= True

    if symbol == pyglet.window.key.L:
        state.lighting ^= True

    if symbol == pyglet.window.key.F:
        state.postprocessing ^= True

    if symbol == pyglet.window.key.S:
        pyglet.image.get_buffer_manager().get_color_buffer().save('out.png')

    if symbol == pyglet.window.key.Q:
        window.close()

    # Model Scanning
    if symbol == pyglet.window.key.M:
        state.model_scan_mode ^= True
        print("Model Scanning Mode ON:\nMove camera to different angles and save scans using 'O' key. Press 'M' again when done to generate multiregistered model pointcloud")
        print(f"Attempting multiregistration with{'' if state.model_scan_use_color else 'out'} color information\n")

    if symbol == pyglet.window.key.K:
        state.model_scan_use_color ^= True
        print(f"Model scan parameter: use_color = {state.model_scan_use_color}")

    if symbol == pyglet.window.key.H:
        state.model_id = (state.model_id + 1) % len(model_types_paths)
        model_type, model_path = model_types_paths[state.model_id]
        print(f"\nUsing model file for - {model_type}")
        state.model_pcd = o3d.io.read_point_cloud(model_path)

    if symbol == pyglet.window.key.J:
        if state.model_pcd is None:
            model_type, model_path = model_types_paths[state.model_id]
            print(f"\nUsing model file for - {model_type}")
            state.model_pcd = o3d.io.read_point_cloud(model_path)
        perform_registration(model_pcd=state.model_pcd)

    if symbol == pyglet.window.key.B:
        state.use_bestfit_estimate ^= True
        if state.use_bestfit_estimate:
            print(f"\nUsing best fitness score estimate for continuous registration")
        else:
            print(f"\nUsing most frequent euler angles for continuous registration")

    # Continuous Registration
    if symbol == pyglet.window.key.G:
        state.continuous_registration ^= True
        if not state.continuous_registration:
            # End of continuous registration
            euler_plots = np.array([ang_trans for ang_trans, _ in angles_translations_history])
            fitness_scores = np.array([fitness for _, fitness in angles_translations_history])
            angles_translations_history = []

            object_pcd = save_scan_pointcloud(color_image, depth_frame_aligned, depth_intrinsics, color_intrinsics, color_source, fast_mode=True)
            if not state.use_bestfit_estimate:
                # Use mode of the euler angle estimates
                nbins = 18
                plot_pose_angles_histograms(euler_plots, nbins=nbins)

                hist, edges = np.histogramdd(euler_plots[:, :3], bins=nbins, range=[[-180, 180], [-90, 90], [-180, 180]])
                r_bin, p_bin, y_bin = np.unravel_index(np.argmax(hist), hist.shape)
                print(f"Max frequency: {hist.max()}, Max frequency index: {np.argmax(hist)}, indices: ({r_bin}, {p_bin}, {y_bin}), hist.shape: {hist.shape}")
                print(f"Ranges: [{edges[0][r_bin]}, {edges[0][r_bin+1]}), [{edges[1][p_bin]}, {edges[1][p_bin+1]}), [{edges[2][y_bin]}, {edges[2][y_bin+1]})")
                
                roll_angles_best = (euler_plots[:,0] >= edges[0][r_bin]) & (euler_plots[:,0] < edges[0][r_bin+1])
                roll_angles = euler_plots[roll_angles_best, 0]
                pitch_angles_best = (euler_plots[:,1] >= edges[1][p_bin]) & (euler_plots[:,1] < edges[1][p_bin+1])
                pitch_angles = euler_plots[pitch_angles_best, 1]
                yaw_angles_best = (euler_plots[:,2] >= edges[2][y_bin]) & (euler_plots[:,2] < edges[2][y_bin+1])
                yaw_angles = euler_plots[yaw_angles_best, 2]

                all_angles_best = (roll_angles_best & pitch_angles_best & yaw_angles_best)
                print(f"Mode options: {all_angles_best.shape[0]}")
                # mode_rotation = t3d.euler.euler2mat(roll_angles.mean()*np.pi/180, pitch_angles.mean()*np.pi/180, yaw_angles.mean()*np.pi/180)
                # mode_rotation = t3d.euler.euler2mat(mode(roll_angles)[0]*np.pi/180, mode(pitch_angles)[0]*np.pi/180, mode(yaw_angles)[0]*np.pi/180)
                mode_angles_deg = euler_plots[all_angles_best, :3][0]
                mode_translation = euler_plots[all_angles_best, 3:][0]
                mode_rotation = t3d.euler.euler2mat(mode_angles_deg[0]*np.pi/180, mode_angles_deg[1]*np.pi/180, mode_angles_deg[2]*np.pi/180)
                mode_transformation = np.eye(4)
                mode_transformation[:3,:3] = mode_rotation
                mode_transformation[:3, 3] = mode_translation
                mode_fitness = fitness_scores[all_angles_best][0]

                print(f"\nPerforming registration with the most frequent euler angles: {mode_angles_deg[0]}, {mode_angles_deg[1]}, {mode_angles_deg[2]}")
                print(f"Translation: {mode_translation}")
                print(f"Fitness score: {mode_fitness}")
                # print(f"Transformation matrix: \n{mode_transformation}")
                draw_registration_result(object_pcd, state.model_pcd, mode_transformation, window_name="Pointclouds registered with most frequent euler angles")
            else:
                # Using euler angles with maximum fitness score
                bestfit_index = np.argmax(fitness_scores)
                best_fitness = fitness_scores[bestfit_index]
                bestfit_angles_deg = euler_plots[bestfit_index, :3]
                bestfit_translation = euler_plots[bestfit_index, 3:]
                bestfit_rotation = t3d.euler.euler2mat(bestfit_angles_deg[0]*np.pi/180, bestfit_angles_deg[1]*np.pi/180, bestfit_angles_deg[2]*np.pi/180)
                bestfit_transformation = np.eye(4)
                bestfit_transformation[:3,:3] = bestfit_rotation
                bestfit_transformation[:3, 3] = bestfit_translation
                print(f"\nPerforming registration with the euler angles with highest fitness score")
                print(f"Euler Angles (degrees): {bestfit_angles_deg[0]}, {bestfit_angles_deg[1]}, {bestfit_angles_deg[2]}")
                print(f"Translation: {bestfit_translation}")
                print(f"Fitness score: {best_fitness}")
                # print(f"Transformation matrix: \n{bestfit_transformation}")
                draw_registration_result(object_pcd, state.model_pcd, bestfit_transformation, window_name=f"Pointclouds registered with euler angles corresponding to highest fitness score {best_fitness}")

            if state.plot_only_bestfits:
                best_fitness = fitness_scores.max()
                euler_plots_best = (fitness_scores >= best_fitness-0.02)
                euler_plots = euler_plots[euler_plots_best]
                fitness_scores = fitness_scores[euler_plots_best]

                print(f"Number of estimates with fitness score >= {best_fitness-0.02}: {euler_plots.shape[0]}")
                dpoints = euler_plots.shape[0]
                plt.figure(f"Euler angles for high fitness scores (>= {best_fitness-0.02})")
                x = np.arange(dpoints)
                plt.plot(x, euler_plots[:,0], label="roll")
                plt.plot(x, euler_plots[:,1], label="pitch")
                plt.plot(x, euler_plots[:,2], label="yaw")
                plt.legend()
                plt.xlabel("Registration run index")
                plt.ylabel("Angle (degrees)")
                plt.show()

            print(f"\nOne-shot registration with most recent object scan")
            oneshot_angles_trans_deg, oneshot_fitness = perform_registration(object_pcd, model_pcd=state.model_pcd, fast_mode=False)

        elif state.model_pcd is None:
            # Start of continuous registration
            model_type, model_path = model_types_paths[state.model_id]
            print(f"\nUsing model file for - {model_type}")
            state.model_pcd = o3d.io.read_point_cloud(model_path)
    
    # Continuous Registration - Plot angles corresponding to high fitness scores
    if symbol == pyglet.window.key.N:
        state.plot_only_bestfits ^= True
        print(f"\nToggling plots for high fitness euler angles after continuous registration: {state.plot_only_bestfits}")
        
window.push_handlers(on_key_press)


def axes(size=1, width=1):
    """draw 3d axes"""
    gl.glLineWidth(width)
    pyglet.graphics.draw(6, gl.GL_LINES,
                         ('v3f', (0, 0, 0, size, 0, 0,
                                  0, 0, 0, 0, size, 0,
                                  0, 0, 0, 0, 0, size)),
                         ('c3f', (1, 0, 0, 1, 0, 0,
                                  0, 1, 0, 0, 1, 0,
                                  0, 0, 1, 0, 0, 1,
                                  ))
                         )


def frustum(intrinsics):
    """draw camera's frustum"""
    w, h = intrinsics.width, intrinsics.height
    batch = pyglet.graphics.Batch()

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            batch.add(2, gl.GL_LINES, None, ('v3f', [0, 0, 0] + p))
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        batch.add(2, gl.GL_LINES, None, ('v3f', top_left + top_right))
        batch.add(2, gl.GL_LINES, None, ('v3f', top_right + bottom_right))
        batch.add(2, gl.GL_LINES, None, ('v3f', bottom_right + bottom_left))
        batch.add(2, gl.GL_LINES, None, ('v3f', bottom_left + top_left))

    batch.draw()


def grid(size=1, n=10, width=1):
    """draw a grid on xz plane"""
    gl.glLineWidth(width)
    s = size / float(n)
    s2 = 0.5 * size
    batch = pyglet.graphics.Batch()

    for i in range(0, n + 1):
        x = -s2 + i * s
        batch.add(2, gl.GL_LINES, None, ('v3f', (x, 0, -s2, x, 0, s2)))
    for i in range(0, n + 1):
        z = -s2 + i * s
        batch.add(2, gl.GL_LINES, None, ('v3f', (-s2, 0, z, s2, 0, z)))

    batch.draw()


LOWER_GREEN = np.array([30, 80, 40], dtype=np.uint8)
UPPER_GREEN = np.array([90, 255, 255], dtype=np.uint8)
def get_bbox_2d(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv, LOWER_RED, UPPER_RED)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    # mask = cv2.inRange(hsv, LOWER_BLACK, UPPER_BLACK)
    # res = cv2.bitwise_and(frame, frame, mask=mask)    

    # Find contours of the largest red object
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours, key = cv2.contourArea)
        return cv2.boundingRect(c)
    else:
        return None


def bbox_3d(color_image, depth_frame, depth_intrin, color_intrin):
    bbox_2d = get_bbox_2d(color_image)
    if bbox_2d is None:
        return

    s1, s2 = color_image.shape[0] / depth_intrin.height, color_image.shape[1] / depth_intrin.width
    u, v = int((bbox_2d[0] + bbox_2d[2] / 2)), int((bbox_2d[1] + bbox_2d[3] / 2))
    dist = depth_frame.get_distance(int(u / s1), int(v / s2))  

    x, y, z = rs.rs2_deproject_pixel_to_point(color_intrin, [int(bbox_2d[0]), int(bbox_2d[1])], dist)
    x2, y2, _ = rs.rs2_deproject_pixel_to_point(color_intrin, [int(bbox_2d[0] + bbox_2d[2]), int(bbox_2d[1] + bbox_2d[3])], dist)
    w, h = x2-x, y2-y
    d = max(w,h)

    vertices = [
        # Front face
        x, y, z,
        x + w, y, z,
        x + w, y + h, z,
        x, y + h, z,

        # Back face
        x, y, z + d,
        x + w, y, z + d,
        x + w, y + h, z + d,
        x, y + h, z + d,
    ]

    indices = [
        # Front face
        0, 1, 1, 2, 2, 3, 3, 0,

        # Back face
        4, 5, 5, 6, 6, 7, 7, 4,

        # Connect front and back faces
        0, 4, 1, 5, 2, 6, 3, 7,
    ]

    colors = [255, 0, 0, 255]  # Red
    batch = pyglet.graphics.Batch()
    batch.add_indexed(8, pyglet.gl.GL_LINES, None, indices, ('v3f', vertices), ('c4B', colors*8))
    batch.draw()


def plot_pose_angles_histograms(euler_plots, nbins=18):
    dpoints = euler_plots.shape[0]
    plt.figure("Euler angles")
    x = np.arange(dpoints)
    plt.plot(x, euler_plots[:,0], label="roll")
    plt.plot(x, euler_plots[:,1], label="pitch")
    plt.plot(x, euler_plots[:,2], label="yaw")
    plt.legend()
    plt.xlabel("Registration run index")
    plt.ylabel("Angle (degrees)")
    
    plt.figure("Histograms - Roll")
    plt.hist(euler_plots[:,0], color='r', bins=nbins)
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Frequency")

    plt.figure("Histograms - Pitch")
    plt.hist(euler_plots[:,1], color='g', bins=nbins)
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Frequency")

    plt.figure("Histograms - Yaw")
    plt.hist(euler_plots[:,2], color='b', bins=nbins)
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Frequency")
    plt.show()
    
    ########### 3D histograms
    ## ROLL-PITCH
    fig = plt.figure("3D Histogram - RP")
    ax = fig.add_subplot(projection='3d')
    hist, xedges, yedges = np.histogram2d(euler_plots[:, 0], euler_plots[:, 1], bins=nbins, range=[[0, 90], [0, 90]])
    
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

    ## PITCH-YAW
    fig = plt.figure("3D Histogram - PY")
    ax = fig.add_subplot(projection='3d')
    hist, xedges, yedges = np.histogram2d(euler_plots[:, 1], euler_plots[:, 2], bins=nbins, range=[[0, 90], [0, 90]])
    
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    plt.show()
    ############


def save_scan_pointcloud(color_image, depth_frame_aligned, depth_intrin, color_intrin, color_source, fast_mode=False):
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    # print(f"hsv: {hsv[0, :3]}")
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        object_contour = max(contours, key = cv2.contourArea)
    else:
        return None

    # Create a mask image that contains the object_contour filled in
    cimg = np.zeros_like(mask)
    cv2.drawContours(cimg, [object_contour], 0, color=255, thickness=-1)

    # Access the image pixels and create a 1D numpy array then add to list
    pts = np.where(cimg == 255)
    object_pixels = list(zip(pts[1].astype(int), pts[0].astype(int)))

    if not fast_mode:
        print("\nnumber of object points:", len(object_pixels))

    xyz = np.zeros([len(object_pixels), 3])
    colors = np.zeros([len(object_pixels), 3])
    for i, object_pixel in enumerate(object_pixels):
        dist = depth_frame_aligned.get_distance(object_pixel[0], object_pixel[1])  
        xyz[i] = rs.rs2_deproject_pixel_to_point(color_intrin, [object_pixel[0], object_pixel[1]], dist)
        colors[i] = color_source[object_pixel[1], object_pixel[0]]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors/255)
    transformation = np.eye(4)
    transformation[:3, :3] = t3d.euler.euler2mat(-np.pi/2, 0, -np.pi/2)
    pcd.transform(transformation)
    color_type = "color" if state.color else "depthcolor"
    if (not state.model_scan_mode) and (not fast_mode):
        o3d.io.write_point_cloud(f"./object_realtime_{color_type}.ply", pcd) 
        print(f"Scan pointcloud saved to ./object_realtime_{color_type}.ply")
        o3d.visualization.draw_geometries([pcd], window_name="Isolated Object Pointcloud", lookat=[ 0.30500000715255737, -0.099672742187976823, 0.0052387155592441871 ],\
                                                                                         up=[ -0.060332680405135584, -0.14247986667677359, 0.98795721327742769],\
                                                                                         front=[-0.95228391684816383, 0.30488379977303537, -0.014184863350600891], \
                                                                                         zoom=0.25)
    elif state.model_scan_mode:
        state.model_scan_count += 1
        o3d.io.write_point_cloud(f"./model_scan_{state.model_scan_count}.ply", pcd)
        print(f"Model scan {state.model_scan_count} saved to ./model_scan_{state.model_scan_count}.ply")
    elif fast_mode:
        return pcd


def create_model_from_scans():
    pcd_paths = []
    for i in range(state.model_scan_count):
        pcd_paths.append(f"./model_scan_{i+1}.ply")
    model = get_multiway_registered_pcd(pcd_paths, visualize=True, use_color=state.model_scan_use_color)
    o3d.io.write_point_cloud(f"./model_multiregistered.ply", model)
    print(f"Model pointcloud saved to ./model_multiregistered.ply\n")
    # o3d.visualization.draw_geometries([model], window_name="Multiregistered Model Pointcloud")    # Visualizing inside registraion method instead
    

@window.event
def on_draw():
    window.clear()

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_LINE_SMOOTH)

    width, height = window.get_size()
    gl.glViewport(0, 0, width, height)

    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.gluPerspective(60, width / float(height), 0.01, 20)

    gl.glMatrixMode(gl.GL_TEXTURE)
    gl.glLoadIdentity()
    # texcoords are [0..1] and relative to top-left pixel corner, add 0.5 to center
    gl.glTranslatef(0.5 / image_data.width, 0.5 / image_data.height, 0)
    image_texture = image_data.get_texture()
    # texture size may be increased by pyglet to a power of 2
    tw, th = image_texture.owner.width, image_texture.owner.height
    gl.glScalef(image_data.width / float(tw),
                image_data.height / float(th), 1)

    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()

    gl.gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)

    gl.glTranslatef(0, 0, state.distance)
    gl.glRotated(state.pitch, 1, 0, 0)
    gl.glRotated(state.yaw, 0, 1, 0)

    if any(state.mouse_btns):
        axes(0.1, 4)

    gl.glTranslatef(0, 0, -state.distance)
    gl.glTranslatef(*state.translation)

    gl.glColor3f(0.5, 0.5, 0.5)
    gl.glPushMatrix()
    gl.glTranslatef(0, 0.5, 0.5)
    grid()
    gl.glPopMatrix()

    psz = max(window.get_size()) / float(max(w, h)) if state.scale else 1
    gl.glPointSize(psz)
    distance = (0, 0, 1) if state.attenuation else (1, 0, 0)
    gl.glPointParameterfv(gl.GL_POINT_DISTANCE_ATTENUATION,
                          (gl.GLfloat * 3)(*distance))

    if state.lighting:
        ldir = [0.5, 0.5, 0.5]  # world-space lighting
        ldir = np.dot(state.rotation, (0, 0, 1))  # MeshLab style lighting
        ldir = list(ldir) + [0]  # w=0, directional light
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (gl.GLfloat * 4)(*ldir))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE,
                     (gl.GLfloat * 3)(1.0, 1.0, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT,
                     (gl.GLfloat * 3)(0.75, 0.75, 0.75))
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_NORMALIZE)
        gl.glEnable(gl.GL_LIGHTING)

    gl.glColor3f(1, 1, 1)
    texture = image_data.get_texture()
    gl.glEnable(texture.target)
    gl.glBindTexture(texture.target, texture.id)
    gl.glTexParameteri(
        gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

    # comment this to get round points with MSAA on
    gl.glEnable(gl.GL_POINT_SPRITE)

    if not state.scale and not state.attenuation:
        gl.glDisable(gl.GL_MULTISAMPLE)  # for true 1px points with MSAA on
    vertex_list.draw(gl.GL_POINTS)
    gl.glDisable(texture.target)
    if not state.scale and not state.attenuation:
        gl.glEnable(gl.GL_MULTISAMPLE)

    gl.glDisable(gl.GL_LIGHTING)

    gl.glColor3f(0.25, 0.25, 0.25)
    frustum(depth_intrinsics)
    axes()
    try:
        bbox_3d(color_image, depth_frame_aligned, depth_intrinsics, color_intrinsics)
    except:
        pass

    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(0, width, 0, height, -1, 1)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    gl.glMatrixMode(gl.GL_TEXTURE)
    gl.glLoadIdentity()
    gl.glDisable(gl.GL_DEPTH_TEST)

    fps_display.draw()

angles_translations = -np.ones(3)
angles_translations_history = []
def run(dt):
    global w, h 
    global color_image, depth_frame_aligned, image_data, color_source
    global angles_translations, angles_translations_history 
    # window.set_caption("RealSense (%dx%d) %dFPS (%.2fms) %s" %
    #                    (w, h, 0 if dt == 0 else 1.0 / dt, dt * 1000,
    #                     "PAUSED" if state.paused else ""))

    if state.paused:
        return

    align_to = rs.stream.color
    align = rs.align(align_to)
    
    success, frames = pipeline.try_wait_for_frames(timeout_ms=0)
    if not success:
        return

    aligned_frames = align.process(frames)
    depth_frame_aligned = aligned_frames.get_depth_frame()
    depth_frame = depth_frame_aligned.as_video_frame()
    other_frame = aligned_frames.first(other_stream).as_video_frame()

    depth_frame = decimate.process(depth_frame)

    if state.postprocessing:
        for f in filters:
            depth_frame = f.process(depth_frame)

    # Grab new intrinsics (may be changed by decimation)
    depth_intrinsics = rs.video_stream_profile(
        depth_frame.profile).get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    color_image = np.asanyarray(other_frame.get_data())

    colorized_depth = colorizer.colorize(depth_frame)
    depth_colormap = np.asanyarray(colorized_depth.get_data())

    if state.color:
        mapped_frame, color_source = other_frame, color_image
    else:
        mapped_frame, color_source = colorized_depth, depth_colormap

    points = pc.calculate(depth_frame)
    pc.map_to(mapped_frame)

    # handle color source or size change
    fmt = convert_fmt(mapped_frame.profile.format())

    if (image_data.format, image_data.pitch) != (fmt, color_source.strides[0]):
        if state.color:
            global color_w, color_h
            image_w, image_h = color_w, color_h
        else:
            image_w, image_h = w, h

        empty = (gl.GLubyte * (image_w * image_h * 3))()
        image_data = pyglet.image.ImageData(image_w, image_h, fmt, empty)

    # copy image data to pyglet
    image_data.set_data(fmt, color_source.strides[0], color_source.ctypes.data)

    verts = np.asarray(points.get_vertices(2)).reshape(h, w, 3)
    texcoords = np.asarray(points.get_texture_coordinates(2))

    if len(vertex_list.vertices) != verts.size:
        vertex_list.resize(verts.size // 3)
        # need to reassign after resizing
        vertex_list.vertices = verts.ravel()
        vertex_list.tex_coords = texcoords.ravel()

    # copy our data to pre-allocated buffers, this is faster than assigning...
    # pyglet will take care of uploading to GPU
    def copy(dst, src):
        """copy numpy array to pyglet array"""
        # timeit was mostly inconclusive, favoring slice assignment for safety
        np.array(dst, copy=False)[:] = src.ravel()
        # ctypes.memmove(dst, src.ctypes.data, src.nbytes)

    copy(vertex_list.vertices, verts)
    copy(vertex_list.tex_coords, texcoords)

    if state.lighting:
        # compute normals
        dy, dx = np.gradient(verts, axis=(0, 1))
        n = np.cross(dx, dy)

        # can use this, np.linalg.norm or similar to normalize, but OpenGL can do this for us, see GL_NORMALIZE above
        # norm = np.sqrt((n*n).sum(axis=2, keepdims=True))
        # np.divide(n, norm, out=n, where=norm != 0)

        # import cv2
        # n = cv2.bilateralFilter(n, 5, 1, 1)

        copy(vertex_list.normals, n)

    if state.model_scan_mode is False and state.model_scan_count > 0:
        create_model_from_scans()
        state.model_scan_count = 0

    if keys[pyglet.window.key.E]:
        points.export_to_ply('./scene.ply', mapped_frame)
        o3d.visualization.draw_geometries([o3d.io.read_point_cloud('./scene.ply')], window_name="Full scene pointcloud")

    if keys[pyglet.window.key.O]:
        save_scan_pointcloud(color_image, depth_frame_aligned, depth_intrinsics, color_intrinsics, color_source)

    if state.continuous_registration:
        try:
            object_pcd = save_scan_pointcloud(color_image, depth_frame_aligned, depth_intrinsics, color_intrinsics, color_source, fast_mode=True)
            angles_translations, fitness_score = perform_registration(object_pcd=object_pcd, model_pcd=state.model_pcd, fast_mode=True)
            angles_translations_history.append((angles_translations, fitness_score))
        except Exception as e:
            print(e)
            pass
    paused_str = "PAUSED" if state.paused else ""
    window.set_caption(f"RealSense ({w}x{h}) {0 if dt == 0 else 1.0 / dt:.1f}FPS ({dt*1000:.2f}ms) Orientation ({angles_translations[0]: 06.1f}, {angles_translations[1]: 06.1f}, {angles_translations[2]: 06.1f}) {paused_str}")


print(USAGE)
pyglet.clock.schedule(run)

try:
    pyglet.app.run()
finally:
    pipeline.stop()
