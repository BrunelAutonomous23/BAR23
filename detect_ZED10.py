import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import csv
import math
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import pyzed.sl as sl
import numpy as np


def detect(save_img = False):

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    #old_img_w = old_img_h = imgsz
    old_img_b = 1

    # Create a ZED camera object

    zed = sl.Camera()
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    init.coordinate_units = sl.UNIT.CENTIMETER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_minimum_distance = 50
    init.depth_maximum_distance = 255
    
    #Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        print("exit1")
        exit(1)

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    tracking_parameters = sl.PositionalTrackingParameters()
    err = zed.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print("exit2")
        exit(1)
    #tracking_parameters.set_as_static = True
    #tracking_parameters.set_floor_as_origin = True
    zed.enable_positional_tracking(tracking_parameters)
    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode (default fps: 60)
    # Use a right-handed Y-up coordinate system
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.coordinate_units = sl.UNIT.METER  # Set units in meters

    # Enable object detection module
    obj_detection_params = sl.ObjectDetectionParameters()
    obj_detection_params.enable_tracking = True
    obj_detection_params.enable_mask_output = True
    obj_detection_params.detection_model = sl.DETECTION_MODEL.MULTI_CLASS_BOX

    err = zed.enable_object_detection(obj_detection_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("enable_object_detection", zed_error, "\nExit program.")
        exit(1)

    # Create a list to store detected objects
    objects = sl.Objects()

    # Create a dictionary to store object depths
    object_depths = {}

    # Set runtime parameters for object detection
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Sensing mode: Standard

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_configuration.camera_resolution
    image_size.width = image_size.width / 3
    image_size.height = image_size.height * 4 / 9

    # Declare your sl.Mat matrices
    point_cloud=sl.Mat()
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_map = sl.Mat()
   
    # Create CSV file
    csv_file = open('cone_depth.csv', mode='w')
    fieldnames = ['Cone', 'Depth']
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    # Run detection
    zed_pose = sl.Pose()
    fixed_cone_coordinates = []
    tracker = cv2.TrackerKCF_create()
    trackers = {}


    # Create a list to store detected objects
    objects = sl.Objects()

    # Create a dictionary to store object depths
    object_depths = {}

    # Set runtime parameters for object detection
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Sensing mode: Standard

    # Detect and track objects for 1000 frames

    while True:
        zed.grab()
        err=zed.grab(runtime)
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image and depth map
            left_image = sl.Mat()
            zed.retrieve_image(left_image, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed.retrieve_image(depth_map,sl.VIEW.DEPTH)
            zed.retrieve_objects(objects)

            im0 = left_image.get_data()
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            new_shape = (imgsz, imgsz)
            im0 = cv2.resize(im0, new_shape)
            cv2.imshow('image', cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
            print("running")
            # Iterate through the detected objects
            for obj in objects.object_list:
                # Get the object's ID
                obj_id = obj.id

                # Check if the object is already stored
                if obj_id not in object_depths:
                    # Get the object's 3D position in the world frame
                    translation = obj.position

                    # Store the object's depth
                    object_depths[obj_id] = translation[2]


    # Close the camera

    # Write object depths to a CSV file
    with open("cone_depth.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Object ID", "Depth"])

        for obj_id, depth in object_depths.items():
            writer.writerow([obj_id, depth])

print("Object depths saved to 'object_depths.csv' file.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()