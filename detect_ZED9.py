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

def detect(save_img=False):
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

    zed = sl.Camera()

    # Set depth mode
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
        exit(1)

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    tracking_parameters = sl.PositionalTrackingParameters()
    err = zed.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
    #tracking_parameters.set_as_static = True
    #tracking_parameters.set_floor_as_origin = True
    zed.enable_positional_tracking(tracking_parameters)

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_configuration.camera_resolution
    image_size.width = image_size.width / 3
    image_size.height = image_size.height * 4 / 9

    # Declare your sl.Mat matrices
    bodies = sl.Bodies()
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
    while True:
        # Get frame from ZED camera
        zed.grab()
        #zed_pose = sl.Pose()
        err=zed.grab(runtime)
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image and depth map
            left_image = sl.Mat()
            zed.retrieve_image(left_image, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed.retrieve_image(depth_map,sl.VIEW.DEPTH)

            
        
        #positional tracking
            state = zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
        # Extract translation and orientation information
            py_translation = sl.Translation()
            tx = round(zed_pose.get_translation(py_translation).get()[0], 3)
            ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
            tz = round(zed_pose.get_translation(py_translation).get()[2], 3)
            print("Translation: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3}\n".format(tx, ty, tz, zed_pose.timestamp.get_milliseconds()))

            # Display the orientation quaternion
            py_orientation = sl.Orientation()
            ox = round(zed_pose.get_orientation(py_orientation).get()[0], 3)
            oy = round(zed_pose.get_orientation(py_orientation).get()[1], 3)
            oz = round(zed_pose.get_orientation(py_orientation).get()[2], 3)
            ow = round(zed_pose.get_orientation(py_orientation).get()[3], 3)
            print("Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))

        # Convert ZED image to OpenCV format
        im0 = left_image.get_data()
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

	
        # Get depth data
        depth_data = depth_map.get_data().astype(np.float32)

        # Resize image and depth map
        new_shape = (imgsz, imgsz)
        im0 = cv2.resize(im0, new_shape)
        depth_data = cv2.resize(depth_data, new_shape)
        print(depth_image_zed.get_data().shape)
        print(depth_image_zed.get_data().dtype)


        # Normalize depth data
        #depth_data /= 100.0  # Convert cm to meters

        # Convert depth map to grayscale
        depth_map_gray = (depth_data * 255 / depth_data.max()).astype(np.uint8)
#img = img[:, :, ::-1].transpose(2, 0, 1)
        # Perform object detection
        img = letterbox(im0, new_shape=imgsz)[0]
        img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        if half:
            img = img.half()
        pred = model(img)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, 0.3, 0.45)
       
        intrinsic_parameters={'fx':500,'fy':500,'cx':320,'cy':240}  

        # Process detections
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in det:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    # Get object ID from class label
                    obj_id = i #was int(cls)

                    # Initialize a tracker for the new object
                    if obj_id not in trackers:
                        tracker = cv2.TrackerKCF_create()
                        trackers[obj_id] = tracker
                        bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])]
                        print(bbox)
                        tracker.init(im0, bbox)

                    # Update the tracker with the new frame
                    tracker = trackers[obj_id]
                    success, bbox = tracker.update(im0)

                    # If the tracker successfully tracks the object, update the bounding box coordinates
                    if success:
                        xyxy = (int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

                    # Draw the bounding box and label on the image
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    # Get depth at object center
                    x_center = int((xyxy[0] + xyxy[2]) / 2)
                    y_center = int((xyxy[1] + xyxy[3]) / 2)
                    depth = depth_data[y_center, x_center] #pixel value
                    depth_measurement = 1 + (((255 - depth[0])/255)*150)
                    #depth_measurement = depth_map.get_value(x_center, y_center)
                    # Write cone depth to CSV
                    # cone_index = i + 1
                    # csv_writer.writerow({'Cone': cone_index, 'Depth': depth_measurement})
                    #Convert image coordinates to camera coordinates
                    fx = intrinsic_parameters['fx']
                    fy = intrinsic_parameters['fy']
                    cx = intrinsic_parameters['cx']
                    cy = intrinsic_parameters['cy']
                    #calculate cone coordinates in camera frame
                    cone_x = (x_center - cx) * depth_measurement / fx
                    cone_y = (y_center - cy) * depth_measurement / fy
                    cone_z = depth_measurement/100
                    print(f'Cone Coordinates: X: {cone_x:.2f}, Y: {cone_y:.2f}, Z: {cone_z:.2f}')
                    # Calculate fixed cone coordinates in world frame
                    fixed_cone_x = tx + cone_x * ox + cone_y * oy + cone_z * oz
                    fixed_cone_y = ty + cone_x * -oy + cone_y * ox + cone_z * ow
                    fixed_cone_z = tz + cone_z
                    # Append fixed cone coordinates to the list
                    fixed_cone_coordinates.append((fixed_cone_x, fixed_cone_y, fixed_cone_z))
                    # Print fixed cone coordinates
                    for i, (x, y, z) in enumerate(fixed_cone_coordinates):
                        print(f'Fixed Cone {i+1} Coordinates: X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}')
                        # Write cone depth to CSV
                        csv_writer.writerow({'Cone': i, 'Depth': cone_z})
                       
                    # #Define rotation matrix from quaternion
                    # rotation_matrix = np.array([
                        # [1 - 2 * (oy**2 + oz**2), 2 * (ox * oy - oz * ow), 2 * (ox * oz + oy * ow)],
                        # [2 * (ox * oy + oz * ow), 1 - 2 * (ox**2 + oz**2), 2 * (oy * oz - ox * ow)],
                        # [2 * (ox * oz - oy * ow), 2 * (oy * oz + ox * ow), 1 - 2 * (ox**2 + oy**2)]
                    # ])
                    # #Apply rotation to cone coordinates
                    # global_cone_coords = np.dot(rotation_matrix, np.array([cone_x, cone_y, cone_z]))
                    # # Add camera position to obtain global coordinates
                    # global_cone_coords += np.array([tx, ty, tz])

                    # # Print global cone coordinates
                    # print(f'Global Cone Coordinates: X: {global_cone_coords[0]:.2f}, '
                  # f'Y: {global_cone_coords[1]:.2f}, Z: {global_cone_coords[2]:.2f}')
    
                    # Display depth on image
                    # Display depth on image
                    # Display depth on image
                    #cv2.putText(im0, 'Depth: ' + str(depth_measurement) + ' cm', (xyxy[0],xyxy[1] - 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    #cv2.putText(im0, (f'X: {cone_x:.2f}, Y: {cone_y:.2f}, Z: {cone_z:.2f}' + ' cm'), (xyxy[0],xyxy[1] - 100),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    
        # Print time (inference + NMS)
        #print(f'inference: {t2 - t1:.3f}s')
        

        # Display image
        cv2.imshow('image', cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
        cv2.imshow('depth map', depth_map_gray )
         
        # Save image with detections
        if save_img:
            out.write(im0)

        # Quit if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release resources
    #zed.close()
    #cv2.destroyAllWindows()

#detect(save_img=True)


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
