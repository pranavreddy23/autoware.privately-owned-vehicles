#! /usr/bin/env python3

import os
import cv2
import math
import json
import argparse
import warnings
import numpy as np
from PIL import Image, ImageDraw
from process_curvelanes import (
    custom_warning_format, 
    round_line_floats,
    normalizeCoords,
    interpLine,
    getLineAnchor
)

warnings.formatwarning = custom_warning_format

PointCoords = tuple[float, float]
ImagePointCoords = tuple[int, int]

# Skipped frames
skipped_dict = {}

# ============================== Helper functions ============================== #


def log_skipped(frame_id, reason):
    skipped_dict[frame_id] = reason


def drawLine(
    img: np.ndarray, 
    line: list,
    color: tuple,
    thickness: int = 2
):
    for i in range(1, len(line)):
        pt1 = (
            int(line[i - 1][0]), 
            int(line[i - 1][1])
        )
        pt2 = (
            int(line[i][0]), 
            int(line[i][1])
        )
        cv2.line(
            img, 
            pt1, pt2, 
            color = color, 
            thickness = thickness
        )


def annotateGT(
    img: np.ndarray,
    frame_id: str,
    bev_egopath: list,
    raw_dir: str, 
    visualization_dir: str,
    normalized: bool
):
    """
    Annotates and saves an image with:
        - Raw image, in "output_dir/image".
        - Annotated image with all lanes, in "output_dir/visualization".
    """

    # Save raw img in raw dir, as PNG
    cv2.imwrite(
        os.path.join(
            raw_dir,
            f"{frame_id}.png"
        ),
        img
    )

    # Draw egopath
    if (normalized):
        h, w, _ = img.shape
        renormed_bev_egopath = [
            (x * w, y * h) 
            for x, y in bev_egopath
        ]
    else:
        renormed_bev_egopath = bev_egopath
    drawLine(
        img = img,
        line = renormed_bev_egopath,
        color = COLOR_EGOPATH
    )

    # Save visualization img in vis dir, as JPG (saving storage space)
    cv2.imwrite(
        os.path.join(
            visualization_dir,
            f"{frame_id}.jpg"
        ),
        img
    )


def interpX(line, y):
    """
    Interpolate x-value of a point on a line, given y-value
    """
    points = np.array(line)
    list_x = points[:, 0]
    list_y = points[:, 1]

    if not np.all(np.diff(list_y) > 0):
        sort_idx = np.argsort(list_y)
        list_y = list_y[sort_idx]
        list_x = list_x[sort_idx]

    return float(np.interp(y, list_y, list_x))


def polyfit_BEV(
    bev_egopath: list,
    order: int,
    y_step: int,
    y_limit: int
):
    x = [point[0] for point in bev_egopath]
    y = [point[1] for point in bev_egopath]
    z = np.polyfit(y, x, order)
    f = np.poly1d(z)
    y_new = np.linspace(
        0, y_limit, 
        int(y_limit / y_step) + 1
    )
    x_new = f(y_new)

    # Sort by decreasing y
    fitted_bev_egopath = sorted(
        tuple(zip(x_new, y_new)),
        key = lambda x: x[1],
        reverse = True
    )

    flag_list = [0] * len(fitted_bev_egopath)
    for i in range(len(fitted_bev_egopath)):
        if (not 0 <= fitted_bev_egopath[i][0] <= BEV_W):
            flag_list[i - 1] = 1
            break
    if (not 1 in flag_list):
        flag_list[-1] = 1

    validity_list = [1] * len(fitted_bev_egopath)
    last_valid_index = flag_list.index(1)
    for i in range(last_valid_index + 1, len(validity_list)):
        validity_list[i] = 0
    
    return fitted_bev_egopath, flag_list, validity_list


def imagePointTuplize(point: PointCoords) -> ImagePointCoords:
    """
    Parse all coords of an (x, y) point to int, making it
    suitable for image operations.
    """
    return (int(point[0]), int(point[1]))


def findSourcePointsBEV(
    h: int,
    w: int,
    egoleft: list,
    egoright: list,
) -> dict:
    """
    Find 4 source points for the BEV homography transform.
    """
    sps = {}

    # Renorm 2 egolines
    egoleft = [
        [p[0] * w, p[1] * h]
        for p in egoleft
    ]
    egoright = [
        [p[0] * w, p[1] * h]
        for p in egoright
    ]

    # Acquire LS and RS
    anchor_left = getLineAnchor(egoleft, h)
    anchor_right = getLineAnchor(egoright, h)
    sps["LS"] = [anchor_left[0], h]
    sps["RS"] = [anchor_right[0], h]

    # CALCULATING LE AND RE BASED ON LATEST ALGORITHM

    midanchor_start = [(sps["LS"][0] + sps["RS"][0]) / 2, h]
    ego_height = max(egoleft[-1][1], egoright[-1][1]) * 1.05

    # Both egos have Null anchors
    if ((not anchor_left[1]) and (not anchor_right[1])):
        midanchor_end = [midanchor_start[0], h]
        original_end_w = sps["RS"][0] - sps["LS"][0]

    else:
        left_deg = 90 if (not anchor_left[1]) else math.degrees(math.atan(anchor_left[1])) % 180
        right_deg = 90 if (not anchor_right[1]) else math.degrees(math.atan(anchor_right[1])) % 180
        mid_deg = (left_deg + right_deg) / 2
        mid_grad = - math.tan(math.radians(mid_deg))
        mid_intercept = h - mid_grad * midanchor_start[0]
        midanchor_end = [
            (ego_height - mid_intercept) / mid_grad,
            ego_height
        ]
        original_end_w = interpX(egoright, ego_height) - interpX(egoleft, ego_height)

    sps["LE"] = [
        midanchor_end[0] - original_end_w / 2,
        ego_height
    ]
    sps["RE"] = [
        midanchor_end[0] + original_end_w / 2,
        ego_height
    ]

    # Tuplize 4 corners
    for i, pt in sps.items():
        sps[i] = imagePointTuplize(pt)

    # Log the ego_height too
    sps["ego_h"] = ego_height

    return sps


def transformBEV(
    img: np.ndarray,
    egopath: list,
    sps: dict
):
    h, w, _ = img.shape

    # Renorm/tuplize drivable path
    egopath = [
        (point[0] * w, point[1] * h) for point in egopath
        if (point[1] * h >= sps["ego_h"])
    ]
    if (not egopath):
        return (None, None, None, None, None)

    # Interp more points for original egopath
    egopath = interpLine(egopath, MIN_POINTS)

    # Get transformation matrix
    mat, _ = cv2.findHomography(
        srcPoints = np.array([
            sps["LS"],
            sps["RS"],
            sps["LE"],
            sps["RE"]
        ]),
        dstPoints = np.array([
            BEV_PTS["LS"],
            BEV_PTS["RS"],
            BEV_PTS["LE"],
            BEV_PTS["RE"],
        ])
    )

    # Transform image
    im_dst = cv2.warpPerspective(
        img, mat,
        np.array([BEV_W, BEV_H])
    )

    # Transform egopath
    bev_egopath = np.array(
        egopath,
        dtype = np.float32
    ).reshape(-1, 1, 2)
    bev_egopath = cv2.perspectiveTransform(bev_egopath, mat)
    bev_egopath = [
        tuple(map(int, point[0])) 
        for point in bev_egopath
    ]

    # Polyfit BEV egopath to get 33-coords format with flags
    bev_egopath, flag_list, validity_list = polyfit_BEV(
        bev_egopath = bev_egopath,
        order = POLYFIT_ORDER,
        y_step = BEV_Y_STEP,
        y_limit = BEV_H
    )

    return (im_dst, bev_egopath, flag_list, validity_list, mat)


# ============================== Main run ============================== #


if __name__ == "__main__":

    # DIRECTORY STRUCTURE

    IMG_DIR = "image"
    JSON_PATH = "drivable_path.json"

    BEV_IMG_DIR = "image_bev"
    BEV_VIS_DIR = "visualization_bev"
    BEV_JSON_PATH = "drivable_path_bev.json"
    BEV_SKIPPED_JSON_PATH = "skipped_frames.json"

    # OTHER PARAMS

    MIN_POINTS = 30

    BEV_PTS = {
        "LS" : [120, 640],          # Left start
        "RS" : [200, 640],          # Right start
        "LE" : [120, 0],            # Left end
        "RE" : [200, 0]             # Right end
    }

    BEV_W = 320
    BEV_H = 640
    BEV_Y_STEP = 20
    POLYFIT_ORDER = 2

    COLOR_EGOPATH = (0, 255, 255)   # Yellow (BGR)

    # PARSING ARGS

    parser = argparse.ArgumentParser(
        description = "Generating BEV from CurveLanes processed datasets"
    )
    parser.add_argument(
        "--dataset_dir", 
        type = str, 
        help = "Processed CurveLanes directory",
        required = True
    )
    # For debugging only
    parser.add_argument(
        "--early_stopping",
        type = int,
        help = "Num. frames you wanna limit, instead of whole set.",
        required = False
    )
    args = parser.parse_args()

    # Parse dataset dir
    dataset_dir = args.dataset_dir
    IMG_DIR = os.path.join(dataset_dir, IMG_DIR)
    JSON_PATH = os.path.join(dataset_dir, JSON_PATH)
    BEV_JSON_PATH = os.path.join(dataset_dir, BEV_JSON_PATH)
    BEV_SKIPPED_JSON_PATH = os.path.join(dataset_dir, BEV_SKIPPED_JSON_PATH)

    # Parse early stopping
    if (args.early_stopping):
        print(f"Early stopping set, stopping after {args.early_stopping} files.")
        early_stopping = args.early_stopping
    else:
        early_stopping = None

    # Generate new dirs and paths
    BEV_IMG_DIR = os.path.join(dataset_dir, BEV_IMG_DIR)
    BEV_VIS_DIR = os.path.join(dataset_dir, BEV_VIS_DIR)

    if not (os.path.exists(BEV_IMG_DIR)):
        os.makedirs(BEV_IMG_DIR)
    if not (os.path.exists(BEV_VIS_DIR)):
        os.makedirs(BEV_VIS_DIR)

    # Preparing data
    with open(JSON_PATH, "r") as f:
        json_data = json.load(f)
    data_master = {}    # Dumped later

    # MAIN GENERATION LOOP

    counter = 0
    for frame_id, frame_content in json_data.items():

        counter += 1

        # Acquire frame
        frame_img_path = os.path.join(
            IMG_DIR,
            f"{frame_id}.png"
        )
        img = cv2.imread(frame_img_path)
        h, w, _ = img.shape

        # Acquire frame data
        this_frame_data = json_data[frame_id]

        # MAIN ALGORITHM
        try:

            # Get source points for transform
            sps_dict = findSourcePointsBEV(
                h = h,
                w = w,
                egoleft = this_frame_data["egoleft_lane"],
                egoright = this_frame_data["egoright_lane"]
            )

            # Transform to BEV space
            
            (im_dst, bev_egopath, flag_list, validity_list, mat) = transformBEV(
                img = img,
                egopath = this_frame_data["drivable_path"],
                sps = sps_dict
            )

            # Skip if invalid frame (due to too high ego_height value)
            if (not bev_egopath):
                log_skipped(
                    frame_id,
                    "Null EgoPath from BEV transformation algorithm."
                )
                continue

            # Save stuffs
            annotateGT(
                img = im_dst,
                frame_id = frame_id,
                bev_egopath = bev_egopath,
                raw_dir = BEV_IMG_DIR,
                visualization_dir = BEV_VIS_DIR,
                normalized = False
            )

            # Register this frame GT to master JSON
            # Each point has tuple format (x, y, flag, valid)
            data_master[frame_id] = {
                "drivable_path" : [
                    (point[0], point[1], flag, valid)
                    for point, flag, valid in list(zip(
                        round_line_floats(
                            normalizeCoords(
                                bev_egopath,
                                width = BEV_W,
                                height = BEV_H
                            )
                        ), 
                        flag_list, 
                        validity_list
                    ))
                ],
                "transform_matrix" : mat.tolist()
            }

        except Exception as e:
            log_skipped(frame_id, str(e))
            continue

        # Break if early_stopping reached
        if (early_stopping is not None):
            if (counter >= early_stopping):
                break

    # Save master data
    with open(BEV_JSON_PATH, "w") as f:
        json.dump(data_master, f, indent = 4)

    # Save skipped frames
    with open(BEV_SKIPPED_JSON_PATH, "w") as f:
        json.dump(skipped_dict, f, indent = 4)