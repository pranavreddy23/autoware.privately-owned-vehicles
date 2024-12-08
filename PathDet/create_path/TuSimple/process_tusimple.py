#! /usr/bin/env python3

import argparse
import os
import pathlib
import json
import numpy as np
from PIL import Image, ImageDraw
import sys
import warnings

# ============================== Dataset structure ============================== #

root_dir = "TUSimple"
train_dir = "train_set"
test_dir = "test_set"

train_clip_codes = ["0313", "0531", "0601"] # Train labels are split into 3 dirs
test_file = "test_label.json"               # Test file name

img_width = 1280
img_height = 720

# ============================== Helper functions ============================== #

def normalizeCoords(lane, width, height):

    return [(x / width, y / height) for x, y in lane]

def getEgoIndexes(anchors):

    # `anchors` is a list of tuples at form (xi, yi)
    # In those labels, lanes are labeled from left to right, so these anchors
    # extracted from them are also sorted x-coords in ascending order.
    for i in range(len(anchors)):
        if (anchors[i][0] >= img_width / 2):
            if (i == 0):
                return "NO LANES on the LEFT side of frame. Something's wrong!"
            left_ego_idx, right_ego_idx = i - 1, i
            return (left_ego_idx, right_ego_idx)
    
    return "NO LANES on the RIGHT side of frame. Something's wrong!"

def getLaneAnchor(lane):

    (x1, y1) = lane[len(lane) // 2]
    (x2, y2) = lane[-1]
    if (x1 == x2):
        warnings.warn(f"Vertical lane  detected: {lane}")
        return (x1, None, None)
    # print("x1, y1:", x1, y1)
    # print("x2, y2:", x2, y2)
    a = (y2 - y1) / (x2 - x1)
    # print("a:", a)
    b = y1 - a * x1
    # print("b:", b)
    x0 = (img_height - b) / a
    # print("x0:", x0)
    
    return (x0, b)

def parseAnnotations(anno_path):

    # Read em raw
    with open(anno_path, "r") as f:
        read_data = [json.loads(line) for line in f.readlines()]

    # Parse em 
    anno_data = {}
    for item in read_data:
        lanes = item["lanes"]
        h_samples = item["h_samples"]
        raw_file = item["raw_file"]

        lanes_decoupled = [
            [(x, y) for x, y in zip(lane, h_samples) if x != -2]
            for lane in lanes
        ]

        lane_anchors = [getLaneAnchor(lane) for lane in lanes_decoupled]

        ego_indexes = getEgoIndexes(lane_anchors)

        if (type(ego_indexes) is str):
            if (ego_indexes.startswith("NO")):
                warnings.warn(f"Error parsing {raw_file}: {ego_indexes}")
                continue

        left_ego = lanes_decoupled[ego_indexes[0]]
        right_ego = lanes_decoupled[ego_indexes[1]]

        # Parse processed data, all coords normalized
        anno_data[raw_file] = {
            "lanes": [normalizeCoords(lane, img_width, img_height) for lane in lanes_decoupled],
            "ego_indexes": ego_indexes,
            "img_size": (img_width, img_height),
        }

        # 

    return anno_data
            
if __name__ == "__main__":

    # ============================== Parsing args ============================== #

    parser = argparse.ArgumentParser(
        description = "Process TuSimple dataset - PathDet groundtruth generation"
    )
    parser.add_argument(
        "--dataset_dir", 
        type = str, 
        help = "TuSimple directory (right after extraction)"
    )
    parser.add_argument(
        "--output_dir", 
        type = str, 
        help = "Output directory"
    )
    parser.add_argument(
        "--test_label_file", 
        type = str
    )
    args = parser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok = True, parents = True)
    test_label_file = args.test_label_file

    # ============================== Parsing annotations ============================== #

    train_label_files = [
        dataset_dir / root_dir / train_dir / f"label_data_{clip_code}.json" 
        for clip_code in train_clip_codes
    ]

    # Make no mistake: the file `TuSimple/test_set/test_tasks_0627.json`` is NOT a label file.
    # It is just a template for test submission. Kinda weird they put it in `test_set` while
    # the actual label file is in `TuSimple/test_label.json`.
    test_label_files = [dataset_dir / root_dir / test_file]

    # Getting list of all json files
    # json_files = sorted([f for f in dataset_dir.glob("*.json")])

    # for json_file in json_files:
    #     parseAnnotations(json_file)

    # print(train_label_files)
    # print(test_label_files)

    first_file = "test_set/clips/0530/1492626760788443246_0/20.jpg"

    test_data = parseAnnotations(test_label_file)
    first_test_data = test_data[first_file]

    print(first_test_data)