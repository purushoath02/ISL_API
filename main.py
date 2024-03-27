from generate_keypoints import *
import os
import json
import multiprocessing
import cv2
import mediapipe as mp
import numpy as np
import gc
import warnings
import pandas as pd 
import torch 
import transformer
import torch.nn as nn
from configs import TransformerConfig
max_frame_len=200
frame_length=1080
frame_width=1920
dict ={
    0: "bank",
    1: "biglarge",
    2: "bird",
    3: "black",
    4: "boy",
    5: "brother",
    6: "car",
    7: "cellphone",
    8: "court",
    9: "cow",
    10: "death",
    11: "dog",
    12: "dry",
    13: "election",
    14: "fall",
    15: "fan",
    16: "father",
    17: "girl",
    18: "good",
    19: "goodmorning",
    20: "happy",
    21: "hat",
    22: "hello",
    23: "hot",
    24: "house",
    25: "i",
    26: "it",
    27: "long",
    28: "loud",
    29: "monday",
    30: "new",
    31: "paint",
    32: "pen",
    33: "priest",
    34: "quiet",
    35: "red",
    36: "shoes",
    37: "short",
    38: "smalllittle",
    39: "storeorshop",
    40: "summer",
    41: "teacher",
    42: "thankyou",
    43: "time",
    44: "trainticket",
    45: "tshirt",
    46: "white",
    47: "window",
    48: "year",
    49: "youplural"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def combine_xy(x, y):
    x, y = np.array(x), np.array(y)
    #_, length = x.shape
    length = 1 
    x = x.reshape((-1, length, 1))
    y = y.reshape((-1, length, 1))
    return np.concatenate((x, y), -1).astype(np.float32)
def interpolate(arr):

        arr_x = arr[:, :, 0]
        arr_x = pd.DataFrame(arr_x)
        arr_x = arr_x.interpolate(method="linear", limit_direction="both").to_numpy()

        arr_y = arr[:, :, 1]
        arr_y = pd.DataFrame(arr_y)
        arr_y = arr_y.interpolate(method="linear", limit_direction="both").to_numpy()

        if np.count_nonzero(~np.isnan(arr_x)) == 0:
            arr_x = np.zeros(arr_x.shape)
        if np.count_nonzero(~np.isnan(arr_y)) == 0:
            arr_y = np.zeros(arr_y.shape)

        arr_x = arr_x * frame_width
        arr_y = arr_y * frame_length

        return np.stack([arr_x, arr_y], axis=-1)
    
def temp(image):
    pose_points_x, pose_points_y = [], []
    hand1_points_x, hand1_points_y = [], []
    hand2_points_x, hand2_points_y = [], []

    holistics = mp.solutions.holistic
    holistic = holistics.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    ## Extract pose and hand landmarks
    pose_x, pose_y = process_pose_keypoints(results)
    hand1_x, hand1_y, hand2_x, hand2_y = process_hand_keypoints(results)

    # pose_x = pose_x if pose_x else [np.nan] * 25
    # pose_y = pose_y if pose_y else [np.nan] * 25

    # hand1_x = hand1_x if hand1_x else [np.nan] * 21
    # hand1_y = hand1_y if hand1_y else [np.nan] * 21
    # hand2_x = hand2_x if hand2_x else [np.nan] * 21
    # hand2_y = hand2_y if hand2_y else [np.nan] * 21
    
    
    
    pose_x = pose_x if pose_x else [0] * 25
    pose_y = pose_y if pose_y else [0] * 25

    hand1_x = hand1_x if hand1_x else [0] * 21
    hand1_y = hand1_y if hand1_y else [0] * 21
    hand2_x = hand2_x if hand2_x else [0] * 21
    hand2_y = hand2_y if hand2_y else [0] * 21

    pose_points_x.append(pose_x)
    pose_points_y.append(pose_y)
    hand1_points_x.append(hand1_x)
    hand1_points_y.append(hand1_y)
    hand2_points_x.append(hand2_x)
    hand2_points_y.append(hand2_y)
    pose = combine_xy(pose_points_x, pose_points_y)
    h1 =  combine_xy(hand1_points_x, hand1_points_y)
    h2 =  combine_xy(hand2_points_x, hand2_points_y)
    #print(pose.shape)
    pose = (
        np.array(list(map(np.array, pose.tolist())))
        .reshape(-1, 50)
        .astype(np.float32)
    )
    #print(h1.shape)
    h1 = (
        np.array(list(map(np.array, h1.tolist())))
        .reshape(-1, 42)
        .astype(np.float32)
    )
    h2 = (
        np.array(list(map(np.array, h2.tolist())))
        .reshape(-1, 42)
        .astype(np.float32)
    )
    final_data = np.concatenate((pose, h1, h2), -1)
    # final_data = np.pad(
    #     final_data,
    #     ((0, self.max_frame_len - final_data.shape[0]), (0, 0)),
    #     "constant",
    # )
    
    gc.collect()
    return final_data

def extract_key(path): 
    holistics = mp.solutions.holistic
    holistic = holistics.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    pose_points_x, pose_points_y = [], []
    hand1_points_x, hand1_points_y = [], []
    hand2_points_x, hand2_points_y = [], []


    if not os.path.isfile(path):
        warnings.warn(path + " file not found")

    cap = cv2.VideoCapture(path)
    config = TransformerConfig('large')
    n_classes =50
    model = transformer.Transformer(config=config, n_classes=n_classes)
        
    model = model.to(device)

    model_path = "/home/red/Downloads/transformer_augs(1).pth"
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt["model"])
    input_data =[]
    model.eval()
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = holistic.process(image)

        ## Extract pose and hand landmarks
        pose_x, pose_y = process_pose_keypoints(results)
        hand1_x, hand1_y, hand2_x, hand2_y = process_hand_keypoints(results)

        
        ## Set to nan so that values can be interpolated in dataloader
        pose_x = pose_x if pose_x else [np.nan] * 25
        pose_y = pose_y if pose_y else [np.nan] * 25

        hand1_x = hand1_x if hand1_x else [np.nan] * 21
        hand1_y = hand1_y if hand1_y else [np.nan] * 21
        hand2_x = hand2_x if hand2_x else [np.nan] * 21
        hand2_y = hand2_y if hand2_y else [np.nan] * 21

        pose_points_x.append(pose_x)
        pose_points_y.append(pose_y)
        hand1_points_x.append(hand1_x)
        hand1_points_y.append(hand1_y)
        hand2_points_x.append(hand2_x)
        hand2_points_y.append(hand2_y)

       

    cap.release()

    ## Set to nan so that values can be interpolated in dataloader
    pose_points_x = pose_points_x if pose_points_x else [[np.nan] * 25]
    pose_points_y = pose_points_y if pose_points_y else [[np.nan] * 25]

    hand1_points_x = hand1_points_x if hand1_points_x else [[np.nan] * 21]
    hand1_points_y = hand1_points_y if hand1_points_y else [[np.nan] * 21]
    hand2_points_x = hand2_points_x if hand2_points_x else [[np.nan] * 21]
    hand2_points_y = hand2_points_y if hand2_points_y else [[np.nan] * 21]

    row = {

        "pose_x": pose_points_x,
        "pose_y": pose_points_y,
        "hand1_x": hand1_points_x,
        "hand1_y": hand1_points_y,
        "hand2_x": hand2_points_x,
        "hand2_y": hand2_points_y,
    
    }
    pose = combine_xy(row["pose_x"], row["pose_y"])
    h1 = combine_xy(row["hand1_x"], row["hand1_y"])
    h2 = combine_xy(row["hand2_x"], row["hand2_y"])

    pose = interpolate(pose)
    h1 = interpolate(h1)
    h2 = interpolate(h2)
    holistic.close()
    del holistic
    gc.collect()
    
    df = pd.DataFrame.from_dict(
        {
            
            "pose": pose.tolist(),
            "hand1": h1.tolist(),
            "hand2": h2.tolist(),
            
        }
    )
        
    #print('#########################')
    #print(df.uid)
    
    #print(pose.shape)
    pose = (
        np.array(list(map(np.array, df.pose.values)))
        .reshape(-1, 50)
        .astype(np.float32)
    )
    #print(h1.shape)
    h1 = (
        np.array(list(map(np.array, df.hand1.values)))
        .reshape(-1, 42)
        .astype(np.float32)
    )
    h2 = (
        np.array(list(map(np.array, df.hand2.values)))
        .reshape(-1, 42)
        .astype(np.float32)
    )
    final_data = np.concatenate((pose, h1, h2), -1)
    final_data = np.pad(
        final_data,
        ((0, max_frame_len - final_data.shape[0]), (0, 0)),
        "constant",
    )
    
    # input_tensor = torch.FloatTensor(final_data)
    # input_tensor_device = input_tensor.to(device)
    
    
    
    # with torch.no_grad():
    #     output = model(input_tensor_device)
    #     predicted_class = torch.argmax(output)
    #     print(dict[predicted_class.item()])
    #     input_tensor.detach()
    gc.collect()

    
    
    

if __name__ == "__main__":

    # save_keypoints(args.dataset, val_paths, "val")
    # save_keypoints(args.dataset, test_paths, "test")
    path = "/home/red/Downloads/INCLUDE/Adjectives/2. quiet/MVI_5180.MOV"
    extract_key(path)
    