import os
import json
import multiprocessing
import os.path
import cv2
import mediapipe as mp
import numpy as np
import warnings
import pandas as pd 
import torch 

def load_json(path):
    with open(path, "r") as f:
        json_file = json.load(f)
    return json_file

class Prep():
    def __init__(self,value = None):
        self.value = value 
        self.dict = {
            "pose_x": [],
            "pose_y": [],
            "hand1_x": [],
            "hand1_y": [],
            "hand2_x": [],
            "hand2_y": [],
        }
    
    def load_label_map(self):
        file_path = f"label_maps/label_map_include50.json"
        label_map = load_json(file_path)
        l = {v: k for k, v in label_map.items()}
        return l
    
    
    
    def combine_xy(self, x, y):
        x, y = np.array(x), np.array(y)
        _, length = x.shape
        x = x.reshape((-1, length, 1))
        y = y.reshape((-1, length, 1))
        return np.concatenate((x, y), -1).astype(np.float32)
    
    
    def process_hand_keypoints(self, results):
        hand1_x, hand1_y, hand2_x, hand2_y = [], [], [], []

        if results.right_hand_landmarks:
            landmarks = results.right_hand_landmarks.landmark
            for landmark in landmarks:
                hand1_x.append(landmark.x)
                hand1_y.append(landmark.y)
        if results.left_hand_landmarks:
            landmarks = results.left_hand_landmarks.landmark
            for landmark in landmarks:
                hand2_x.append(landmark.x)
                hand2_y.append(landmark.y)
        return hand1_x, hand1_y, hand2_x, hand2_y

    def process_pose_keypoints(self, results):
        pose_x, pose_y = [],[]
        if results.pose_landmarks: 
            count = 0
            for landmark in results.pose_landmarks.landmark:
                if count<25:
                    pose_x.append(landmark.x)
                    pose_y.append(landmark.y)
                    count+=1
                else:
                    count=0
                    break

        return pose_x, pose_y

    def process_frame(self, img, ret, num=1): 
        if not ret: 
            return
        holistics = mp.solutions.holistic
        holistic = holistics.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        pose_points_x, pose_points_y = [], []
        hand1_points_x, hand1_points_y = [], []
        hand2_points_x, hand2_points_y = [], []
    
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        #print(results)
        pose_x, pose_y = self.process_pose_keypoints(results)
        hand1_x, hand1_y, hand2_x, hand2_y = self.process_hand_keypoints(results)
        pose_x = pose_x if pose_x else [np.nan] * 25
        pose_y = pose_y if pose_y else [np.nan] * 25
        hand1_x = hand1_x if hand1_x else [np.nan] * 21
        hand1_y = hand1_y if hand1_y else [np.nan] * 21
        hand2_x = hand2_x if hand2_x else [np.nan] * 21
        hand2_y = hand2_y if hand2_y else [np.nan] * 21
        
        # pose_x = pose_x if pose_x else [0] * 25
        # pose_y = pose_y if pose_y else [0] * 25
        # hand1_x = hand1_x if hand1_x else [0] * 21
        # hand1_y = hand1_y if hand1_y else [0] * 21
        # hand2_x = hand2_x if hand2_x else [0] * 21
        # hand2_y = hand2_y if hand2_y else [0] * 21
        if num > 10:
            self.dict.clear

        self.dict['pose_x'].append(pose_x)
        self.dict['pose_y'].append(pose_y)
        self.dict['hand1_x'].append(hand1_x)
        self.dict['hand1_y'].append(hand1_y)
        self.dict['hand2_x'].append(hand2_x)
        self.dict['hand2_y'].append(hand2_y)
        return self.dict
    
    def get_data(self, dic):
        max_frame_len=256
        frame_length=1080
        frame_width=1920
        #print(dic)
        pose = self.combine_xy(dic["pose_x"], dic["pose_y"])
        h1 = self.combine_xy(dic["hand1_x"], dic["hand1_y"])
        h2 = self.combine_xy(dic["hand2_x"], dic["hand2_y"])
        df = pd.DataFrame.from_dict(
            {
                "pose": pose.tolist(),
                "hand1":h1.tolist(), 
                "hand2":h2.tolist(), 
            }
        )
        pose = (
            np.array(list(map(np.array, df.pose.values)))
            .reshape(-1, 50)
            .astype(np.float32)
        )
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
        return torch.FloatTensor(final_data)
        
        
class prediction: 
    def __init__(self,value =None):
        self.value = value
    def get_prediction(self,data,model,label_map):
        model.eval()
        with torch.no_grad():
            #print(data)
            output = model(data.cuda()).detach().cpu()
            output = torch.argmax(torch.softmax(output, dim=-1), dim=-1).numpy()
            prediction ={"predicted_label": label_map[output[0]]}
        return prediction

         
        
