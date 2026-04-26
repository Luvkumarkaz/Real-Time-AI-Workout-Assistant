import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset

class ExerciseDataset(Dataset):
    def __init__(self,root_dir):
        landmarks_path=os.path.join(root_dir,"landmarks.csv")
        labels_path=os.path.join(root_dir,"labels.csv")

        print(f"Loading data from {root_dir}...")
        self.landmarks_df=pd.read_csv(landmarks_path)
        self.labels_df=pd.read_csv(labels_path)

        self.data=pd.merge(self.landmarks_df,self.labels_df,on="pose_id")
        self.features=self.data.drop(columns=['pose_id','pose'])
        self.labels=self.data['pose']
        print(f"Features Columns: {list(self.features.columns)}")

        unique_labels=sorted(self.labels.unique())
        self.label_map={label:i for i, label in enumerate(unique_labels)}
        print(f"Classes Detected: {self.label_map}")

        self.X=torch.tensor(self.features.values,dtype=torch.float32)
        self.y=torch.tensor([self.label_map[l] for l in self.labels],dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        current_pose=self.X[idx]
        current_label=self.y[idx]

        landmarks=current_pose.reshape(33,3)
        center=(landmarks[23]+landmarks[24])/2.0
        centered_landmarks=landmarks-center

        max_dist = np.max(np.linalg.norm(centered_landmarks, axis=1))
        if max_dist < 1e-6: max_dist = 1
        normalized_landmarks = centered_landmarks / max_dist

        final_input=normalized_landmarks.flatten()
        
        return final_input,current_label

if __name__=="__main__":
    data_path=os.path.join("..","data","processed")
    try:
        dataset=ExerciseDataset(data_path)
        sample_x,sample_y=dataset[0]
        print(f"Sample input shape: {sample_x.shape}")
        print(f"Sample label ID: {sample_y}")
    except Exception as e:
        print(f"Error Loading Data: {e}")
