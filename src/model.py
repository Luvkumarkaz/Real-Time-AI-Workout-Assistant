import torch.nn as nn

class WorkoutClassifier(nn.Module):
    def __init__(self,input_size,num_classes):
        super(WorkoutClassifier,self).__init__()
        self.network=nn.Sequential(
            nn.Linear(input_size,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64,num_classes)
        )

    def forward(self,x):
        return self.network(x)

if __name__=="__main__":
    model=WorkoutClassifier(input_size=99,num_classes=10)
    print(model)