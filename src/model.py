model_f= models.resnet34(weights=True)
num_ftrs = model_f.fc.in_features
model_f.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs,512),
    nn.ReLU(),
    nn.Linear(512,256),
    nn.ReLU(),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Linear(128,5)
    )

model_f = model_f.to(device)