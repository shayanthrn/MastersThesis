import torch
from torch import nn
from torchvision import datasets, transforms, models
from PIL import Image

import warnings
warnings.filterwarnings("ignore")


checkpoint = torch.load("./resnet50model.pth", map_location='cpu')
model = models.resnet50(pretrained=False)
classifier = nn.Linear(2048, 101)
model.fc = classifier
model.load_state_dict(checkpoint['model_state'], strict=False)
criterion = nn.CrossEntropyLoss()

with open('../datasets/food-101/meta/classes.txt', 'r') as txt:
    classes = [l.strip() for l in txt.readlines()]

#move model to gpu
model.cuda()
model.eval()

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.TenCrop(224),
                                      transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                      ])

test_data = datasets.ImageFolder("../datasets/food-101/test", transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size= 2, shuffle = True)
with torch.no_grad():
    img = Image.open(r'test.jpg')
    img = test_transforms(img)
    img = img.unsqueeze(0)
    img = img.cuda()
    bs, ncrops, c, h, w = img.size()
    temp_output = model(img.view(-1, c, h, w))
    output = temp_output.view(bs, ncrops, -1).mean(1)
    _, pred = torch.max(output, 1)  
    print(f"detected category is {classes[pred.item()]}")