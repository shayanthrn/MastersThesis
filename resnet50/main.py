import torch
from torch import nn
from torchvision import datasets, transforms, models
import numpy as np


from collections import defaultdict
import os
from shutil import copy
def prepare_data(filepath, src, dest):
  classes_images = defaultdict(list)
  with open(filepath, 'r') as txt:
      paths = [read.strip() for read in txt.readlines()]
      for p in paths:
        food = p.split('/')
        classes_images[food[0]].append(food[1] + '.jpg')

  for food in classes_images.keys():
    print("\nCopying images into ",food)
    if not os.path.exists(os.path.join(dest,food)):
      os.makedirs(os.path.join(dest,food))
    for i in classes_images[food]:
      copy(os.path.join(src,food,i), os.path.join(dest,food,i))
  print("Copying Done!")

# print("Creating train data...")
# prepare_data('../datasets/food-101/meta/train.txt', '../datasets/food-101/images', '../datasets/food-101/train')
# print("Creating test data...")
# prepare_data('../datasets/food-101/meta/test.txt', '../datasets/food-101/images', '../datasets/food-101/test')


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

# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.TenCrop(224),
                                      transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                      transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(crop) for crop in crops]))])

test_data = datasets.ImageFolder("../datasets/food-101/test", transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size= 2, shuffle = True)

with torch.no_grad():
  for data, target in test_loader:
    data, target = data.cuda(), target.cuda()
    print(target)
    print("****")
    ## For 10-crop Testing
    bs, ncrops, c, h, w = data.size()
    # forward pass: compute predicted outputs by passing inputs to the model
    temp_output = model(data.view(-1, c, h, w))
    output = temp_output.view(bs, ncrops, -1).mean(1)
    # calculate the batch loss
    loss = criterion(output, target)
    # update average test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    print(pred)
    print("------------")
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(len(target)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(len(classes)):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %.2f%% (%2d/%2d)' % (classes[i], 100 * class_correct[i] / class_total[i],
                                                         np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %.2f%% (%2d/%2d)' % (100. * np.sum(class_correct) / np.sum(class_total),
                                                      np.sum(class_correct), np.sum(class_total)))