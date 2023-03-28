# MastersThesis
This is my master's thesis source codes, I will update description and readme at the end of project for further usage

# train
python train.py --img 416 --batch 16 --epochs 3 --data platebowl.yaml --weights yolov5s.pt --workers 1

# infer
python detect.py --data platebowl.yaml --weights best.pt --source groundtruthtest.jpg

# resnet50 model
https://drive.google.com/file/d/1Qq6NY-RgpHZvylaJb2sVMIiDA6dkUr_d/view?usp=sharing

# food101-dataset
https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz