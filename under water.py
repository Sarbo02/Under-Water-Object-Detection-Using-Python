

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/ultralytics/yolov5  # clone repo
# %cd yolov5
# %pip install -qr requirements.txt # install dependencies
# %pip install -q roboflow

import torch
import os
from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="IkvwJ4W1q1izN5aroJDU")
project = rf.workspace("under-water-object-detection").project("under-water-object-detection-3vl35")
dataset = project.version(6).download("yolov5")

os.environ["DATASET_DIRECTORY"] = "/content/yolov5/pjtyolo5-3"

!python train.py --img 416 --batch 16 --epochs 250 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir runs

!python detect.py --weights /content/yolov5/runs/train/exp/weights/best.pt --img 416 --conf 0.1 --source {dataset.location}/test/images

import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")

