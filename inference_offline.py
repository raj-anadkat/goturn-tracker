import torch
from torchvision import models, transforms
from torchvision.transforms import functional as F
import torchvision.models as models
from objectTrackerModel import ObjectTrackerModel
from image_processor import ImageProcessor

data_folder = 'data'
annotations_file = 'data/groundtruth.txt'

# load model
model = ObjectTrackerModel()
model.load_state_dict(torch.load('weights/model.pth'))
model.eval()

# call image processor to apply cropping and transformations
processor = ImageProcessor(annotations_file, data_folder, model)

# to start and give the model the first bbox, give an index
index = 0
while index < 252:
    original_image, gt_bbox, pred_bbox = processor.process_image(index)
    
    success = processor.display_image_with_bbox("Frame", original_image, gt_bbox, pred_bbox)
    if not success:
        break
    index += 1
