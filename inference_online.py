"""
Struggle ;(
"""

import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from objectTrackerModel import ObjectTrackerModel

data_folder = 'data'

# load the saved model parameters
model = ObjectTrackerModel()
model.load_state_dict(torch.load('weights/model.pth'))
model.eval()

index = 0
annotations_file = 'data/groundtruth.txt'
annotations = np.loadtxt(annotations_file, delimiter=',')
initial_bbox = annotations[index]
gt_init = [min(initial_bbox[0::2]), min(initial_bbox[1::2]), max(initial_bbox[0::2]), max(initial_bbox[1::2])]


class ImageProcessor:
    def __init__(self, data_folder, model,initial_bbox):


        self.data_folder = data_folder
        self.all_files = os.listdir(data_folder)
        self.image_files_count = len([file for file in self.all_files if file.endswith('.jpg')])
        self.model = model
        self.output_size = (227, 227)
        self.context_factor = 2
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        
        self.initial_bbox = initial_bbox
        self.prev_pred_bbox = None

    def process_image(self, index):

        image_files = [os.path.join(self.data_folder, f'{index + 1:08}.jpg') for index in range(self.image_files_count)]
        image_current = cv2.imread(image_files[index])
        image_next = cv2.imread(image_files[index + 1])
        image_current = cv2.cvtColor(image_current, cv2.COLOR_BGR2RGB)
        image_next = cv2.cvtColor(image_next, cv2.COLOR_BGR2RGB)
        original_image_next = image_next
        original_image_next = cv2.cvtColor(original_image_next, cv2.COLOR_RGB2BGR)


        # normalize images
        image_current = image_current.astype(np.float32) / 255.0
        image_next = image_next.astype(np.float32) / 255.0


        if self.prev_pred_bbox is not None:
            bbox = self.prev_pred_bbox
            xmin, ymin, xmax, ymax = bbox[0] , bbox[1], bbox[2], bbox[3]
        
        else:
            bbox = self.initial_bbox
            xmin, ymin, xmax, ymax = bbox

        target_crop = self.crop_and_resize(image_current, (xmin, ymin, xmax, ymax), 2, self.output_size)
        search_region_crop, (xmin_context, ymin_context, xmax_context, ymax_context) = self.crop_and_resize(image_next, (xmin, ymin, xmax, ymax), self.context_factor, self.output_size, return_context=True)

        # calculate scale factors based on the cropped (before resize) and target dimensions.
        original_width = xmax_context - xmin_context
        original_height = ymax_context - ymin_context
        target_width, target_height = self.output_size
        scale_w = target_width / original_width
        scale_h = target_height / original_height


        # convert everything to tensor.
        target_crop = F.to_tensor(target_crop)
        search_region_crop = F.to_tensor(search_region_crop)
        target_crop = self.normalize(target_crop)
        search_region_crop = self.normalize(search_region_crop)

        target_crop = target_crop.unsqueeze(0)  # Add batch dimension
        search_region_crop = search_region_crop.unsqueeze(0)  # Add batch dimension

        # pass the data through the model
        with torch.no_grad():
            outputs = self.model(target_crop, search_region_crop)

        outputs = outputs[0].cpu().numpy()
        output_bbox_relative = [coord * 227 for coord in outputs]

        scaled_to_original_w = output_bbox_relative[0] / scale_w
        scaled_to_original_h = output_bbox_relative[1]/ scale_h
        scaled_to_original_wmax = output_bbox_relative[2] / scale_w
        scaled_to_original_hmax = output_bbox_relative[3] / scale_h

        # translate the coordinates to the original image frame
        final_xmin = scaled_to_original_w + xmin_context
        final_ymin = scaled_to_original_h + ymin_context
        final_xmax = scaled_to_original_wmax + xmin_context
        final_ymax = scaled_to_original_hmax + ymin_context

        output_bbox_absolute = [final_xmin,final_ymin, final_xmax,final_ymax]

        self.prev_pred_bbox = output_bbox_absolute

        return original_image_next, (xmin, ymin, xmax, ymax), output_bbox_absolute
    

    def crop_and_resize(self, image, bbox, context_factor, output_size, return_context=False):
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin

        context_width = width * (context_factor - 1)
        context_height = height * (context_factor - 1)
        xmin_context = max(int(xmin - context_width / 2), 0)
        ymin_context = max(int(ymin - context_height / 2), 0)
        xmax_context = min(int(xmax + context_width / 2), image.shape[1])
        ymax_context = min(int(ymax + context_height / 2), image.shape[0])

        cropped_image = image[ymin_context:ymax_context, xmin_context:xmax_context]
        resized_image = cv2.resize(cropped_image, output_size, interpolation=cv2.INTER_LINEAR)

        if return_context:
            return resized_image, (xmin_context, ymin_context, xmax_context, ymax_context)
        return resized_image

    def display_image_with_bbox(self, window_name, original_image, pred_bbox):
        # draw the predicted bounding box on the original image
        pred_image = cv2.rectangle(original_image.copy(), (int(pred_bbox[0]), int(pred_bbox[1])), (int(pred_bbox[2]), int(pred_bbox[3])), (0, 255, 0), 2)
        
        cv2.imshow(window_name, pred_image)

        key = cv2.waitKey(100)
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return False

        return True


processor = ImageProcessor(data_folder, model,gt_init)

while index < 251:
    original_image, gt_bbox, pred_bbox = processor.process_image(index)
    gt_index_bbox = annotations[index+1]
    gt_index_bbox = [min(gt_index_bbox[0::2]), min(gt_index_bbox[1::2]), max(gt_index_bbox[0::2]), max(gt_index_bbox[1::2])]
    success = processor.display_image_with_bbox("Frame", original_image, pred_bbox)
    if not success:
        break
    index += 1
