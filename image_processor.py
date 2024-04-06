import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

"""
Helper class for inference
"""

class ImageProcessor:
    def __init__(self, annotations_file, data_folder, model):
        self.annotations = np.loadtxt(annotations_file, delimiter=',')
        self.data_folder = data_folder
        self.model = model
        self.output_size = (227, 227)
        self.context_factor = 2.0
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])

    def crop_and_resize(self, image, bbox, context_factor, output_size, return_context=False):
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin

        # adjusting the bounding box to add context
        context_width = width * (context_factor - 1)
        context_height = height * (context_factor - 1)
        xmin_context = max(int(xmin - context_width / 2), 0)
        ymin_context = max(int(ymin - context_height / 2), 0)
        xmax_context = min(int(xmax + context_width / 2), image.shape[1])
        ymax_context = min(int(ymax + context_height / 2), image.shape[0])

        # crop and resize
        cropped_image = image[ymin_context:ymax_context, xmin_context:xmax_context]
        resized_image = cv2.resize(cropped_image, output_size, interpolation=cv2.INTER_LINEAR)

        if return_context:
            return resized_image, (xmin_context, ymin_context, xmax_context, ymax_context)
        return resized_image

    def process_image(self, index):
        image_files = [os.path.join(self.data_folder, f'{index + 1:08}.jpg') for index in range(len(self.annotations))]
        image_current = cv2.imread(image_files[index])
        image_next = cv2.imread(image_files[index + 1])
        image_current = cv2.cvtColor(image_current, cv2.COLOR_BGR2RGB)
        image_next = cv2.cvtColor(image_next, cv2.COLOR_BGR2RGB)
        original_image_next = image_next
        original_image_next = cv2.cvtColor(original_image_next, cv2.COLOR_RGB2BGR)

        image_current = image_current.astype(np.float32) / 255.0
        image_next = image_next.astype(np.float32) / 255.0

        # convert to x1 y1 x2 y2 format (top left and bottom right)
        bbox = self.annotations[index]
        xmin, ymin, xmax, ymax = min(bbox[0::2]), min(bbox[1::2]), max(bbox[0::2]), max(bbox[1::2])

        # proceed with cropping and reziing to match alexnets input dimensions
        target_crop = self.crop_and_resize(image_current, (xmin, ymin, xmax, ymax), 1.0, self.output_size)
        search_region_crop, (xmin_context, ymin_context, xmax_context, ymax_context) = self.crop_and_resize(image_next, (xmin, ymin, xmax, ymax), self.context_factor, self.output_size, return_context=True)

        # calculate scale factors based on the cropped (before resize) and target dimensions.
        original_width = xmax_context - xmin_context
        original_height = ymax_context - ymin_context
        target_width, target_height = self.output_size
        scale_w = target_width / original_width
        scale_h = target_height / original_height

        # adjust ground truth bbox relative to the search region and apply scale.
        gt_bbox_relative_scaled = (
            (xmin - xmin_context) * scale_w,
            (ymin - ymin_context) * scale_h,
            (xmax - xmin_context) * scale_w,
            (ymax - ymin_context) * scale_h
        )

        gt_bbox_relative_scaled = (
            gt_bbox_relative_scaled[0] / self.output_size[0],  # Normalize x_min
            gt_bbox_relative_scaled[1] / self.output_size[1],  # Normalize y_min
            gt_bbox_relative_scaled[2] / self.output_size[0],  # Normalize x_max
            gt_bbox_relative_scaled[3] / self.output_size[1]   # Normalize y_max
        )

        # convert everything to tensor.
        target_crop = F.to_tensor(target_crop)
        search_region_crop = F.to_tensor(search_region_crop)
        target_crop = self.normalize(target_crop)
        search_region_crop = self.normalize(search_region_crop)
        gt_bbox_relative_scaled = torch.tensor(gt_bbox_relative_scaled, dtype=torch.float32)

        target_crop = target_crop.unsqueeze(0)  # Add batch dimension
        search_region_crop = search_region_crop.unsqueeze(0)  # Add batch dimension

        # forward pass
        with torch.no_grad():
            outputs = self.model(target_crop, search_region_crop)

        outputs = outputs[0].cpu().numpy()
        output_bbox_relative = [coord * 227 for coord in outputs]

        # scale back to original image dimensions   
        scaled_to_original_w = output_bbox_relative[0]*0.5/ scale_w
        scaled_to_original_h = output_bbox_relative[1]*0.5 / scale_h
        scaled_to_original_wmax = output_bbox_relative[2]*1.1 / scale_w
        scaled_to_original_hmax = output_bbox_relative[3]*1.1 / scale_h

        # translate the coordinates to the original image frame
        final_xmin = scaled_to_original_w + xmin_context
        final_ymin = scaled_to_original_h + ymin_context
        final_xmax = scaled_to_original_wmax + xmin_context
        final_ymax = scaled_to_original_hmax + ymin_context

        output_bbox_absolute = [final_xmin,final_ymin, final_xmax,final_ymax]

        return original_image_next, (xmin, ymin, xmax, ymax), output_bbox_absolute

    def display_image_with_bbox(self, window_name, original_image, gt_bbox, pred_bbox):
        # ground truth bounding box on the original image
        gt_image = cv2.rectangle(original_image.copy(), (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 0, 255), 2)

        # predicted bbox
        pred_image = cv2.rectangle(gt_image, (int(pred_bbox[0]), int(pred_bbox[1])), (int(pred_bbox[2]), int(pred_bbox[3])), (0, 255, 0), 2)

        text = "Tracker"
        org = (int(pred_bbox[0]), int(pred_bbox[1]) - 10)  # Adjust position of text above the bounding box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)  # White color for the text
        thickness = 1
        cv2.putText(pred_image, text, org, font, font_scale, color, thickness)

        cv2.imshow(window_name, pred_image)

        cv2.waitKey(1)

        return True