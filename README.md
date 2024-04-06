# goturn-tracker
Tracker inspired from goturn's architecture and method for appearance based object tracking. This is the Pytorch implementation with some tweaks.

# Vehicle Tracking Algorithm using Regression Networks

The objective of this  is to develop an algorithm capable of tracking a vehicle across a sequence of images, a task falling under single object detection. Given the nature of the problem, capturing both spatial and temporal information of object states is crucial.

## Approach Overview

Expanding on the idea of comparing similar frames, we opted for a strategy based on regression networks. Instead of tracking each frame independently, the network is provided with both the frame at t-1 (the previous frame) and the frame at t (the current frame). For each frame, the region around the target object is cropped, with the frame at t centered around the previous crop region. Additionally, for frame t, a context factor is incorporated to include background information, providing the network with context on potential object movement.

These cropped regions from both frames are then passed through a lightweight pretrained backbone - AlexNet, functioning as a feature extractor. The subsequent layers capture intricate relationships between various attributes, such as motion and appearance. Finally, a last layer regresses the normalized coordinates for the bounding box in the search region frame. The coordinates are passed through a series of transformations to get back to the original image scale.

This approach offers several advantages, including the network’s ability to learn patterns of object motion and appearance across frames. It’s worth noting that this concept was partly inspired by the efficiency of GOTURN, which achieves real-time object tracking at 100 frames per second.

## Network Architecture

![Network Architecture](https://github.com/raj-anadkat/goturn-tracker/assets/109377585/a2300f6e-4d18-46e8-92c8-214ce8fd1f9d)

## Model Training

The model was trained for 25 epochs with a batch size of 4. Employing the Adam optimizer alongside a learning rate scheduler, adjustments to the learning rate were made every 5 epochs by a factor of 0.1 based on the training’s progress. An initial learning rate of 0.0001 was chosen due to convergence issues with higher values.

The training and test loss plots displayed below illustrate that while the training loss continued to decrease, the test loss reached a plateau, indicating a limitation in further reduction. Initially, the model displayed signs of overfitting, with the test loss increasing. However, this was addressed by introducing dropout layers in the Fully Connected Network (FCN) and fine-tuning the learning rate, resulting in improved performance.
![Model Training](https://github.com/raj-anadkat/goturn-tracker/assets/109377585/2fc2c306-de98-49b4-bbe4-fd0159da3c03)

## Visualizing Outputs

The illustrations below demonstrate the model’s ability to predict the bounding box of the next frame based on the previous frame cropped around its bounding box and the current frame. The red box denotes the predicted bounding box, while the green box represents the ground truth. Overall, the model’s performance on the test set appears satisfactory considering the constraints of our limited dataset.

![Visualizing Outputs](https://github.com/raj-anadkat/goturn-tracker/assets/109377585/7cd1fcd7-f683-45a8-bb5c-923adac0d808)

## Transforming Predictions back to the Original Image

The predictions, initially in the frame of reference of the search region, undergo a multi-step process for normalization. Firstly, they are denormalized relative to the search region. Subsequently, they are rescaled to match the dimensions of the previous image crop. This rescaling is facilitated by tracking the bounding box coordinates of the context region. Finally, they are adjusted by subtracting from the coordinates of the final image dimensions, ensuring alignment within the overall image frame.

![Transforming Back](https://github.com/raj-anadkat/goturn-tracker/assets/109377585/90571013-2747-4145-acb6-3619c5cc8ae4)


## Conclusion

This project presents an algorithmic approach for vehicle tracking using regression networks, leveraging both spatial and temporal information across frames. While the model shows promising results within the constraints of the dataset, further improvements are expected with access to larger and more diverse datasets.

