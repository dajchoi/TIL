### Keypoint RCNN (detection) 
(=Human Pose Detection) 
Pytorch를 사용한 Keypoint RCNN에 대해서 알아보겠습니다. 


기본적으로 받는 값 input 그리고 도출되는 값 output에 대해 나타낸다면
## input: image tensor
[batch size * num_channels * h * w]
batch size는 대체로 1
h : height, w : width 

## output: list of dict
 ##### -boxes (FloatTensor) [N, 4]
 [x1, y1, x2, y2]
 x는 0 과 w 사이
 y는 0 과 h 사이
 
 ##### -labels (Int64Tensor) [N]
 predicted labes for each image
 
 ##### -scores (Tensor) [N]
 scores or each prediction
 
 ##### -keypoint (FloatTensor) [N, K, 3]
 locations of the predicted keypoints
 [x, y, v]
 + keypoint를 잇기 위해서는 definction of the pairs of keypointss that we need to join이 필요함
 
 
 필요한 library는 두가지가 있겠습니다.
 import cv2
 import matplotlib 
 
 
 
