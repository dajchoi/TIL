## Data Preprocessing & Augmentation

#### Augmentation 이란?
새로운 데이터를 요구하지 않고도 데이터를 증강하는 기법입니다. 이 방법에는 Noise삽입, 색상, 밝기 변형 등이 있습니다. 
### Data Augmentation의 도구: Keras의 Image Generator
학습 도중에 이미지에 임의 변형 및 정규화를 적용 --> 이를 배치 단위로 불러올 수 있는 Generator를 생성해줍니다. 
이미지에 줄 수 있는 변화에는 다양한 방법이 존재합니다. 
*rotation_range width_shift_range height_shift_range brightness_range zoom_range horizontal_flip vertical_flip rescale 등*
+ 참고자료 1) https://deepestdocs.readthedocs.io/en/latest/003_image_processing/0030/ 
+ 참고자료 2) https://libertegrace.tistory.com/entry/3-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A6%9D%EA%B0%95%EA%B8%B0%EB%B2%95-Data-Augmentation
주의점은 실제로 모델에 적용할 때 transformation같은 경우는 train set에만 적용해도 되지만, resizing/rescaling은 train, test set모두 적용해줘야 된다는 점입니다. 
#### 더 구체적으로
일반적으로 데이터의 양에 제한이 있을 경우 그리고 그 안에서 최고의 성능을 내야 할 경우 crop, 즉 잘라내는 방법을 많이 이용합니다. 
반면, CNN자체가 크기가 커지면 회전이 적용된 이미지도 잘 분류할 수 있기에 회전의 방법은 많이 쓰지 않습니다.
색감보정에는 색 대조를 높이거나 각 R, G, B에 특정 값을 더해주어 색감 조정해주는 방법도 있습니다.
Noise삽입은 거치 데이터를 훈련하기 전에 데이터에 변형을 줄 수 있는 방법인데, 이를 통해 때에 따라 성능이 좋아질 수도 나빠질 수도 있습니다. 
데이터에 필수적인 특징을 찾아냈을 경우 순기능을 발휘한 것이고
반대로 기존 데이터가 이미 편향돼 있는 상태였을 경우 성능이 오히려 나빠지는 악기능 발휘됩니다. 
 
##### Scaling --> 스케일링
이미지가 0 에서 255사이의 값으로 표현되는 보통 255로 나누어줍니다. 그러면 [0,1]에서의 값으로 나타나지게 됩니다. 또는 [-1,1]에서의 값으로 나타내어지게 할 수 있습니다. 어떤 범위에 맞추는지에 따라 sigmoid 또는 tanh의 함수의 값으로 비교할 수 있게 됩니다. 

#### Data Preprocessing (Image)는 어떻게?
##### 과정
일반적으로 그 과정을 설명할 때에는 
데이터를 인풋으로 받고 이미지를 리사이즈 시켜주고 Noise를 제거해주고 segmentation을 거치고 모서리를 더 완만하게 만들어줍니다.
Segmentation은 상황에 따라 필요성의 정도가 달라집니다. 
