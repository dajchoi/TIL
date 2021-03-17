######## Semantic Segmentation

##### Semantic Image Segmentation 이란?
### 이미지에서 각 pixel을 분류하는 것이다. 이를 통해 객체들은 각 특성에 따라 분할된다. 결국은 신경망을 pixel-wise mask로 나타낼 수 있겠끔 거치는 일련의 과정이라 일컬을 수 있겠다. 
### (=모든 픽셀의 label을 예측)
### + _Instance Segmentation_ 과는 다르다. Instance Segmentation은 class의 instance까지 구별해준다. 

##### 적용되는 분야
### 자율주행이나 의료영상에 흔히 적용되고 있다. 이외에도 객체의 위치판독, 모양 분류, 또는 더 광범위하게 객체 자체에서의 분류까지 다양한 목적으로 사용되고 있다고 보면 되겠다. 

##### 과정
### 일반적인 과정은 일반적으로 Encoder과 Decoder의 작용으로 정의를 내릴 수 있다. 
### Encoder: pre-trained classification network (학습된 모델-VGG/ResNet) 
### Decoder: Encoder에서의 구분된 특징들을 픽셀별로 구분지어 더 정밀한 분류를 거치는 단계 
### Encoder와 Decoder의 작용에도 다른 접근 방식이 있다. 
#### 1. Region-based Semantic Segmentation
### object detection을 기반으로 CNN과정을 거치는 방식 (유의미한 부분만 따로 추려내 CNN을 실행시키는 것)
## 이미지 입력 - 후보 영역 추출 - CNN 특징 계산 - 영역 분류  
## 기존의 CNN에서 후보 영역을 추가로 추출하는 것이 Region-Based CNN의 주요한 특징이라 볼 수 있겠다. 따라서 기존의 AlexNet, VGG, GoogleNet, ResNet에 위에다 덧붙여서 사용이 가능하다. 비교적 섬세하게 pixel-mask를 구현할 수 있는 반면에 time cost라는 단점이 존재한다. 
#### 2. Fully-Conventional Network-based Semantic Segmentation
### 기존의 특징을 추출하는 것이 아닌 pixel-by-pixel로 mapping을 거치는 방식이다. Convolutional 과 Pooling 의 layer 만으로 진행되는 방식인데 downsampling의 문제점이 발생할 수 있다. 
#### 3. Weakly Supervised Semantic Segmentation 
### object label로만 object localization을 하는 방식이다. 실제로 labeling 된 데이터가 있는 경우가 많지 않고 bounding box, segmentation annotation을 만들어주는 과정이 수고스럽기 때문에 이를 대비하여 고사한 방식이라 할 수 있겠다. 



