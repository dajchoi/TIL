### More About Augmentations
이미지를 학습시킬 때 중요한 것 중하나는 image의 형태변환입니다. 
TorchVision이 torchvision.transforms를 제공해주기에 이를 이미지에 적절하게 적용할 수 있습니다.
-
OneOf를 사용하여 list안에 있는 transform들 중 하나를 random하게 가져올 수 있습니다.  이는 부여한 확률에 따라 선택되는 확률도 달라집니다.
*oneof([  ], p=0.5)*
--> 0.5의 확률로 해당 transform을 스킵한다는 뜻입니다. 
만약 리스트 안에 3개의 transform이 존재하ㄴ다면 각 transform들은 1/3 * 1/2 = 1/6, 즉 1/6의 확률로 선택이 되는 것입니다. 

이 안에는 다양한 Augmentation기법들이 있습니다.
##### Resize
##### Crop - (1) center crop (2) random crop
##### Flip - (1) Vertical (2) Horizontal
##### Rotate - certain degrees provided

이 다음에는 ToTensor로 변형하여 Normalization의 과정을 거칩니다.


대게 미리 학습된 모델 pretrained model은 인풋으로 같은 방식으로 정규화된 이미지를 받습니다.
(C: channel, H: height, W: width) C는 대게 3 (3 channels) H와 W는 최소한의 224의 수를 부여받습니다.

이미지는 0과 1의 범위 내에서 받아들여집니다. [0,1]
그리고 보통 totensor이후에 오는 이미지는 mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]로 정규화 방식을 거칩니다. 

_________________________________________

### 실제로 Image Dataset에 대한 평균과 표준편차 구하기
대게 이미지는 주변환경에 따라 그 명도나 채도들이 서로 다릅니다. 

이미지를 동일한 환경으로 맞춰줍니다. 후처리로 전체이미지에 대한 화소값의 평균.ㅍ준편차 구해 이미리에 일괄적으로 적용합니다. 
transoform = transforms.Compose([transforms.ToTensor()])를 거친 데이터셋을 받고 

mean = dataset.train_data.mean((axis=(0,1,2))
std = dataset.train_data.std((axis=(0,1,2))을 구합니다.
mean =/ 255.
std =/ 255.

*augmentations 지원 라이브러리 -> albumentations (속도가 빠름)

