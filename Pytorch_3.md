### 딥러닝 학습 향상을 위한 방법들
#### Pytorch기반_
##### 1. 가중치 감소 (Weight Decay)
목표: 가중치 정형화
의미: 가중치 값이 커질수록 overfitting 발생하는데, 이를 최소화하기위해 손실함수를 더해주는 것입니다. 
더해주는 값은 (1)L1정형화 (2)L2정형화로 결정됩니다.
pytorch --> optimizer = torch.optim.SGD(model.parameter(), lr=learning_rate, *weight_decay=0.1* )
weight_decay값을 너무 크게하면 반대로 언더피팅현상이 일어날 수 있으므로 주의해야 합니다.

##### 2. 드롭아웃 (Drop Out) 
목표: 오버피팅 억제
의미: 특정확률로 뉴런을 비활성화 시킴으로써 연산에 포함되지 않게 합니다. 
*학습이 끝나고 평가시에는 model.eval()이 필요합니다.*

mnist_train = dset.MNIST(
    './', 
    train=True, 
    transform = transforms.Compose([
        transforms.Resize(34),                        # 원래 28x28인 이미지를 34x34로 늘립니다.
        transforms.CenterCrop(28),                    # 중앙 28x28를 뽑아냅니다.
        transforms.RandomHorizontalFlip(),            # 랜덤하게 좌우반전 합니다.
        transforms.Lambda(lambda x: x.rotate(90)),    # 람다함수를 이용해 90도 회전해줍니다.
        transforms.ToTensor(),                        # 이미지를 텐서로 변형합니다.
    ]),
    target_transform=None,
    download=True
)

##### 3. 데이터 증강 (Data Augmentation) 
목표: overfitting 해소 / 테스트 데이터의 정확도 향상
의미: 알고리즘을 통해 훈련데이터의 수를 늘리는 방법입니다. 

##### 4. 가중치 초기화 (Weight Initialization)
목표: 가중치 편향 현상 방지
의미: 신경망이 깊어질수록 (1) 가중치값 분포 쏠림현상 (2) 특정값 부분으로 모이는 현상이 가능합니다. 이를 방지하기 위해 가중치를 적당한 선에 초기화해주는 것을 말합니다.

##### 5. 학습률 스케쥴러 (Learning rate scheduler)
목표: 낮은 손실값
의미: 상황에 맞게 학습률을 변경해주는 것을 뜻합니다.

##### 6. 학습 데이터 정규화 (Data Normalization)
목표: 높은 정확도
의미: input data를 공간상 분포를 정규화해주는 것을 의미합니다. 이 때 정규화하는 방법으로는 (데이터-평균)/표준편차로 볼 수 있습니다.

##### 7. 경사 하강법 (Gradient Descent)
의미: 최초의 손실값을 구하는 방식을 이동해주는 방법들을 경사하강법이라 합니다. 손실함수의 미분을 통한 기울기를 얻음으로써 경사하강법 적용이 가능합니다.
종류- (1) SGD (2) Adam  [3과 4의 조합]  
(3) Momentum (4) adagrad

##### 8. 배치 정규화 (Batch Normalization)
목표: 학습속도 개선 / 오버피팅 억제
-> 이로인해 1.가중치감소 과 2.드롭아웃의 필요성이 낮아집니다. 
의미: 각 신경망의 활성화 값의 분포가 적당히 퍼지도록 해주는 것을 의미합니다.
-> 데이터의 연산결과 평균과 분산이 1이 되록 재가공시켜주고 특정값을 곱하고 더하는 Scaling Shifting의 과정을 거치게 됩니다. 
*학습이 끝나고 평가시에는 model.eval()이 필요합니다.*
