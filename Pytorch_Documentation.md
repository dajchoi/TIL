### Pytorch 정의
*과학 연산 패키지 (python base), 딥러닝 연구 platform*
-NumPy로 대체하면서, GPU를 이용하여 연산합니다
-최대한의 유연성과 속도를 제공합니다.

Tensor := NumPy의 ndarray, GPU를 사용한 연산 가속도 가능합니다.

#### 행렬 생성? 
import torch 
x = torch.empty(a,b) # (a,b)의 shape을 가진 초기화되지 않은 행렬을 생성함.
x = torch.rand(a,b) # " 초기화된 행렬을 생성함.
x = torch.zeros(a,b,dtype=torch.long) # dtype이long이고 0으로 채워진 행렬을 생성함.
x = torch.tensor([]) # 직접적으로 데이터로부터 tensor를 생성함.
x = x.new_ones(a,b,dtype=torch.double) # 크기를 받는 메소드 똑같이 (a,b)의 shape을 가짐. 
x = torch.randn_like(x,dtype=torch.float) # dtype을 오버라이드, 결과는 동일한 크기를 가짐. 

##### torch의 Size? 
torch.size()로 tuple 타입입니다. 
모든 튜플 연산을 지원합니다.

###### 튜플 연산에는 add가 있습니다. 
torch.add(x,y) = y.add_(x) = x.add_(y)
torch.add(x,y,out=result) # 그러면 아웃풋으로 result의 변수가 생성됨. 
###### torch의 크기(size)나 모양(shape) 변경을 위한 메소드
view()

ex) x = torch.randn(4,4)
y = x.view(16)
z = x.view(8,-1) # -1은 다른 차원에서 유추, 그리고 그 결과 2의 값을 가지게 됨. 

텐서에 하나의 값만 존재한다면 .item()을 사용하여 그 값을 얻어낼 수 있습니다. 

#### torch to numpy to torch
여기서 주의할 부분은 torch tensor가 CPU상에 있다면 Torch Tensor와 NumPy배열은 메모리 공간을 공유하기에 하나를 변경하게되면 다른 하나도 변경된다는 것입니다. 
###### torch -> numpy 
b = a.numpy()

###### numpy -> torch
a = torch.from_numpy(a)

##### CUDA Tensors 를 이용한 장치 이해
if torch.cuda.is_available():
  device = torch.device("cuda") # GPU사용함
  y = torch.ones_like(x,device=device)
  x = x.to(device)
  z = x + y
  print(z)
  print(z.to("cpu",torch.double)) # dtype도 함께 변경함
  
  ________________________________________________________
  
 #### Autograd (자동 미분) 
 :tensor의 모든 연산에 대해 자동미분을 제공합니다.
 ##### *torch.Tensor*
 *.requires_grad = True*
 모든 연산을 추적합니다. 
 그리고 그 계산을 완료 한 다음 
 *.backward()* 
 모든 변화도(gradient)를 
 자동으로 계산합니다. 
 
 연산에 대한 기록추적을 중단하기 위해서는 
 (1) .detach()의 방법
 (2) with torch.no_grad()의 블록 생성
 변화도는 필요치 않습니다.
 
 requires_grad=True 설정 시에
 학습 가능한 매개변수를 갖는 모델을 평가(evaluate)할 때 유용합니다. 
 
 ##### *Function*
 .grad_fn : 연산의 결과로 생성된 값입니다
 requires_grad=True 설정한 다음 .grad_fn을 부를 수 있습니다.
 
 변화도 (Gradient)
 역전파 (Backprop) 
 
  ________________________________________________________
  
 #### Classfier 분류기
 데이터를 다루는 법:
 이미지, 텍스트, 오디오, 비디오 데이터를 표준 Python 패키지에 포함된 NumPy배열로 불러오면 됩니다.
 그런 다음 torch.Tensor로 변환 시켜 줍니다.
 
 데이터의 종류에 따라 유용하게 쓰이는 패키지 또한 다릅니다. 
 이미지 : Pillow OpenCV
 오디오 : SciPy LibROSA
 텍스트 : Python / Cython, NLTK / SpaCy
 영상/비디오 : torchvision -> ImageNet, CIFAR10, MNIST
 
 .datasets           ㅡㅡㅜ
 data transformer    ㅡㅡㅡ>    torch.utils.data.DataLoader
 
 ##### 이미지 분류의 학습 단계

1. 학습용/시험용 데이터셋 불러와 정규화(Normalizing)를 진행합니다. 
2. 합성곱 신경망(CNN)을 정의합니다.
3. 손실함수를 정의합니다.
4. 학습용 데이터를 사용하여 신경망을 학습합니다.
5. 시험용 데이터를 사용하여 신경망을 검사합니다. 
*학습용 데이터는 신경망 학습을 뜻하고
*시험용 데이터는 신경망 검사를 뜻합니다. 


우선 1의 source code로는 
import torchvision.transforms as transforms를 할 경우

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root=  , train=True, download=True, transform=transform) #위에 것 사용
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root=  , train=False, download=True, transform=transform) #위에 것 사용대신 테스트용이므로 train은 False로
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes= () #이미지에 따른 분류 클래스 , 나중에 매칭을 통해 정확도 계산이 가능해짐
