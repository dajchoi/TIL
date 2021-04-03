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
