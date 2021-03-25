
### Pytorch와 Tensorflow의 차이점

Pytorch: Facebook에서의 제작
##### framework for doing fast computation written in C (C프로그램을 구현) 

Tensorflow: Google에서의 제작
##### how you organize and perform operations on data
--> Session object: class for running tensorflow operations 

### Pytorch tensor와 Numpy의 차이점

Pytorch Tensor: GPU사용 => 수치연산 가속화

Numpy: 딥러닝, 연산그래프, 변화도를 알 수 있음


Tensor-> can run on GPU, 
1. can be created directly from data
2. can be created from numpy arrays

(1)import torch
x_data = torch.tensor(data)

(2)import numpy as np
x_np = torch.from_numpy(np_array)


#### new tensor를 만들 경우 (shape, datatype)과 함께 만들어짐
1) x_ones = torch.ones_like(x_data)
2) x_rand = torch.rand_like(x_data, dtyple=torch.float)

+ default로 tensors are created on CPU,
tensors를 GPU로 돌리고 싶은 경우 ->
if torch.cuda is_available():
  tensor = tensor.to('cuda')

tensors를 서로 붙이고 싶은 경우 joining tensors
1) torch.cat([tensor,tensor],dim=1)
2) torch.stack 
