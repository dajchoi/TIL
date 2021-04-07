Rule of thumb for “in_channels” on your first Conv2d layer:
— If your image is black and white, it is 1 channel. (You can ensure this by running transforms.Grayscale(1) in the transforms argument of the dataloader.)
— If your image is color, it is 3 channels (RGB).
— If there is an alpha (transparency) channel, it has 4 channels.

What about the out_channels you say? That’s your choice for how deep you want your network to be. Basically, your out_channels dimension, defined by Pytorch is:
out_channels (int) — Number of channels produced by the convolution

Note: The value of kernel_size is custom, and although important, doesn’t lead to head-scratching errors, 
so it is omitted from this tutorial. Just make it an odd number, typically between 3–11, 
but sizes may vary between your applications.

 torch.Size([ , , , , ])
    # 1d: [batch_size] 
    # use for target labels or predictions.

    # 2d: [batch_size, num_features (aka: C * H * W)]
    # use for nn.Linear() input.

    # 3d: [batch_size, channels, num_features (aka: H * W)]
    # when used as nn.Conv1d() input.
    # (but [seq_len, batch_size, num_features]
    # if feeding an RNN).

    # 4d: [batch_size, channels, height, width]
    # use for nn.Conv2d() input.

    # 5D: [batch_size, channels, depth, height, width]
    # use for nn.Conv3d() input.

"""The in-between dimensions are the hidden layer dimensions, you just pass in the last of the previous as the first of the next."""

The very last output, aka your output layer depends on your model and your loss function. 
If you have 10 classes like in MNIST, and you’re doing a classification problem, 
you want all of your network architecture to eventually consolidate into those final 10 units 
so that you can determine which of those 10 classes your input is predicting.


____________________________________________________________________________________________________________________________________________________
forward 함수만 정의하고 나면, (변화도를 계산하는) backward 함수는 autograd 를 사용하여 자동으로 정의됩니다. 
forward 함수에서는 어떠한 Tensor 연산을 사용해도 됩니다.


torch.Tensor - backward() 같은 autograd 연산을 지원하는 다차원 배열 입니다. 또한 tensor에 대한 변화도(gradient)를 갖고 있습니다.

nn.Module - 신경망 모듈. 매개변수를 캡슐화(encapsulation)하는 간편한 방법 으로, GPU로 이동, 내보내기(exporting), 
                  불러오기(loading) 등의 작업을 위한 헬퍼(helper)를 제공합니다.

nn.Parameter - Tensor의 한 종류로, Module 에 속성으로 할당될 때 자동으로 매개변수로 등록 됩니다.

autograd.Function - autograd 연산의 전방향과 역방향 정의 를 구현합니다. 
                           모든 Tensor 연산은 하나 이상의 Function 노드를 생성하며, 각 노드는 Tensor 를 생성하고 
                           이력(history)을 부호화 하는 함수들과 연결하고 있습니다.



손실 함수는 (output, target)을 한 쌍(pair)의 입력으로 받아, 출력(output)이 정답(target)으로부터 얼마나 멀리 떨어져있는지 추정하는 값을 계산합니다.
output = net(input)
target = torch.randn(10)  # 예시를 위한 임의의 정답
target = target.view(1, -1)  # 출력과 같은 shape로 만듦
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

nn 패키지에는 여러가지의 손실 함수들 이 존재합니다. 간단한 손실 함수로는 출력과 대상간의 평균제곱오차(mean-squared error)를 계산하는 nn.MSEloss 가 있습니다.



이제 .grad_fn 속성을 사용하여 loss 를 역방향에서 따라가다보면, 이러한 모습의 연산 그래프를 볼 수 있습니다:

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
<MseLossBackward object at 0x7f520fa2ad30>
<AddmmBackward object at 0x7f520fa2ada0>
<AccumulateGrad object at 0x7f520fa2ada0>


가중치 갱신 코드: 새로운 가중치(weight) = 가중치(weight) - 학습률(learning rate) * 변화도(gradient)
---------------------------------------------------------------------------------------------------------------
신경망을 구성할 때 SGD, Nesterov-SGD, Adam, RMSProp 등과 같은 다양한 갱신 규칙을 사용하고 싶을 수 있습니다. 
이를 위해서 torch.optim 라는 작은 패키지에 이러한 방법들을 모두 구현해두었습니다. 

import torch.optim as optim

# Optimizer를 생성합니다.
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 학습 과정(training loop)에서는 다음과 같습니다:
optimizer.zero_grad()   # 변화도 버퍼를 0으로
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 업데이트 진행

_________________________________
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss (예제에 포함되는) 
