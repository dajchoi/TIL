import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # GPU에서 실행하려면 "cuda:0"

# N은 배치 크기, D_in은 입력의 차원, H는 은닉층의 차원, D_out은 출력 차원을 가리킵니다.
N, D_in, H, D_out = 64, 1000, 100, 10


# torch.randn: Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution

\text{out}_{i} \sim \mathcal{N}(0, 1)
out 
i
​	
 ∼N(0,1)
 
# 무작위의 입력과 출력 데이터를 생성합니다.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 무작위로 가중치를 초기화합니다.
w1 = torch.randn(D_in, H, device=device, dtype=dtype) # (D_in, H)
w2 = torch.randn(H, D_out, device=device, dtype=dtype) # (H, D_out)

# 학습률을 설정해줍니다. 
learning_rate = 1e-6
for t in range(500):
    # 순전파 단계: 예측값 y를 계산합니다.
    # torch.mm : matrix multiplication  해서 실행할 두 matrix 넣는 경우 / 특정 torch x1 에 관해 mm 실행할 하나의 특정 matrix, x2 넣는 경우 
    h = x.mm(w1) # 후자에 해당합니다. 
    h_relu = h.clamp(min=0) # 특정 범위안에 속하게 설정합니다. min max 둘다 넣는 경우도 존재합니다. torch.clamp(특정 torch, min, max)
    y_pred = h_relu.mm(w2)

    # 손실(loss)을 계산하고 출력합니다.
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # 손실에 따른 w1, w2의 변화도를 계산하고 역전파합니다.
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred) # .t()는 transpose의 의미
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 경사하강법(gradient descent)를 사용하여 가중치를 갱신합니다.
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
