#### 평가 지표 / Regression ERROR의 이해
###### 1. MAPE/nMAE              
(1) MAPE = Mean Absolute Percentage Error
:1/n * sum of ( |f_t - a_t| / a_t )
MAPE는 MAE를 percentage로 변환한 것입니다. MAE와 비슷한 특징을 띄며 모델이 편향될 우려가 있습니다.
robust하다고 표현합니다. robust란 이상치에 대한 저항도 가지고 있고, 데이터 특성을 잘 나타냅니다. 

(2) NMAE = Normalized Mean Absolute Error (*정규화된*)
:(1/n * sum of ( |f_t - a_t| )) / (1/n * sum of a_t)
= (sum of ( |f_t - a_t| )) / (sum of a_t)
\\즉 a_t에도 적용

###### 2. MAE 
MAE = Mean of Absolute Error
:1/n * sum of ( |f_t - a_t| ) 
MAE는 직관성이 높습니다. 
하지만 underperformance인지 overperformance인지 구별할 수 없습니다.
underperformance란 모델이 실제보다 낮은 값으로 예측
overperformance란 모델이 실제보다 높은 값으로 예측
한 것을 말합니다. 

###### 3. RMSE
RMSE = Root of Mean Squared Error
:sqrt(1/n * sum of (f_t - a_t)^2)
RMSE는 큰 오류값 차이에 대해서 크게 패널티를 주는 이점이 있습니다.
즉, 특이값에 대해 크게 흔들리지 않습니다. 

