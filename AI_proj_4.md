#### 평가 지표 / ERROR의 이해
###### 1. MAPE/nMAE              
(1) MAPE = Mean Absolute Percentage Error
:1/n * sum of ( |f_t - a_t| / a_t )

(2) NMAE = Normalized Mean Absolute Error (*정규화된*)
:(1/n * sum of ( |f_t - a_t| )) / (1/n * sum of a_t)
= (sum of ( |f_t - a_t| )) / (sum of a_t)
즉 a_t에도 적용

###### 2. MAE 
MAE = Mean of Absolute Error
:1/n * sum of ( |f_t - a_t| )

###### 3. RMSE
RMSE = Root of Mean Squared Error
:sqrt(1/n * sum of (f_t - a_t)^2)
RMSE는 큰 오류값 차이에 대해서 크게 패널티를 주는 이점이 있습니다.
즉, 특이값에 대해 크게 흔들리지 않습니다. 

