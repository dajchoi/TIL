
## Starting with Image

#### 데이터 불러오기
+ 그 외에 따로 동일한 세트의 난수 생성을 위해 np.random.seed()를 사용합니다
+ reshape에서 -1의 의미는 (원래 배열의 길이, 그리고 남은 차원으로부터 추정합니다) 

pd.read_csv(경로, index_col=칼럼이름)
#### Reading an image 
cv2.imread() --> full path를 사용하여 read image 또는 working directory에서의 데이터 불러옵니다
cv2.IMREAD_COLOR  --> loads a color image
    IMREAD_GRAYSCALE  --> loads image in grayscale mode
    IMREAD_UNCHANGED  --> loads image including alpha channel
    
    또는 따로 imread를 사용하여 위와 같은 3가지 경우를 -1,0,1로 second argument에서 설정해서 넣습니다
#### Writing an image
cv2.imwrite(파일 이름, img) -

*Showing an image
대게 matplotlib을 사용하여 image를 시각화합니다. 
from matplotlib import pyplot as plt를 하게 되면 
두가지 경우로 나눌 수 있게 됩니다 
