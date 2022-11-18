# Convolutional Neural Networks; CNNs

## 1. Deep Neural Networks; DNNs (깊은 신경망)
* 다층 퍼셉트론에 은닉층을 여러 개(일반적으로 5개 이상) 추가하면 깊은 신경망이 됨
  * 다층 퍼셉트론 : 특징추출 & 공간변환 => 정보를 추상화
  * 파라미터 개수가 많아지면 자유도도 올라가지만 학습 대상도 많아진다.
 
* 배경 : 80년대 초 깊은 신경망 발명 제안됨 -> BUT, 학습 실현 불가능
  * 1. 경사 소멸(gradient vanishing) 문제 => ReLU, 교차 엔트로피, 로그우도 등으로 해결
  * 2. 소수의 훈련 데이터 집합 => ImageNet과 같은 data set 으로 해결
  * 3. 과다한 연산과 시간 소요 => GPU의 등장으로 해결

* 기계학습의 패러다임 변화
  * 얕은 신경망(e.g. 다층 퍼셉트론)
    * 은닉층==특징 추출기, 범용적 근사함수
    * 얕은 신경망은 가공하지 않은 간단한 규칙만 학습(제한적 특징만 추출)하여 낮은 성능
    * 수작업으로 특징을 추출하는 과정이 필요함(특징 공학)
   
  * 깊은 신경망 (e.g. 합성곱 신경망)
    * raw data로부터 자동화된 특징 추출(data-driven features)하도록 학습 
    
        : 계층적인 표현 학습(representation learning) & 종단간 학습(end-to-end learning) 가능
    <img src="https://user-images.githubusercontent.com/97011426/202654232-a13af0de-76a3-417e-abb3-8aa035aae01b.png" width="60%" height="60%"/>

 
 * 깊은 신경망의 특징(표현)학습
 <img src="https://user-images.githubusercontent.com/97011426/202654502-ce929d32-632e-4cdc-badb-b5fb37a18e05.png" width="80%" height="80%"/>
 
  * 입력이 영상인 경우,
      * 낮은 단계 은닉층 (입력쪽): 선이나 모서리와 같은 간단한 (저급) 특징 추출
      * 높은 단계 은닉층 (출력쪽): 추상적인 형태abstractive representation의 복잡한 (고급) 특징을 추출
      
      → 표현 학습이 강력해짐에 따라 다양한 인공지능 응용에서 획기적인 성능 향상

## 2. Convolutional Neural Networks; CNNs (합성곱 신경망)
* 컴퓨터 비전 문제
  * 분류 classification
  * 검색 retrieval
  * 검출 detection
  * 분할 segmentation

* 컴퓨터 비전이 어려운 이유
  * 1. 관점의 변화 : 동일한 객체라도 카메라의 이동에 따라 모든 픽셀값이 변화됨
  
      -> 컴퓨터는 pixel 값으로 사진을 보기 때문
  * 2. 경계색(보호색)으로 배경과 구분이 어려운 경우
  * 3. 조명에 따른 변화
  * 4. 기형적인 형태의 영상 존재
  * 5. 일부가 가려진 영상 존재
  * 6. 같은 종류 간의 변화가 큼
  
* 과거 컴퓨터 비전 문제 해결 방법의 예
  * Scale Invariant Feature Transform(SIFT) 특징 추출
    * 색의 변화가 큰 곳 : edge 판별(by gradient) => 특징판별(사람이 결정)
    * 사람이 정리해둔 필터 사용 : 사람에 의한 정보 손실

* 합성곱 신경망
  * 구조적 특징
    * 은닉층은 높이height, 폭width, 깊이depth 단위로 처리
  * 기본 구성 요소
    * 합성곱 층(convolutional layer (CONV+RELU)) <- 사진의 지역성locality, 정상성stationarity 활용
    * 지역성(locality) : 주변의 픽셀값들(특징)이 비슷하다, 정상성(stationarity) : 특징이 반복(재생)되어 사용된다
    * 이러한 특징들을 은닉층에서 활용하지만 단순히 선형연산을 합성곱으로 대체한것일 뿐 (선형성과 비선형성은 유지)
    <img src="https://user-images.githubusercontent.com/97011426/202656171-24d5b487-d97b-4994-ac7e-17ac28d75259.png" width="60%" height="60%"/>
    
    * 통합층(pooling layer (POOL))
    * 완전 연결층(fully connected layer(FC))
    <img src="https://user-images.githubusercontent.com/97011426/202656417-8da4cfeb-5847-4fdb-bd5d-83bc55774594.png" width="60%" height="60%"/>
    
* 신경망 연결성 비교
  * 완전 연결층
    * 완전 연결(fully connection)구조로 높은 복잡도
    * 학습이 매우 느리고 overfitting 문제
  * 합성곱 연결층
    * 컨볼루션 연산을 이용한 부분연결(희소연결, 가중치 공유) 구조로 복잡도를 크게 낮춤
    
      -> 사진은 지역성(locality)특징 때문에 주변 픽셀에서만 정보를 얻어도 충분
    * 컨볼루션 연산은 영상과 같이 격자 구조를 갖는 데이터 특징 추출에 적합함 
    
      ->: 입출력 특징 형상(공간 정보) 유지
    <img src="https://user-images.githubusercontent.com/97011426/202658383-414777e8-ac84-4ac5-bf2b-efb9c2f1632a.png" width="80%" height="80%"/>
    
* 합성곱의 가중치 공유(weight sharing OR parameter sharing)
  * 모든 노드가 동일한 커널을 사용 : 모델의 복잡도가 크게 낮아짐 (아래 예시에서는 매개변수가 3개)
  <img src="https://user-images.githubusercontent.com/97011426/202658750-676de312-9154-4e6a-8c33-045c7c6001a3.png" width="30%" height="30%"/>

* 합성곱 연산 : 선형연산을 구간 쪼개서 한다는 개념
  * 사진 필터링(image filtering)
  <img src="https://user-images.githubusercontent.com/97011426/202659150-0a7f549d-6bfb-4e18-8a25-2aae6f8c42e3.png" width="60%" height="60%"/>
  
  * 교차 연관성(cross-correlation) 연산
  <img src="https://user-images.githubusercontent.com/97011426/202659427-3042c3f9-01da-4bce-9c33-e3eec81be4c6.png" width="90%" height="90%"/>
  
  * 합성곱(convolution) 연산 : 위의 교차 연관성과 수식적으로는 다르지만 연산적으로는 같다.
  <img src="https://user-images.githubusercontent.com/97011426/202659732-1690028f-02de-4311-96de-13cf9916d1fa.png" width="90%" height="90%"/>

* 인공 신경망의 합성곱 연산
  * 컨볼루션은 해당하는 요소끼리 곱하고 결과를 모두 더하는 선형 연산
  <img src="https://user-images.githubusercontent.com/97011426/202660218-68bf73a2-bd8f-4c8b-b0a8-4c36bcef6941.png" width="70%" height="70%"/>
  
  * u는 커널(필터), z는 입력, s는 출력(특징맵), h는 커널의 크기
  * 영상에서 특징을 추출하기 위한 용도로 사용됨(= 공간 필터)
  <img src="https://user-images.githubusercontent.com/97011426/202660711-46bbc943-eb36-4020-b4f8-94e3cd2ee78d.png" width="70%" height="70%"/>
  
  * 이때 주변에 빈공간이 생겨 공간이 축소되는데, 이를 유지하기 위해 수를 채워주는 padding을 진행한다.
    
    근데 아무 수나 넣어주게되면 data 변형이 생길 수 있기에 0으로 padding 해준다.
  
* 3차원 합성곱 연산
  * RGB 컬러 영상은 3*m*n의 3차원 텐서 (채널 3개)
  
    -> 이때는 무조건 입력과 커널층의 두께(채널)가 같아야함
    <img src="https://user-images.githubusercontent.com/97011426/202661429-e2c3ae9f-2601-46fa-8c98-8d03faf806ae.png" width="70%" height="70%"/>
    
  * 3차원 합성곱 연산 : 여러개의 특징맵 (filter 개수 == feature map 개수)
  <img src="https://user-images.githubusercontent.com/97011426/202661699-00a46575-a20c-49ef-b95a-2ef034cd09f7.png" width="50%" height="50%"/>







  
