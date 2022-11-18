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

* 합성곱 연산시 합성곱 신경망의 특성
 * 이동에 동변 : 신호가 이동하면 이동 정보가 그대로 특징맵에 반영
    
    -> 영상 인식에서 물체 이동이나 음성 인식에서 발음 지연에 효과적으로 대처(형상유지)
    
    <img src="https://user-images.githubusercontent.com/97011426/202672608-7543a181-ad71-42b5-800a-d9d2aa185bad.png" width="30%" height="30%"/>
    
  * 병렬분산 구조
    * 각 노드는 독립적으로 계산 가능하므로 병렬 구조(GPU활용 가능)
    * 노드는 깊은 층을 거치면서 전체에 영향을 미치므로 분산 구조 : 특징 점진적 통합 -> 추상화
   <img src="https://user-images.githubusercontent.com/97011426/202673074-cfb17ba1-d4f5-46cf-8efd-bd74e4d87df6.png" width="50%" height="50%"/>

* 채널이 k개인 3차원 격자 구조 (동영상은 사진의 연속)
<img src="https://user-images.githubusercontent.com/97011426/202673677-99aeb449-1fb3-468e-ae0d-8ef2090b527e.png" width="80%" height="80%"/>

* 복수의 특징맵 추출
 * 만약, 하나의 커널만 사용한다면 너무 빈약한 특징이 추출됨
 * 커널의 매개변수 값에 따라 커널이 추출하는 특징이 달라짐
 <img src="https://user-images.githubusercontent.com/97011426/202674045-4b93a0d7-7089-459d-88de-10eddcae0757.png" width="60%" height="60%"/>
 
 * 3개의 커널을 사용하여 3개에 대응되는 특징맵을 추출함 & 실제 커널의 매개변수는 +1(bias u0)
   * 실제로는 수십~수백 개의 커널 사용
   * 𝑢𝑖𝑘는 k번째 합성곱 필터의 i번째 매개변수
  <img src="https://user-images.githubusercontent.com/97011426/202674601-768b3482-3b41-4806-bc07-9d5ad0e0ac8f.png" width="60%" height="60%"/>
 
 * 합성곱 필터를 사람이 설계하지 않고, 학습으로 찾음 by 오차 역전파
 
* 하이퍼 파라미터 : 사람이 정하는 매개변수
 * 은닉층의 개수
 * 합성곱 필터 설정 : 높이(height), 폭(weidth), 깊이(depth), 보폭(stride), 덧대기(padding) 등등
   * pooling & stride : 중간에 저차원으로 내리기 가능
  
* 큰 보폭(stride)에 의한 축소(down-sampling)
   * 지금까지는 stride = 1 가정
   * stride=2라면?
     * 일반적으로 보폭이 k이면, k개 마다 하나씩 추출하여 필터 적용
     
       -> 2차원 영상의 경우 특징맵이  1/k**2로 작아짐
       
       <img src="https://user-images.githubusercontent.com/97011426/202676030-d46297b0-efe3-480d-ab91-dc8a4a4e9aec.png" width="60%" height="60%"/>

* 합성곱 층의 입력/출력 크기와 매개변수 개수의 상관관계
<img src="https://user-images.githubusercontent.com/97011426/202676305-33e8c69c-ecaf-4fb9-afde-0c2ffa8d78c2.png" width="60%" height="60%"/>


* 지금까지 Convolution층에 관한 내용이었고 Convolution층에서는 선형연산이 이루어지며 이제 뒤에서 비선형연산(특징공간변환)이 이루어진다.

* 통합(pooling) 연산
  * 통계적 대표값을 활용하여 특징맵의 정보요약(down-sampling) 혹은 부각 : 학습X, 기계적 계산
    * 특징맵의 깊이(channel)를 유지, 크기(Height, Width)만 줄임
    * 최대값, 평균값등 활용
  
  * 매개변수가 없음
 
  * 연산 효율화, 작은 변화에 둔감 -> 물체 인식이나 영상 검색등에 효과적임 (== minor정보들 손실)

 <img src="https://user-images.githubusercontent.com/97011426/202677391-ba1f29d9-9d5a-498b-b15d-efe383348e84.png" width="70%" height="70%"/>


* 합성곱 신경망 전체 구조
<img src="https://user-images.githubusercontent.com/97011426/202677565-1a6ace40-3dc5-4833-8513-3bb07b75abb9.png" width="70%" height="70%"/>


## 3. 대표적인 합성곱 신경망
* 합성곱 신경망 역사
  * ILSVRC 영상 인식 대회 우승 : 1000가지 사진 분류
  * ImageNet : 14백만 사진, 20000여개 종류
  * 대표적인 합성곱 신경망
     * LeNet(1998)
     * AlexNet(2012)
     * VGGNet(2014)
     * GoogLeNet(2014)
     * ResNet(2015)
     

* LeNet-5
  * 초창기 합성곱 신경망 사례
    *CNN의 첫 번째 성공 사례; 필기 숫자 인식기 만들어 수표 인식 자동화 시스템 구현
  * 특징 추출 : CONV-POOL-CONV-POOL-CONV의 다섯층 -> 28*28 명암 영상ㅇ르 120차원 특징 벡터로 변환
  * 분류 : 1개의 완전 연결층(softmax 로 확률화)
  <img src="https://user-images.githubusercontent.com/97011426/202678665-95e34780-62ec-4c19-83aa-ebfcc308ce94.png" width="60%" height="60%"/>
  
* AlexNet
  * 합성곱 층 5개와 완전연결 층 3개 => 총 8층
  * 합성곱 층은 200만개, 완전연결 층은 6500만개 가량의 매개변수 : 완전 연결층에 30배 많은 매개변수
  <img src="https://user-images.githubusercontent.com/97011426/202679017-3a7a840f-5300-404f-b10f-26a795a9133b.png" width="90%" height="90%"/>

  * AlexNet 학습 성공의 외적 요인
    * ImageNet 훈련 사용 (data 증대)
    * GPGPU 활용한 병렬 처리 : GPU분할 학습 수행
    <img src="https://user-images.githubusercontent.com/97011426/202679427-7a2d21f0-e7fc-4c37-b7ce-f7f474fa26bb.png" width="50%" height="50%"/>
    
  * AlexNet 학습 성공의 내적 요인
    * ReLU 활성화 함수 사용
    * 지역 반응 정규화 기법 적용 -> 지금은 사용하지 않는 기술
    * 여러 규제 기법 적용
      * 데이터 확대augmentation (cropping & mirroring으로 2048배로 확대) => 단! 원본특징은 유지되어야 함
      * 드롭아웃dropout (완전연결층에서 사용) =>  단! 학습할때만 dropout시키고 test할때는 완전연결로
    * 추론inference 단계에서 앙상블 적용 : 동일 모델에 대하여 data에 대한 결과를 평균내기
    
    <img src="https://user-images.githubusercontent.com/97011426/202680381-d26d1234-8f61-4523-b460-fb327a26a8f6.png" width="100%" height="100%"/>

* VGGNet
  * 3*3의 작은 필터 사용 -> 신경망을 더욱 깊게 만듬 (신경망의 깊이가 어떤 영향을 주는지 확인)
  <img src="https://user-images.githubusercontent.com/97011426/202680742-2db2195c-d329-4e5f-8f9f-5a91e6428603.png" width="90%" height="90%"/>

  * VGGNet 작은 필터의 이점
    * 큰 크기의 필터는 여러 개의 작은 크기 필터로 분해될 수 있음 -> 매개변수의 수는 줄어들면서 신경망은 깊어지는 효과
    <img src="https://user-images.githubusercontent.com/97011426/202685637-55480ad1-0696-4527-904c-2f1ccf623d94.png" width="70%" height="70%"/>
    
  * 1*1 합성곱 필터 : depth 개수만 변경 => 차원변경
    * VGGNet에서 가능성을 실험했지만 최종 적용은 하지 않음, GoogLeNet에서 적용
    * 차원 통합 (depth가 1로 변함)
    <img src="https://user-images.githubusercontent.com/97011426/202686614-78eaaa3e-ab61-4027-a4cf-06b9691e77c8.png" width="40%" height="40%"/>
   
    * 차원 축소 효과
    
      <img src="https://user-images.githubusercontent.com/97011426/202690375-a18f3871-54fb-48f2-8229-5866423a2281.png" width="70%" height="70%"/>

* [참고] NIN 구조
  <img src="https://user-images.githubusercontent.com/97011426/202690902-ea9cd28e-328d-462a-a05f-5c8eaac99cad.png" width="80%" height="80%"/>
  
* GooLeNet
  * 인셉션 모듈 inception module
    * 다양한 특징을 추출하기 위해 NIN의 구조에 기반해 복수의 수용장 크기를 병렬적으로 적용한 층
    * 미소 신경망 대신 4가지의 합성곱 연산 사용
      <img src="https://user-images.githubusercontent.com/97011426/202691272-1611379e-69d5-4618-a34b-16ae7382df01.png" width="90%" height="90%"/>
      
    * 같은 층에서 4가지의 특징맵을 출력 BUT 나중에 합칠때 Depth가 다 달라서 연산량이 너무 많아짐 -> 1*1 합성곱 사용하여 차원 축소
    
  * GoogLeNet 전체 구조
    * 인셉션 모듈(I)을 9개 결합
    * 매개변수가 있는 층 22개, 없는 층 5개로 총 27개 층을 가짐
    * 완전연결 층은 1개에 불과 => 1백만 개의 매개변수를 가지며, VGGNet의 완전연결 층에 비하면 1%에 불과
      <img src="https://user-images.githubusercontent.com/97011426/202691721-128fb66b-7dfe-48d4-b714-7a86f236ec76.png" width="90%" height="90%"/>
      
    * 아무리 ReLU라 하더라도 이정도 깊이면 gradient 손실이 발생함 -> 이 문제를 보조 분류기가 해결
    *  보조 분류기 (auxiliary classifier)
       * 원분류기의오류역전파결과와보조분류기의오류역전파결과를결합하여경사소멸문제완화
       * 훈련할 때 도우미 역할을 하고, 추론할 때 제거됨

* ResNet
  * 잔류(잔차) 학습residual learning이라는 개념을이용하여 성능저하를피하면서은닉층 수를 대폭늘림
  * 기존 합성곱 신경망 학습의 문제 -> 56-layer가 train과 test가 다 안좋음 : overfitting이 아닌 optimizer가 안됨
    <img src="https://user-images.githubusercontent.com/97011426/202692282-10ad3c6b-46a7-478f-a0fb-5ed1ee884e7a.png" width="60%" height="60%"/>

  * 잔류 학습 -> 깊은 은닉층의 최적화 효율성 높임 => 그 전까지 학습이 잘 되었으면 학습 pass 개념
    <img src="https://user-images.githubusercontent.com/97011426/202692861-30114465-2f0e-42e2-87ce-5921bd84136a.png" width="60%" height="60%"/>
    
  * 잔류 학습은 지름길skip connection 연결된 𝐱를 더한 𝐅 𝐱 + 𝐱에 𝛕를 적용. 𝐅 𝐱 는 잔류
  <img src="https://user-images.githubusercontent.com/97011426/202693174-27f81021-f04d-462e-9b02-c0e206d3e12f.png" width="60%" height="60%"/>

  * 지름길 연결의 장점
    * 깊은 신경망도 최적화가 가능해짐
    * 단순한 학습의 관점의 변화를 통한 신경망 구조 변화
       * 단순 구조의 변경으로 매개변수 수에 영향이 없음
       * 덧셈 연산만 증가하므로 전체 연산량 증가도 미비함
    * 깊어진 신경망으로 인해 성능 개선 가능
    * 오류 역전파의 경사 소멸 문제 해결
    
    <img src="https://user-images.githubusercontent.com/97011426/202693530-e62073e7-ef6a-4986-a293-977cd618809d.png" width="80%" height="80%"/>


## 4. Deep Learning Feature
* 컴퓨터 비전 문제에 적용

  <img src="https://user-images.githubusercontent.com/97011426/202694210-a523eb4c-0aa5-47ad-97ea-37d827071acf.png" width="80%" height="80%"/>

* 생성 모델과 분별 모델 비교
<img src="https://user-images.githubusercontent.com/97011426/202694353-19acf199-21c0-45d2-bcec-12d25ba72ba2.png" width="60%" height="60%"/>

* 생성 모델
  * 현실에 내재한 데이터 발생 분포 Pdata(X) & 확률분포의 명시적 추정 -> 알아 낼 수 없음
  * 심층학습을 사용하여 확률분포를 암시적으로 표현 : GAN, VAE, RBM
  
* 생성 모델 종류
<img src="https://user-images.githubusercontent.com/97011426/202694705-dc02a180-2c04-4a8e-bb40-97dfb1571c75.png" width="90%" height="90%"/>

* 생성적 적대 신경망(Generative Adverserial Network; GAN)의 핵심
  * 생성기(generator) G와 분별기(discriminator) D의 대립(minimax game algorithm)
  * G는 가짜 샘플 생성(위조지폐범) & D는 가짜와 진짜를 구별(경찰)
  * GAN의 목표는 위조지폐범의 승리
  <img src="https://user-images.githubusercontent.com/97011426/202695181-7597e8c9-44d7-488e-bbe9-338a21925638.png" width="90%" height="90%"/>


  
