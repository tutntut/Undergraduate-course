# Artificial Neural Networks (인공 신경망)

### 1. Artificial Neural Networks (인공 신경망)
* 인공 신경망
  * 기계학습 역사에서 가장 오래된 학습 모델
  * 현재 가장 대표적인 학습 모델이며, 깊은 신경망 형태를 가짐
  * 심층학습(deep learning)의 기초
* 인공 신경망 종류
  * 전방 신경망(forward)과 순환 신경망(recurrent)
  * 얕은 신경망(shallow)과 깊은 신경망(deep)
  <img src="https://user-images.githubusercontent.com/97011426/202591625-016d2af9-9291-4251-b4f3-a1c6c7f1ae17.png" width="50%" height="50%"/>
  
  * 결정론적 신경망과 확률론적 신경망
  <img src="https://user-images.githubusercontent.com/97011426/202592338-b80352c3-9fde-43ec-b7bd-20068712a449.png" width="50%" height="50%"/>
  
### 2. Perceptron (퍼셉트론)
* 퍼셉트론 특징
  * 인간 뉴런을 모방한 최초의 인공 신경망
  * 노드(node), 가중치(weight), 층(layer)과 같은 개념을 도입하고 학습 알고리즘을 제시
  * 원시 신경망이지만, 심층학습을 포함한 현대 인공신경망의 중요한 구성 요소
  <img src="https://user-images.githubusercontent.com/97011426/202592647-5a932da1-d646-47ae-bc92-a49cb8d91c86.png" width="80%" height="80%"/>
  
* 퍼셉트론 구조
  * 입력층과 출력층을 가짐
    + 입력층은 연산을 하지 않으므로 단일층 구조로 간주
    + 입력층의 i번째 노드는 특징 벡터 𝐱 = (𝑥1, 𝑥2, ⋯ , 𝑥𝑑)T의 요소 𝑥𝑖를 담당
    + 항상 1이 입력되는 편향bias 노드
    + i번째 입력 노드와 출력 노드를 연결하는 변edge은 가중치 𝑤𝑖를 가짐
    + 출력층은 한 개의 노드
 <img src="https://user-images.githubusercontent.com/97011426/202593320-e21bb812-b5bc-4016-84b1-91d308f974e2.png" width="70%" height="70%"/>
 
 * 퍼셉트론 동작
    - 선형 연산 (입력값과 가중치를 곱하고 더하여 s 구함) + 비선형 연산 (활성함수τ 적용)
  
       → 계단 함수step function를 활성함수 𝜏로 사용한다면, 최종 출력 y는 +1 또는 -1 얻음
       <img src="https://user-images.githubusercontent.com/97011426/202593570-b50115a9-4caf-4e26-bae9-dedb3aaf02cf.png" width="50%" height="50%"/>
       ->이때 활성함수τ는 비선형 연산 & s는 선형연산
      
* 행렬 표기
  * 선형 연산 표기
    <img src="https://user-images.githubusercontent.com/97011426/202593905-9a316499-cb81-4b02-a16b-45cdb32cc814.png" width="50%" height="50%"/>
  + 편향을 추가하면
    <img src="https://user-images.githubusercontent.com/97011426/202593962-f5d26e87-0799-4e31-a1cc-e9a5fca59e81.png" width="50%" height="50%"/>
    
          -> 이때 편향은 함수의 원점을 옮긴다는 개념으로 이해 가능
   * 비선형 연산을 포함한 퍼셉트론 표기
    <img src="https://user-images.githubusercontent.com/97011426/202594103-0b034c0b-92d7-4230-8fc6-a5196bdb1dd6.png" width="20%" height="20%"/>

* 퍼셉트론 동작의 기하하적 의미
  * 2차원 특징 공간 : 결정 직선 ( 주어진 특징 공간을 +1과 -1의 두 부분 공간으로 분할하는 분류기 역할
  <img src="https://user-images.githubusercontent.com/97011426/202594887-34acfa01-2983-44c7-b70b-4466191c2abe.png" width="60%" height="60%"/>
  
  * d차원 특징 공간
  
    <img src="https://user-images.githubusercontent.com/97011426/202594993-a08a1cf3-e1a0-4b8e-b536-e2aff042f08e.png" width="60%" height="60%"/>

* 일반적인 분류기의 학습 수행 과정
  - 1 단계: 분류기의 가설(모델) 설정
  - 2 단계: 해당 분류기의 목적함수 𝐽(Θ) 정의
  - 3 단계: 𝐽(Θ)를 최소화하는 Θ를 찾기 위한 최적화 방법 수행
  
* 퍼셉트론 분류기의 학습 수행 과정 2단계 : 목적함수 정의
  - 퍼셉트론의 매개변수를 𝐰 = (𝑤0, 𝑤1, ⋯ , 𝑤𝑑)T 라 하면, 매개변수 집합: Θ = 𝐰 , 목적함수: 𝐽(Θ) 또는 𝐽(𝐰) 표기
  - 퍼셉트론의 목적함수 조건 ( 단, 모든 샘플은 선형 분리가 가능함을 가정함 )
    1. 𝐽(𝐰)>= 0 이다 (1)
    2. w가 최적이면, 즉 모든 샘플을 맞히면 𝐽(𝐰) = 0이다. (2)
    3. 틀리는 샘플이 많은 w일수록 𝐽(𝐰)는 큰 값을 가진다. (3)
    
* 퍼셉트론 목적 함수

<img src="https://user-images.githubusercontent.com/97011426/202595666-c644f0e4-2e14-4535-827c-b5dd890f210f.png" width="80%" height="80%"/>

* 퍼셉트론 분류기의 학습 수행 과정 3단계 : 최적화
  *  경사 하강법(gradient descent): 𝐽(Θ) 의 기울기(미분) 정보를 활용한 반복 탐색(𝚯𝐭+𝟏 ← 𝚯𝐭 − 𝜌𝐠)을 수행하여 극소값을 찾음
  <img src="https://user-images.githubusercontent.com/97011426/202595954-8abc5f7c-a6bb-46be-b244-dcb0c01e57d9.png" width="60%" height="60%"/>
  
       -> 왼쪽 그래프에서 왼쪽 부근의 최저점을 local minimum, 오른쪽의 최저점을 global minimum이라 한다. (학습률의 중요성에서 다시)
     
* 경사도 계산: 가중치 갱신 규칙 𝚯𝐭+𝟏 ← 𝚯𝐭 − ρ𝐠를 적용하려면 경사도 𝐠 필요, 학습률(learning rate) ρ

  <img src="https://user-images.githubusercontent.com/97011426/202596504-83e27eeb-e66a-4ee5-a7df-1ec2c2a07538.png" width="80%" height="80%"/>

     -> 학습률이 너무 크면 이동을 너무 크게 해 최저점을 발견하지 못하고, 학습률이 너무 작으면 시간이 오래 걸린다. 
     

* 행렬 표기하여 간결하게 표기 :

<img src="https://user-images.githubusercontent.com/97011426/202596842-26cde160-a42c-47d8-bf3c-c9bffb1ba375.png" width="80%" height="80%"/>

-> 배치 버전 : 전체 학습 데이터를 하나의 배치로 묶어 학습시키는 방법

-> 스토캐스틱 버전 : 단 하나의 데이터를 이용하여 학습 1회 진행하는 방법 (배치크기가 1)


### 3. Multi-Layer Perceptron (다층 퍼셉트론)
* 퍼셉트론은 선형 분류기linear classifier, 활성함수가 미분 분가능 한계
  - 선형 분리 불가능한 상황에서는 일정한 양의 오류
    + XOR 문제는 75% 정확도 한계
    <img src="https://user-images.githubusercontent.com/97011426/202598041-bbb0c3f5-958f-468f-8283-3e7ccdeec075.png" width="60%" height="60%"/>
    
* 다층 퍼셉트론 핵심
  - 은닉층hidden layer: 특징 공간을 과업 수행에 유리한 새로운 특징 공간으로 변환
  - 시그모이드sigmoid 활성함수activation function: 경성hard 출력의 계단 함수 → 연속적인 연성냀 출력의 시그모이드 함수
  - 오류 역전파 학습 알고리즘: 복수층이 순차적으로 이어진 구조로 역방향 경사도를 한층 씩 계산하고 가중치를 갱신

* XOR 문제 해결
  * 논리 회로 활용 관점
    <img src="https://user-images.githubusercontent.com/97011426/202598289-facabd62-5c8b-4d42-ab57-ada87a6b2577.png" width="60%" height="60%"/>












  



