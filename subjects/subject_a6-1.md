# 미션: AI의 속 엔진(두뇌)을 내 손으로 직접 만들어보기

> 개인 필수 미션 · AI/SW 심화 · 딥러닝 · 80시간

## 1. 미션 소개

NumPy로 Tensor와 AutoGrad 엔진을 구현하고 Gradient Checking으로 정확성을 검증한다. Zero, Random, He 초기화의 학습 차이를 실험하여 구현→검증→실험의 딥러닝 프레임워크 개발 흐름을 경험한다.

## 2. 최종 결과물

- Tensor, autograd, layer, optimizer 모듈로 구성된 미니 프레임워크
- 수치 미분과 역전파 오차를 기록한 검증 리포트
- Zero/Random/He 초기화별 Loss 그래프와 실험 리포트
- 구조, 설치, 실행, 모듈 사용 예제를 담은 `README.md`
- XOR 학습 및 MNIST 분류 검증 스크립트

## 3. 기능 요구 사항

### Tensor와 AutoGrad

- `Tensor`: `data`, `grad`, `_backward` 속성
- 계산 그래프를 따라 연쇄 법칙으로 기울기를 전파하는 `backward()`
- `+`, `-`, `*`, `/`, MatMul의 순전파·역전파

### Gradient Checking

- 중심차분: `[f(x+ε)-f(x-ε)]/(2ε)`, `ε=1e-5`
- Analytic/Numerical Gradient 상대 오차 `1e-7` 이하
- 모든 레이어에 대한 테스트 포함

### 레이어·초기화·최적화

- Linear, ReLU, Sigmoid, Softmax
- Zero, 표준정규 Random, He `N(0,√(2/n_in))`, Xavier `N(0,√(2/(n_in+n_out)))`
- Zero/Random/He 3종 비교 필수, Xavier 구현 필수
- SGD와 Adam 직접 구현
- XOR과 MNIST로 정상 동작 검증

## 4. 성공 기준

| 항목 | 기준 |
|---|---|
| Gradient Check | 모든 레이어 상대 오차 `< 1e-7` |
| XOR | He Init, 100 Epoch 내 Loss `< 0.1` 또는 정확도 95% 이상 |
| MNIST | 선택 기준: 1 Epoch 정확도 80% 이상 |
| Zero Init | 50 Epoch 후 Loss 감소 없음 재현 |

초기화 비교 그래프는 `figures/`에 저장한다.

## 5. 보너스 과제

- Inverted Dropout과 L2 Weight Decay
- Tanh, LeakyReLU, ELU와 역전파

## 6. 개발 환경과 제약

- Python 3.10 이상 권장
- 실제 구현은 NumPy만 사용
- PyTorch/TensorFlow는 로직 참고만 허용
- 자동 데이터 로더 금지
- MNIST 등은 UBYTE/CSV를 직접 파싱해 NumPy로 로드
- 다운로드에는 `requests`, `urllib`, `gzip` 사용 가능

