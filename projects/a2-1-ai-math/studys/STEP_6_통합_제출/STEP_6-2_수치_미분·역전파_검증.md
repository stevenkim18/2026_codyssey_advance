# STEP 6-2. 수치 미분·역전파 검증

> 코디세이 A2-1 AI 수학 · STEP 6 통합 실습과 제출 · Python/NumPy 입문자용

## 학습 목표

- 중심차분 오차 <1e-4를 검증한다.
- Gradient 화살표를 등고선과 함께 나타낸다.
- 2→2→1 역전파와 수치미분을 교차 검증한다.

## 준비 환경

```bash
python -m pip install numpy matplotlib
```

## 1. 평가 요구사항과 연결하기

이 단원은 STEP 2와 STEP 3을 한 번에 연결한다. **손계산→수치 검증** 원리를 그대로 활용한다.

| 평가 | 확인할 것 |
|---|---|
| 수치미분 | 해석값과 절대오차 `1e-4` 이내 |
| Gradient | 등고선에 수직 방향으로 화살표 표시 |
| 역전파 | 2→2→1 Sigmoid+BCE의 중간값·손실·기울기·업데이트 |

## 2. 수치미분과 Gradient 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

f=lambda x:x**3
h=1e-5;x0=3.
num=(f(x0+h)-f(x0-h))/(2*h)
exact=3*x0*x0
print(f'수치 미분 = {num:.9f} / 해석값 = {exact:.9f} / 오차 = {abs(num-exact):.2e}')
assert abs(num-exact)<1e-4

# f(x,y) = (x-1)^2 + 2(y+2)^2
xx,yy=np.meshgrid(np.linspace(-3,5,200),np.linspace(-6,2,200))
zz=(xx-1)**2+2*(yy+2)**2
px,py=np.meshgrid(np.linspace(-2.5,4.5,8),np.linspace(-5.5,1.5,8))
gx=2*(px-1); gy=4*(py+2)
length=np.hypot(gx,gy)
ux=np.divide(gx,length,out=np.zeros_like(gx),where=length>0)
uy=np.divide(gy,length,out=np.zeros_like(gy),where=length>0)
plt.figure(figsize=(7,6));plt.contour(xx,yy,zz,levels=18)
plt.quiver(px,py,ux,uy,color='red')
plt.scatter([1],[-2],c='black');plt.axis('equal')
plt.title('Gradient vectors normal to contours');plt.show()
```

## 3. 2→2→1 역전파 및 기울기 검증

이 예제의 `x`, 가중치, 편향은 STEP 3-4 및 3-6과 동일하다. **위 코드와 독립적으로 실행**할 수 있다.

```python
import numpy as np

x=np.array([1.,.5]);y=1.
W1=np.array([[.2,-.3],[.4,.1]]);b1=np.array([.1,-.2])
W2=np.array([[.7],[-.5]]);b2=np.array([.05])

def sigmoid(z):return 1/(1+np.exp(-z))
def forward():
    z1=x@W1+b1; a1=sigmoid(z1)
    z2=a1@W2+b2; pred=float(sigmoid(z2)[0])
    p=np.clip(pred,1e-12,1-1e-12)
    loss=-(y*np.log(p)+(1-y)*np.log(1-p))
    return loss,z1,a1,z2,pred

loss,z1,a1,z2,pred=forward()
d2=pred-y
gW2=a1[:,None]*d2; gb2=np.array([d2])
d1=(W2[:,0]*d2)*a1*(1-a1)
gW1=np.outer(x,d1);gb1=d1
print('순전파: z1=',z1,'a1=',a1,'z2=',z2,'pred=',pred,'loss=',loss)
print('역전파: dW1=',gW1,'db1=',gb1,'dW2=',gW2,'db2=',gb2)

h=1e-5
for param,analytic,name in [(W1,gW1,'W1'),(b1,gb1,'b1'),(W2,gW2,'W2'),(b2,gb2,'b2')]:
    numeric=np.zeros_like(param)
    for index in np.ndindex(param.shape):
        old=param[index]
        param[index]=old+h;lp=forward()[0]
        param[index]=old-h;lm=forward()[0]
        param[index]=old
        numeric[index]=(lp-lm)/(2*h)
    error=np.max(np.abs(analytic-numeric))
    print(name,'gradient check max abs error:',error)
    assert error<1e-4

# 가중치 업데이트가 올바르게 진행되는지 확인
alpha=.1
W1-=alpha*gW1;b1-=alpha*gb1
W2-=alpha*gW2;b2-=alpha*gb2
new_loss=forward()[0]
print('한 번 업데이트 후 손실:',new_loss)
assert new_loss<loss
```

## 4. 평가 설명 순서

1. 수치미분 공식을 설명하고 오차를 출력한다.
2. 등고선과 Gradient의 방향이 수직인 이유(`∇f·접선=0`)를 설명한다.
3. 2→2→1의 `shape`와 순전파 값, BCE를 보여 준다.
4. `δ2=ŷ-y`와 은닉층 `δ1` 계산을 설명한다.
5. 각 파라미터 기울기와 수치미분을 비교해 검증한다.

---

## 직접 풀어보기

① `h=1e-4`로도 수치미분 평가를 통과하는지 보라. ② `W2[0,0]`를 대상으로만 중앙차분을 수행하라. ③ `alpha`를 .01로 낮추면 1회 손실 감소량이 어떻게 달라지는가?

<details>
<summary>정답·힌트 확인</summary>

① `x³`의 중심차분 오차는 `h²` 수준이라 통과한다. ② `W2`를 `±h` 변경한 손실 차이를 `2h`로 나눌 것. ③ 보통 손실 감소량이 작아진다.

</details>

## 미션 평가 체크

- [ ] 미적분 2항목과 역전파 순전파·기울기 검증의 출력 증거를 모두 갖추었는가?
- [ ] 핵심 수식을 말로 설명할 수 있다.
- [ ] Python 코드를 직접 실행하고 출력 또는 그래프를 확인했다.

---

[← 이전: 6-1 선형대수 실습 정리](STEP_6-1_선형대수_실습_정리.md) · [전체 목차](../README.md) · [다음: 6-3 경사하강법·Momentum 실험 →](STEP_6-3_경사하강법·Momentum_실험.md)
