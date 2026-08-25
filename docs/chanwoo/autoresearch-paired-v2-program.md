# Paired fixed-step trajectory autoresearch v2

## 목적과 경계

- 대상 데이터는 `trajectories_v2` 150개 scene의 기존 고정 분할이다.
- 학습은 train 90개, 모델 선택은 validation 30개만 사용한다.
- 잠긴 test 30개는 이 실험에서 읽지 않는다.
- 결과는 시뮬레이터 궤적 validation 결과이며 실제 급식실 안전 성능이 아니다.

## 이전 실험에서 바꾼 점

- 300초 동안 가능한 만큼 학습하던 방식을 모든 실행 70,000 step으로 바꾼다.
- 후보를 seed 0 기준값 하나와 비교하지 않고, seed 0·1·2 각각 같은 seed의 Transformer 기준값과 비교한다.
- 기존 `training/autoresearch` 결과는 보존하고 새 결과는 `training/autoresearch-paired-v2`에만 쓴다.

## 사전 등록한 실행

각 실행은 batch 512, weight decay 0이며 아래 여섯 조합만 허용한다.

| 이름 | 모델 | 학습률 |
|---|---|---:|
| `lstm-lr1e3` | LSTM | 0.001 |
| `lstm-lr6e4` | LSTM | 0.0006 |
| `transformer-lr1e3` | Transformer | 0.001 |
| `transformer-lr6e4` | Transformer | 0.0006 |
| `cvae-lr1e3` | CVAE | 0.001 |
| `cvae-lr6e4` | CVAE | 0.0006 |

모든 조합을 seed 0·1·2로 실행해 총 18개 결과를 만든다. 학습률 외 구조, 손실 함수, 데이터 증강, 평가 방식은 기존 고정 계약과 같다.

## 같은 seed 보호 조건

각 후보 실행은 같은 seed의 `transformer-lr1e3`과 비교한다.

- 재현율 하락: 0.01 이하
- FDE@1.6s 증가: 2% 이하
- CPU p95 지연 증가: 20% 이하
- 학습 파라미터 증가: 20% 이하

세 seed가 모두 보호 조건을 통과한 후보만 선택 대상이다. 그중 validation F2 중앙값이 가장 높은 후보가 Transformer 기준 중앙값보다 0.01 이상 높아야 최종 후보가 된다. 조건을 만족하는 후보가 없으면 `transformer-lr1e3`을 유지한다.

## 재현성과 중단 기준

- PyTorch 결정론 연산을 강제하고 CUDA 작업공간 설정을 고정한다.
- 재현성은 PyTorch 저장 파일 바이트가 아니라 텐서 이름·자료형·형태·값의 SHA-256으로 확인한다. 저장 파일 안에는 임시 파일명이 들어가므로 파일 전체 해시는 같은 가중치에도 달라질 수 있다.
- 각 실행 제한 시간은 900초다. 손실이 유한하지 않거나 child process가 실패하면 본실험을 중단한다.
- 완료된 `(variant, seed)`는 결과 로그를 기준으로 건너뛰어 중단 후 이어서 실행할 수 있다.

## Test 사용 금지

validation 승자가 정해져도 test는 자동 실행하지 않는다. 최소 개선 조건을 만족한 뒤 별도 확인을 받아 한 번만 평가한다.
