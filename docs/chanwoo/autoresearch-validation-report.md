# Transformer 자동실험 validation 결과

> 이 결과는 v2 validation에서 고른 시뮬레이터 궤적 예측 성능이다.
> locked test는 아직 실행하지 않았고 실제 급식실 안전 성능을 뜻하지 않는다.

> 주의: 이 수치는 엄격한 파일 격리 수정 전에 생성됐다. test JSON을 파싱하거나 평가하지는 않았지만 manifest 무결성 검사에서 test 파일의 SHA-256을 읽었다.

## 실행 요약

- 기록 23개: 성공 22, 실패 1, keep 1
- 빠른 탐색: warmup 10 step 제외 300초, seed 0, 최대 20개 또는 2시간
- 실제 종료: 12개 후보, keep 이후 6회 정체와 gated residual 비교 후 종료
- 선택: 후보와 기존 Transformer를 seed 0·1·2로 재검증

## 300초 기준 모델

| 모델 | F2 | 재현율 | 정밀도 | FDE@1.6s(m) | CPU p95(ms) | 파라미터 |
|---|---:|---:|---:|---:|---:|---:|
| lstm | 0.8486 | 0.9308 | 0.6270 | 0.3017 | 0.2124 | 28783 |
| transformer | 0.7976 | 0.9390 | 0.4978 | 0.2791 | 0.2647 | 115631 |
| cvae | 0.8657 | 0.9215 | 0.6969 | 0.2784 | 0.2971 | 117226 |

## 빠른 탐색

| 후보 | F2 | 재현율 | FDE@1.6s(m) | 보호 조건 | 판정 |
|---|---:|---:|---:|---|---|
| trial-001-pre-norm | 0.8590 | 0.9380 | 0.2938 | fde16 | revert |
| trial-002-mean-pool | 0.8622 | 0.9349 | 0.2941 | fde16 | revert |
| trial-003-hidden-48 | 0.7932 | 0.9432 | 0.2920 | fde16 | revert |
| trial-004-one-layer | 0.8670 | 0.9525 | 0.3007 | fde16 | revert |
| trial-005-lr-0006 | 0.8619 | 0.9390 | 0.2817 | 통과 | keep |
| trial-006-lr-0003 | 0.8568 | 0.9360 | 0.2944 | fde16 | revert |
| trial-007-pre-norm-lr-0006 | 0.8502 | 0.9205 | 0.2868 | recall, fde16 | revert |
| trial-008-mean-pool-lr-0006 | 0.8603 | 0.9401 | 0.2949 | fde16 | revert |
| trial-009-one-layer-lr-0006 | 0.8542 | 0.9390 | 0.3079 | fde16 | revert |
| trial-010-ff-ratio-2 | 0.7885 | 0.9349 | 0.2937 | fde16 | revert |
| trial-011-dropout-005 | 0.8593 | 0.9411 | 0.2947 | fde16 | revert |
| trial-012-gated-residual | 0.7963 | 0.9411 | 0.2908 | fde16, cpu_p95_ms | revert |

## 3-seed 재검증

| 대상 | 중앙 F2 | 중앙 재현율 | 중앙 FDE@1.6s(m) | 중앙 CPU p95(ms) | 모든 seed 보호 통과 |
|---|---:|---:|---:|---:|---|
| baseline-transformer | 0.7922 | 0.9339 | 0.2863 | 0.2652 | 기준 |
| trial-005-lr-0006 | 0.8524 | 0.9277 | 0.2894 | 0.2656 | False |

## 결론

- 선택: `baseline-transformer`
- 최소 개선(F2 +0.01): `False`
- 3-seed 중앙 F2 개선: 0.0000
- 빠른 탐색 keep 후보는 단일 seed에서 개선됐지만 3-seed 모두 보호 조건을 통과하지 못했다.
- 따라서 기존 Transformer로 되돌렸고 locked test를 실행하지 않는다.
