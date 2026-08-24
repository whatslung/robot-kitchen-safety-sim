# 궤적 Transformer 자동실험 작업 규칙

이 문서는 `karpathy/autoresearch`의 고정 시간 예산, 단일 변경 지점, 성공·실패 전부 기록,
개선안만 보존하는 운영 원칙을 이 저장소의 궤적 예측 문제에 맞게 적용한 실행 계약이다.
원본 LLM 학습 코드를 복사하거나 실행하지 않는다.

## 변경 경계

- 수정 허용: `trajectory/autoresearch_candidate.py` 하나
- 수정 금지: 데이터, manifest, loss, evaluator, guard, worker, runner, contract lock,
  test gate
- 자동실험은 v2 `train`으로 학습하고 `val`로만 순위를 정한다.
- 잠긴 `test`는 탐색, 후보 선택, 실패 분석에 사용하지 않는다.
- 한 시도의 학습 시간은 warmup 10 step을 제외하고 300초다.
- child 전체 제한 시간은 600초다.
- 빠른 탐색은 후보 20개 또는 2시간 중 먼저 도달한 시점에 끝낸다.

## 한 시도의 순서

1. `apply_patch`로 후보 파일만 변경한다.
2. 후보 출력 형상 테스트를 실행한다.
3. parent runner로 후보를 학습·validation 평가한다.
4. `training/autoresearch/results.jsonl`의 마지막 record를 읽는다.

```powershell
uv run --group serve --with pytest python -m pytest tests/test_autoresearch_candidate.py -q
uv run --group serve python train/run_autoresearch_experiment.py --trial-id trial-001 --model candidate --seed 0 --budget-seconds 300 --timeout-seconds 600
```

## keep와 revert

- keep: `status=ok`, `guards.passed=true`, Transformer 기준 대비 `f2_gain>0`, 현재 최고
  F2 초과를 모두 만족한다.
- keep이면 후보 파일만 `experiment: keep <trial-id> transformer candidate` 메시지로
  커밋한다.
- revert: 그 밖의 모든 결과다. 후보 파일만 마지막 keep 커밋 상태로 복원한다.
- NaN, 무한값, GPU 메모리 부족, timeout, child 비정상 종료도 실패 record로 남긴다.
- smoke와 3-seed 재검증 record는 빠른 탐색 순위에 섞지 않는다.

## 금지 사항

- test 실행
- `training/autoresearch/results.jsonl` 수정 또는 삭제
- `git reset --hard`, `git clean`
- 다른 worktree 조작
- 후보 파일 이외의 추적 파일 변경
- 자동 push, PR 생성, merge

결과는 시뮬레이터 v2 validation 궤적에 한정된다. 이 실험의 향상은 실제 급식실 안전 성능을
뜻하지 않는다.
