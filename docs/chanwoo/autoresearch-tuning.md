# autoresearch식 궤적 예측기 학습 튜닝

> 생성 2026-08-24 · 방법 참고: [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (2026-03)
> 실행: `uv run --group serve python train/autoresearch_traj.py` · 로그: [results/autoresearch-log.tsv](results/autoresearch-log.tsv)

## 무엇을 가져왔나 (정직 경계)

autoresearch는 **단일 GPU LLM(nanochat) 학습을 에이전트가 자동 실험**하는 도구다. GPT/Muon/BPE
코드는 우리 과제(좌표 회귀)와 무관해 **재사용하지 않았다.** 가져온 건 두 가지:
1. **실험 방법론**: 고정 설정 + 단일 val 지표 + 실험 로그(tsv) + 개선분만 채택 + 단순성 우선.
2. **학습 레시피 아이디어**: AdamW·weight decay, LR **warmup(10%) + warmdown(후반 50%)** 스케줄.

백본은 P0-1 승자 **Transformer 고정**, head·정규화·split·eval/CI 동일. 바꾼 건 **학습 레시피뿐**.

## 실험 (val minADE_moved, 낮을수록↑)

| 레시피 | 변경 | val minADE_moved | 판정 |
|---|---|---|---|
| baseline | Adam, flat LR (현행) | 0.7542 | 기준 |
| adamw_wd | AdamW + wd 0.01 | 0.7600 | discard(악화) |
| lr_sched | warmup+warmdown | 0.7441 | keep |
| adamw_sched | AdamW+wd + 스케줄 | 0.7476 | keep |
| **adamw_sched_big** | + 용량↑(layers 2→3·heads 4→8) | **0.7359** | **채택 (+2.4%)** |

- **LR 스케줄이 핵심 레버**(단독 -1.3%). AdamW 단독은 오히려 악화 → 스케줄과 결합해야 효과.
- 용량↑가 스케줄 위에 조금 더 보탬. 최종 채택 = `adamw_sched_big`.

## test(held-out) 확인 — 개선 유지

선택은 val, **test는 확인 1회**(P0-1 프로토콜). 라이브 1.6s 지평선:

| 예측기 | ADE@1.6s | FDE@1.6s | minADE@3 FDE |
|---|---|---|---|
| Transformer(최빈) | 0.207 | 0.375 | 0.202 |
| **Transformer-tuned(최빈)** | **0.192** | **0.349** | **0.188** |

→ test에서도 FDE@1.6s 0.375→0.349(**~7%↓**), val→test 일관. 튜닝이 held-out에서 유지됨.

## 한계·후속

- 개선폭은 **모듈러(+2.4% val, ~7% test FDE)** — 압도적은 아니며 CI가 일부 겹친다.
- 이번엔 5개 레시피 1회 루프(수동). autoresearch의 진짜 힘은 **밤새 100+ 실험 자동 루프**이므로,
  더 넓은 탐색(스케줄 모양·wd·depth·heads·batch)으로 추가 이득 여지.
- 공정성: 튜닝 레시피를 LSTM/CVAE에도 적용하면 백본 A/B를 같은 레시피 위에서 다시 비교 가능(후속).
- `model_transformer_tuned.pt`는 HF-managed(gitignore) — 재현은 `autoresearch_traj.py`(결정적 SEED=0).
