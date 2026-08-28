# 발표용 파이프라인 그림 (교수님 발표)

이동경로 예측 안전 파이프라인을 설명하는 발표용 그림 4장. 실제 sim 스크린샷 + 데이터 근거 차트.

## 최종 산출물 (`@2x`, 고해상도)

| 파일 | 내용 | 방어하는 질문 |
|---|---|---|
| `fig1_pipeline@2x.png` | 5단계 파이프라인 개요 (입력→검출→추적→예측→안전) | "이 시스템이 뭘 하나" |
| `fig2_why_6_cameras@2x.png` | 안전링 커버리지 4대 vs 6대 (24 seed) | "카메라 왜 6대?" |
| `fig3_why_learned_prediction@2x.png` | 위험진입 recall — CV/Kalman vs LSTM/Transformer | "직선 예측이면 안 되나?" |
| `fig4_dataset_sample@2x.png` | 궤적 학습 데이터 샘플 (전체 씬 + obs8→pred12 윈도우) | "무슨 데이터로 학습했나" |
| `fig5_world_fusion@2x.png` | 멀티카메라 월드 융합 원리 (아핀 → 바닥좌표 → 병합 → 트랙) | "여러 카메라가 본 사람을 어떻게 합치나" |

## 재현

```bash
# 소스 이미지·figures.html이 같은 폴더에 있어야 함
python -m http.server 8791          # 이 폴더에서
# figures.html         → fig1~fig3 (원본), #2x 해시로 2배 렌더
# make_dataset_figure.py → dataset_figure.html 생성 → fig4
python make_dataset_figure.py
```

- 2배 해상도: 브라우저에서 `figures.html#2x` 로 열면 `zoom:2` 적용 → 요소 스크린샷 시 2배 픽셀.
- 차트 색은 색맹 안전 검증(파랑↔주황 ΔE 24.7) 통과 팔레트.
- fig4 좌표·수치 출처: `dataset/trajectories/island_h58_seed10_0009.json` (val split),
  학습 윈도우 수(32,488 / 8,646)는 `dataset/trajectories/README.md`.
- 근거 수치 출처: `docs/chanwoo/nadir-zone-fusion.md` §5-14(커버리지), §5-15(예측기 비교), §5-9(월드 융합).
- fig5는 순수 SVG(외부 이미지 없음): `fig5_world_fusion.html` → `#2x`로 열어 device 스케일 스크린샷.

소스 이미지(재현용): `1_input_multicamera.png` … `5_safety_rings.jpg` — 각 단계 박스에 들어간 실제 sim 캡처.
