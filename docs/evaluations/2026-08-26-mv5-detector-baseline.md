# 4+1뷰 기존 YOLO 사람 검출 1차 baseline

## 결론

현재 배포용 나디르 파인튜닝 가중치는 새 `mvNW`, `mvNE`, `mvSW`, `mvSE`, `mvCenter`
뷰에서 ByteTrack의 입력 검출기로 사용할 수 없다. 운영 confidence `0.25`, IoU `0.5`에서
person precision과 recall이 모두 `0.000`이었다. confidence를 `0.05`까지 낮춰도 recall은
`0.029`에 그쳤다.

Stock COCO `yolo11s.pt`가 현재 배포 가중치보다 낫지만 운영 confidence `0.25`에서
precision `0.174`, recall `0.114`로 여전히 부족하다. 따라서 이 PR의 Homography,
글로벌 ID, 미래예측 경로는 런타임 수직 슬라이스로는 유효하지만, 4+1뷰 detector를
재학습하기 전에는 실제 트랙 품질이나 회피 입력 품질을 주장하지 않는다.

## 평가 설정

- 날짜: 2026-08-26
- 입력: 동기화된 SIM 장면 10개 × 5개 뷰 = 50 PNG
- 해상도: 960×720, 추론 `imgsz=640`
- GT: 이미지별 YOLO person 박스 35개
- 매칭: confidence 내림차순 one-to-one greedy, IoU ≥ 0.5
- 장치: CPU
- 추론 인자: `batch=8`, `min_confidence=0.01`, `nms_iou=0.7`
- 기존 모델: `chanubc/robot-kitchen-nadir-yolo11s/best.pt`
- 비교 모델: stock COCO `yolo11s.pt`
- 실행 코드: `train/eval_multiview_detector.py`
- 증거: [데이터 체크섬·장면 매니페스트](artifacts/mv5-baseline-manifest.json),
  [배포 모델 평가 JSON](artifacts/mv5-deploy-detector-report.json),
  [stock 모델 평가 JSON](artifacts/mv5-stock-yolo11s-report.json)

이 데이터는 빠른 go/no-go용 1차 표본이다. 앞 6개 장면에는 환경 설비 랜덤화가 포함되고,
뒤 4개 장면은 렌더 시간 단축을 위해 설비 배치 지터만 끈 상태다. 사람 위치·포즈·인원,
화재와 센서 변화는 유지했다. 최종 성능 인용에는 pose를 고정한 200장면 이상의 별도
val/test 세트를 사용해야 한다.

최초 브라우저 캡처 하네스가 `Math.random` seed와 장면별 실제 인원 수를 저장하지 않은
한계도 있다. 따라서 저장소만으로 pixel-identical 재생성할 수 없으며 이 수치를 재현 가능한
성능 벤치마크로 인용하면 안 된다. 대신 매니페스트에 원본 100개 이미지·라벨을 식별하는
전체/장면별 SHA-256을 남겼다. 다음 정식 baseline은 seed, 사람 수, pose, 환경 설정을 장면별로
저장하고 scene 단위 split 전에 매니페스트를 고정한다.

## 전체 결과

| 모델 | confidence | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 기존 나디르 파인튜닝 | 0.05 | 1 | 111 | 34 | 0.009 | 0.029 | 0.014 |
| 기존 나디르 파인튜닝 | 0.25 | 0 | 18 | 35 | 0.000 | 0.000 | 0.000 |
| 기존 나디르 파인튜닝 | 0.35 | 0 | 6 | 35 | 0.000 | 0.000 | 0.000 |
| Stock YOLO11s | 0.10 | 7 | 40 | 28 | 0.149 | 0.200 | 0.171 |
| Stock YOLO11s | 0.25 | 4 | 19 | 31 | 0.174 | 0.114 | 0.138 |
| Stock YOLO11s | 0.35 | 4 | 17 | 31 | 0.190 | 0.114 | 0.143 |

기존 모델의 best-F1도 confidence `0.12`에서 precision `0.020`, recall `0.029`뿐이었다.
Stock 모델의 best-F1은 confidence `0.20`에서 precision `0.207`, recall `0.171`이었다.

## 카메라별 진단

Stock YOLO11s의 운영 confidence `0.25` 결과다.

| 카메라 | GT person | TP | precision | recall |
|---|---:|---:|---:|---:|
| `mvNW` | 5 | 0 | 0.000 | 0.000 |
| `mvNE` | 3 | 0 | 0.000 | 0.000 |
| `mvSW` | 1 | 0 | 0.000 | 0.000 |
| `mvSE` | 12 | 1 | 0.250 | 0.083 |
| `mvCenter` | 14 | 3 | 0.429 | 0.214 |

기존 나디르 파인튜닝 모델은 같은 threshold에서 모든 카메라의 TP가 0이었다. 중앙 보조뷰가
상대적으로 더 많은 TP를 낸 예비 신호는 있지만, 10장면의 작고 불균형한 표본만으로 카메라
설계의 우위를 확인했다고 해석할 수 없다. 현재 4개 코너뷰의 GT 분포도 불균형하다.

GT 가시성만 본 10장면의 person-positive 장면 수는 `mvNW=3`, `mvNE=3`, `mvSW=1`,
`mvSE=6`, `mvCenter=8`이었다. 코너 4개 중 하나라도 person GT가 있던 장면은 7/10,
중앙뷰를 추가하면 8/10이었다. 모든 5뷰가 비어 있던 장면도 2/10이었다. 표본이 작아 최종
coverage 수치로 인용할 수는 없지만, detector 학습 전에 카메라 조준점과 랜덤 사람 배치
범위를 다시 탐색해야 한다는 충분한 신호다.

## 결정과 다음 gate

1. 현재 배포 가중치로 ByteTrack과 글로벌 융합 성능을 평가하지 않는다. 입력 recall이 0에
   가까우므로 tracker 튜닝으로 복구할 수 없다.
2. GT 기반 카메라 ablation으로 5개 pose를 다시 조정한다. 최소 기준은 평가 동선에서
   all-view blind 장면 0, 카메라별 person-positive 표본의 심한 쏠림 제거다.
3. pose를 고정한 뒤 동기 4+1뷰 데이터 200장면 이상을 생성하고 scene 단위로 split한다.
4. 새 detector는 stock YOLO11s에서 시작해 4+1뷰로 파인튜닝한다. 나디르 체크포인트는
   비교군으로만 둔다.
5. held-out SIM에서 per-view recall ≥ 0.85, precision ≥ 0.80, 4+1 fused recall ≥ 0.95를
   만족한 뒤 ByteTrack·Homography 글로벌 ID·미래예측·회피 PR을 순서대로 연결한다.

## 재현 명령

```powershell
uv run --group serve python train/eval_multiview_detector.py `
  --dataset dataset/mv5-baseline-20260826 `
  --output training/mv5_detector_baseline.json `
  --device cpu --imgsz 640 --batch 8 `
  --min-confidence 0.01 --nms-iou 0.7 --match-iou 0.5

uv run --group serve python train/eval_multiview_detector.py `
  --dataset dataset/mv5-baseline-20260826 `
  --model yolo11s.pt `
  --output training/mv5_detector_stock_baseline.json `
  --device cpu --imgsz 640 --batch 8 `
  --min-confidence 0.01 --nms-iou 0.7 --match-iou 0.5
```

`dataset/`, `training/`, 다운로드한 `.pt`는 저장소의 기존 `.gitignore` 규칙에 따라 로컬에만
유지한다. 절대 경로와 중복 상세를 제거한 평가 JSON 및 원본 식별용 체크섬 매니페스트는 위
`docs/evaluations/artifacts/`에 커밋했다.
