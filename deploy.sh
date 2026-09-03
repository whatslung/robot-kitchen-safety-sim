#!/usr/bin/env bash
# Cloud Run 배포 — detect_server 를 "예측 전용(경량)" 서비스로 올린다.
# 배포 형상(리전·min/max·메모리 등)을 코드로 고정해 재배포 시 플래그를 기억할 필요가 없게 한다.
#
# 사전 준비(1회):
#   1) gcloud CLI 설치            (winget install --id Google.CloudSDK -e)
#   2) gcloud auth login          (본인 Google 계정)
#   3) gcloud config set project <PROJECT_ID>   (결제 연결된 GCP 프로젝트)
#   ※ 처음 배포 시 필요한 API(run·cloudbuild·artifactregistry) 활성화·저장소 생성을
#     물어보면 y 로 진행.
#
# 사용:  bash deploy.sh
#   (Windows 는 Git Bash / WSL / Cloud Shell 에서 실행. 환경변수로 덮어쓸 수 있다:
#    SERVICE=... REGION=... bash deploy.sh)
set -euo pipefail

SERVICE="${SERVICE:-detect-server}"
REGION="${REGION:-asia-northeast3}"   # 서울

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 1 \
  --max-instances 3 \
  --port 8080

echo
echo "배포 완료. 위 출력의 Service URL 로 검증:"
echo "  curl <SERVICE_URL>/health   # {\"status\":\"ok\",\"mode\":\"off\",\"predict_net\":\"lstm\"} 기대"
