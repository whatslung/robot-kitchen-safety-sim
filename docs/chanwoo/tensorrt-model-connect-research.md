# NVIDIA TensorRT-Model-Connect 조사 메모

> 조사일: 2026-08-20  
> 근거: NVIDIA가 운영하는 공식 문서와 공식 GitHub 저장소만 사용했다. 이 메모는 프로젝트 적용 판단을 위한 기술 조사이며, 프로젝트 코드 자체의 분석은 포함하지 않는다.

## 결론

TensorRT-Model-Connect(TRTMC)는 **지원되는 Hugging Face 또는 로컬 체크포인트를 TensorRT 엔진과 버전 지정 `.bundle`로 빌드하고, C++ 작업 API로 실행하게 하는 NVIDIA의 모델 패밀리별 참조 구현 모음**이다. 빌드 시에는 Python이 체크포인트 해석과 TensorRT 엔진 생성을 맡고, 일반적인 native 프로필의 실행 시에는 PyTorch 없이 C++로 추론한다. [NVIDIA 프로젝트 개요](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/project-overview/)

따라서 TRTMC는 ONNX를 대체하는 범용 교환 표준이 아니다. 지원 모델에서는 `PyTorch → ONNX → TensorRT` 사이의 ONNX 내보내기를 생략하는 **별도 TensorRT 통합 경로**다. NVIDIA도 프레임워크 사이의 이식성이 우선이거나 ONNX 산출물이 필수인 경우 ONNX를 쓰라고 명시한다. [NVIDIA 프로젝트 개요](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/project-overview/)

## ONNX와의 관계

| 질문 | 답 |
| --- | --- |
| TRTMC 빌드에 ONNX export가 필요한가? | 지원 체크포인트의 TRTMC 경로에서는 아니다. 모델 패밀리별 builder가 TensorRT API로 직접 컴파일한다. |
| ONNX를 없애도 되는가? | 그 모델의 정확한 TRTMC profile이 지원되고, C++ `.bundle` 배포가 목적일 때에만 검토할 수 있다. |
| ONNX를 대체하는가? | 아니다. ONNX는 여러 원본 프레임워크 사이의 휴대 가능한 교환 형식이고, TRTMC는 NVIDIA TensorRT용 모델별 참조·배포 경로다. |
| PyTorch만으로 충분한 배포에 필요한가? | 아니다. NVIDIA는 PyTorch 내부에 계속 둘 경우 Torch-TensorRT, ONNX 산출물이 필요한 경우 ONNX를 각각 권장한다. |

근거: [TensorRT 경로 선택 표](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/project-overview/), [NVIDIA TensorRT ONNX 경로 안내](https://github.com/NVIDIA/TensorRT/blob/main/documents/import_workflows.md).

## 지원 입력·프레임워크·작업 범위

- 공개 인터페이스의 출발점은 지원되는 Hugging Face 모델 ID, 로컬 체크포인트(일부 Diffusers 모델 디렉터리 포함)다. 임의의 일반 PyTorch 모듈을 자동 변환하는 범용 변환기는 아니다. 정확한 지원 단위는 `HF ID × 체크포인트 해석 × TRTMC profile × 빌드 설정`이다. 같은 모델 계열의 미검증 fine-tune은 best-effort일 뿐, 지원 보장이 아니다. [지원 모델 해석법](https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/overview/)
- 모델·작업 범위는 텍스트 생성, 임베딩·재순위화, 번역, 비전 언어·OCR, 음성 인식·합성, 이미지·비디오 확산 생성, 분할, 시계열 예측, 신경 연산자까지 넓다. 다만 이 목록은 모든 모델·모든 하드웨어에서 검증됐다는 뜻이 아니므로, 반드시 지원 표의 **정확한** 체크포인트 행을 확인해야 한다. [프로젝트 개요](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/project-overview/), [지원 행렬](https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/overview/)
- 2026-07-29 NVIDIA GB300 비교 스냅샷에는 76개 모델 계열의 단일 프로세스 프로필 105개가 들어 있다. 이는 해당 날짜·하드웨어·설정에서의 성능 비교 근거이지, 일반 호환성 보장은 아니다. [지원 모델의 릴리스 성능 스냅샷](https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/overview/)

## 플랫폼과 운영 제약

- 릴리스 wheel 경로: Linux `aarch64`, Python 3.10 또는 3.12, glibc 2.39 이상, NVIDIA TensorRT 11.1.0.106 조합이다.
- Linux `x86_64`는 현재 릴리스 wheel이 없으며, Docker와 NVIDIA Container Toolkit을 사용하는 소스 빌드 경로를 써야 한다. 소스 빌드는 Linux `x86_64`와 `aarch64`를 문서화한다.
- 실행 환경은 호환되는 NVIDIA 드라이버, CUDA/TensorRT 세대(cohort), 동적 로더와 시스템 라이브러리가 필요하다. 다른 아키텍처나 TensorRT 세대의 wheel·DSO·bundle·TensorRT 라이브러리를 섞지 말아야 한다.
- 일부 프로필은 C++ native 실행 대신 보조 Python 실행 파일을 쓰는 혼합형(hybrid)이며, 그 실행 의존성은 E2E manifest에 선언된다.

근거: [시스템 요구 사항](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/environment-and-repro/), [설치 안내](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/installation/), [빌드·실행 경계](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/project-overview/).

## 성숙도 판단

공개 NVIDIA 저장소의 패키지 기본 버전은 `0.1.0`이고, NVIDIA는 TRTMC를 명시적으로 **reference implementation(참조 구현)** 이라고 부른다. 또한 NVIDIA의 선택 기준은 “빠른 초기 탐색과 넓은 모델 범위”에는 TRTMC를, 엣지 환경에서 성능을 최우선으로 하는 운영 LLM/VLM에는 TensorRT Edge-LLM을 먼저 쓰라는 것이다. 따라서 현재의 합리적인 판단은 **기능과 지원 모델 수는 빠르게 늘고 있지만, 핵심 운영 경로에 무조건 투입할 안정화된 범용 SDK로 보기는 이르다**이다. [NVIDIA 프로젝트 개요](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/project-overview/), [공식 패키지 설정](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/pyproject.toml)

## 일반적인 통합 방법

1. **정확한 지원 대상 확인**: 사용할 HF ID, revision, TRTMC profile, 정밀도, 대상 GPU 조합을 지원 표에서 확인한다. 지원 표에 없는 체크포인트나 fine-tune은 검증 지원으로 간주하지 않는다.
2. **환경 선택**: aarch64라면 제약을 만족하는 release wheel, x86_64라면 NVIDIA GPU가 보이는 Docker 기반 소스 빌드를 선택한다.
3. **bundle 생성**: 예를 들어 Qwen의 경우 `trtmc build Qwen/Qwen3-0.6B --precision bf16 --max-cache-length 16384 --output qwen3-0.6b.bundle`처럼 빌드한다. 첫 빌드에서는 모델 다운로드와 TensorRT 엔진 컴파일이 일어난다.
4. **검사와 검증**: `trtmc inspect`와 `trtmc inspect --list-engines`로 모델 계열, 런타임 전략, 정밀도, 엔진을 확인하고, `trtmc run`으로 기준 입출력을 확인한다.
5. **네이티브 앱에 연결**: 생성된 `.bundle`을 C++에서 `trtmc::load()`로 읽고, 모델 작업에 맞는 API(예: `generate()`, `transcribe()`, `embed()`)를 호출한다. 즉 Python은 빌드 쪽, C++ 애플리케이션은 bundle 실행 쪽으로 경계가 나뉜다.

공식 예시와 API: [첫 NLP 추론](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/quick-start/), [C++ API](https://nvidia.github.io/TensorRT-Model-Connect/api/cpp-api/), [프로젝트 README의 두 명령 예시](https://github.com/NVIDIA/TensorRT-Model-Connect#-example-code).

## 도입 전 확인 목록

- 목표 모델이 지원 표의 정확한 checkpoint/profile인가?
- 배포 대상이 Linux와 호환 NVIDIA GPU·드라이버·TensorRT 조합인가?
- wheel 제약 또는 Docker 기반 소스 빌드를 수용할 수 있는가?
- ONNX가 다른 런타임·프레임워크와의 교환 산출물로 필요한가? 필요하면 TRTMC로 치환하지 않는다.
- C++ bundle 경계가 실제 애플리케이션 구조에 맞는가? Python/PyTorch 서비스 안에 계속 둘 계획이면 TRTMC의 이점은 작을 수 있다.
- 참조 구현·정확 모델별 지원이라는 현재 성격을 수용할 수 있는가? 운영 성능이 최우선인 엣지 LLM/VLM이면 TensorRT Edge-LLM도 함께 비교한다.

## 공식 1차 자료

- [NVIDIA TensorRT-Model-Connect 공식 저장소](https://github.com/NVIDIA/TensorRT-Model-Connect)
- [NVIDIA 프로젝트 개요](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/project-overview/)
- [NVIDIA 지원 모델·성능 스냅샷](https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/overview/)
- [NVIDIA 시스템 요구 사항](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/environment-and-repro/)
- [NVIDIA 설치 안내](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/installation/)
