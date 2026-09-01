Robot Kitchen Safety Sim - Data Augmentation Capture Package
============================================================

목적
----
이 폴더는 발표자료(PowerPoint)에 바로 사용할 수 있도록
robot-kitchen-safety-sim의 데이터 생성 화면에서 실제로 추출한 결과를 정리한 것입니다.

기준 소스
--------
- Repository: https://github.com/whatslung/robot-kitchen-safety-sim
- Branch: main
- Commit: 1d0399a8b947d0130f0d81d0c4ba720579b09fad

폴더 구성
--------
- 01_ppt_ready
  발표자료에 바로 삽입할 수 있는 비교 이미지입니다.
  4분할 이미지의 배치는 LAYOUT.txt에 설명되어 있습니다.

- 02_normal_sets
  화재가 없는 정상 장면 6세트입니다.

- 03_fire_sets
  화재 진행 단계가 다른 장면 4세트입니다.

- 04_sensor_rgb_examples
  Clean, Standard CCTV, Low-cost CCTV, Night 조건의 RGB 비교 예시입니다.
  이 폴더의 제약사항은 NOTICE.txt를 확인하십시오.

- 05_by_output_type
  PPT에서 같은 종류의 이미지만 골라 쓰기 쉽도록 RGB, class mask,
  instance mask, depth 6 m, depth-near 3 m 기준으로 다시 모은 폴더입니다.

세트별 파일
----------
각 normal/fire 세트에는 아래 7개 파일이 있습니다.

1. 01_rgb.png                 원본 RGB 화면
2. 02_class_mask.png          클래스별 색상 마스크
3. 03_instance_mask.png       객체 인스턴스별 색상 마스크
4. 04_depth_6m.png            최대 6 m 기준 깊이 영상
5. 05_depth_near_3m.png       가까운 영역 3 m 기준 깊이 영상
6. 06_yolo_labels.txt         YOLO 형식 라벨
7. 07_metadata.json           생성 조건 및 객체 메타데이터

검증 결과
--------
- paired normal/fire sets: 10
- files per paired set: 7
- paired image resolution: 960 x 720
- unique paired RGB images: 10 / 10
- folder and file names: ASCII only
- image composites: visually inspected

중요 사항
--------
현재 main에서 저해상도 센서 프리셋을 사용하면 출력 단계 사이에 렌더 크기가
달라질 수 있습니다. 서로 다른 해상도의 결과를 임의로 늘리거나 자르지 않았습니다.
따라서 완전한 7종 paired set은 Clean sensor 조건으로 만들었고,
센서 열화 조건은 04_sensor_rgb_examples에 RGB 예시만 별도로 정리했습니다.

Windows 전달 호환성
------------------
최상위 폴더, 모든 하위 폴더, 파일 이름, ZIP 내부 경로에는
한글이나 특수 문자를 사용하지 않았습니다.

