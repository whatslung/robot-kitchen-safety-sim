"""궤적 예측 코어 — cooking-robot-safety/trajectory 에서 이식(2026-08-19, 이슈 #2 3단계).

types/predictors/evaluator 는 원본을 **바이트 그대로** 가져왔다(검증된 등속·칼만·ADE/FDE).
sim 전용 코드(로더 sim_traj, 스테이션 휴리스틱 sim_predictors)는 여기서 새로 얹는다.
원본이 바뀌면 이 사본은 별도로 관리(포크)된다 — 설계 스펙 참조.
"""
