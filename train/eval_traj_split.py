"""P0-1 — train/val/test 분리 평가 + scene-level 95% CI (설계 §3-5·§4·§5).

seed 단위 manifest split(train/val/test) 위에서 ADE/FDE·안전 진입 recall/precision 을
**scene 단위 bootstrap CI**와 함께 낸다. val = 모델·운영점 선택 근거, test = 최종 1회.
docs/chanwoo/prediction-eval.md(정확도)·prediction-safety-eval.md(안전)를 재생성한다.

실행:  uv run --group serve python train/eval_traj_split.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trajectory.predictors import ConstantVelocityPredictor, KalmanPredictor   # noqa: E402
from trajectory.sim_predictors import StationHeuristicPredictor                # noqa: E402
from trajectory.learned_predictor import LearnedPredictor, K                   # noqa: E402
from trajectory.evaluator import (ade, fde, enters_radius,                     # noqa: E402
                                  recall_precision, entry_confusion)
from trajectory.bootstrap import scene_bootstrap_ci                            # noqa: E402
from trajectory.sim_traj import load_windows, OBS, PRED                        # noqa: E402

STEP_DT = 0.4
SAFE_R = 3.1                 # 정지반경(SAFE.NOM_STOP)
SAFE_HORIZON_STEPS = 4       # 라이브 제어 1.6s (발표 대표 조건)
B = 2000                     # bootstrap 반복


def _weights():
    w = Path(os.environ.get("PREDICT_MODEL", str(ROOT / "training" / "traj_predictor" / "model.pt")))
    if w.exists():
        return str(w)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=os.environ.get("PREDICT_MODEL_REPO", "chanubc/human-move-lstm"),
                           filename=os.environ.get("PREDICT_MODEL_FILE", "model.pt"))


def _steps_from_path(path):
    return [(STEP_DT * (i + 1), x, z, 0.0) for i, (x, z) in enumerate(path)]


def _ci(scene_lists, statistic):
    """scene별 값 리스트 → (point, lo, hi). scene 단위 복원추출."""
    return scene_bootstrap_ci(scene_lists, statistic, B=B, seed=0)


def _mean_concat(list_of_lists):
    vals = [v for sub in list_of_lists for v in sub if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def _recall(rows):
    tp = sum(r[0] for r in rows); fn = sum(r[2] for r in rows)
    return tp / (tp + fn) if (tp + fn) else float("nan")


def _precision(rows):
    tp = sum(r[0] for r in rows); fp = sum(r[1] for r in rows)
    return tp / (tp + fp) if (tp + fp) else float("nan")


# ── 정확도(ADE/FDE) ──────────────────────────────────────────────────────────
ADE_PREDS = ["등속(const-vel)", "칼만(Kalman)", "스테이션(goal)",
             "학습형 LSTM(최빈)", f"학습형 LSTM(minADE@{K})"]
# 배포 가능한(오라클 아닌) 예측기만 '선택 대표' 후보로. 스테이션(goal)=현재 목표를 앎,
# minADE@K=GT로 최선 모드 고름 → 상한 참고값이라 선택 대상에서 제외한다.
DEPLOYABLE_ACC = ["등속(const-vel)", "칼만(Kalman)", "학습형 LSTM(최빈)"]


def eval_accuracy(split):
    wins = load_windows(split)
    cv = ConstantVelocityPredictor(n_steps=PRED)
    kf = KalmanPredictor(n_steps=PRED)
    sh = StationHeuristicPredictor(n_steps=PRED)
    lp = LearnedPredictor(weights_path=_weights(), device="cpu")
    modes_all = lp.predict_batch([[(o[1], o[2]) for o in w.scene.agents[0].history] for w in wins])

    # scene → predictor → {"ade":[per-window], "fde":[], "ade_m":[], "fde_m":[]}
    per = defaultdict(lambda: {p: {"ade": [], "fde": [], "ade_m": [], "fde_m": []} for p in ADE_PREDS})
    for w, modes in zip(wins, modes_all):
        cv_s = cv.predict(w.scene).per_agent[0][0].steps
        kf_s = kf.predict(w.scene).per_agent[0][0].steps
        sh_s = sh.predict_steps(w.scene.agents[0], w.scene.now, w.scene.horizon, w.goal)
        ml_s = _steps_from_path(modes[0]["path"])
        mink_ade = min(ade(_steps_from_path(m["path"]), w.gt) for m in modes)
        mink_fde = min(fde(_steps_from_path(m["path"]), w.gt) for m in modes)
        vals = {
            "등속(const-vel)": (ade(cv_s, w.gt), fde(cv_s, w.gt)),
            "칼만(Kalman)": (ade(kf_s, w.gt), fde(kf_s, w.gt)),
            "스테이션(goal)": (ade(sh_s, w.gt), fde(sh_s, w.gt)),
            "학습형 LSTM(최빈)": (ade(ml_s, w.gt), fde(ml_s, w.gt)),
            f"학습형 LSTM(minADE@{K})": (mink_ade, mink_fde),
        }
        for p, (a, f) in vals.items():
            d = per[w.scene_id][p]
            d["ade"].append(a); d["fde"].append(f)
            if w.moved:
                d["ade_m"].append(a); d["fde_m"].append(f)

    scenes = list(per.keys())
    out = {"split": split, "n_scenes": len(scenes), "n_windows": len(wins), "preds": {}}
    for p in ADE_PREDS:
        out["preds"][p] = {
            "ade": _ci([per[s][p]["ade"] for s in scenes], _mean_concat),
            "fde": _ci([per[s][p]["fde"] for s in scenes], _mean_concat),
            "ade_moved": _ci([per[s][p]["ade_m"] for s in scenes], _mean_concat),
            "fde_moved": _ci([per[s][p]["fde_m"] for s in scenes], _mean_concat),
        }
    return out


# ── 안전(진입 recall/precision, 라이브 1.6s) ─────────────────────────────────
SAFE_PREDS = ["등속(const-vel)", "칼만(Kalman)", "학습형 LSTM(최빈)", "학습형 LSTM(전모드)"]


def _cur_dist(w):
    last = w.scene.agents[0].history[-1]
    return ((last[1] - w.robot[0]) ** 2 + (last[2] - w.robot[1]) ** 2) ** 0.5


def eval_safety(split, R=SAFE_R, horizon_steps=SAFE_HORIZON_STEPS):
    wins = load_windows(split)
    cv = ConstantVelocityPredictor(n_steps=PRED)
    kf = KalmanPredictor(n_steps=PRED)
    lp = LearnedPredictor(weights_path=_weights(), device="cpu")
    modes_all = lp.predict_batch([[(o[1], o[2]) for o in w.scene.agents[0].history] for w in wins])

    def cut(path):
        return path[:horizon_steps]

    per = defaultdict(lambda: {p: [0, 0, 0] for p in SAFE_PREDS})   # scene → pred → [TP,FP,FN]
    n_anti = n_pos = 0
    for w, modes in zip(wins, modes_all):
        cur = _cur_dist(w)
        if cur < R:
            continue
        n_anti += 1
        gt_xz = cut([(x, z) for (_, x, z) in w.gt])
        gt_entry = enters_radius(gt_xz, w.robot, R)
        if gt_entry:
            n_pos += 1
        cv_xz = cut([(x, z) for (_, x, z, _) in cv.predict(w.scene).per_agent[0][0].steps])
        kf_xz = cut([(x, z) for (_, x, z, _) in kf.predict(w.scene).per_agent[0][0].steps])
        pe = {
            "등속(const-vel)": enters_radius(cv_xz, w.robot, R),
            "칼만(Kalman)": enters_radius(kf_xz, w.robot, R),
            "학습형 LSTM(최빈)": enters_radius(cut(modes[0]["path"]), w.robot, R),
            "학습형 LSTM(전모드)": any(enters_radius(cut(m["path"]), w.robot, R) for m in modes),
        }
        for p in SAFE_PREDS:
            cell = entry_confusion(cur, gt_entry, pe[p], R)
            if cell == "TP": per[w.scene_id][p][0] += 1
            elif cell == "FP": per[w.scene_id][p][1] += 1
            elif cell == "FN": per[w.scene_id][p][2] += 1

    scenes = list(per.keys())
    out = {"split": split, "R": R, "horizon_steps": horizon_steps,
           "n_scenes": len(scenes), "n_anti": n_anti, "n_pos": n_pos, "preds": {}}
    for p in SAFE_PREDS:
        rows = [per[s][p] for s in scenes]
        tp = sum(r[0] for r in rows); fp = sum(r[1] for r in rows); fn = sum(r[2] for r in rows)
        out["preds"][p] = {
            "tp": tp, "fp": fp, "fn": fn,
            "recall": _ci(rows, _recall),
            "precision": _ci(rows, _precision),
        }
    return out


# ── 출력 ──────────────────────────────────────────────────────────────────────
def _f(ci):
    p, lo, hi = ci
    if p != p:
        return "n/a"
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def _acc_table(res):
    lines = ["| 예측기 | ADE(m) 95%CI | FDE(m) 95%CI | ADE(움직임) | FDE(움직임) |",
             "|---|---|---|---|---|"]
    for p in ADE_PREDS:
        d = res["preds"][p]
        lines.append(f"| {p} | {_f(d['ade'])} | {_f(d['fde'])} | {_f(d['ade_moved'])} | {_f(d['fde_moved'])} |")
    return "\n".join(lines)


def _safe_table(res):
    lines = ["| 예측기 | recall 95%CI | precision 95%CI | (TP/FP/FN) |", "|---|---|---|---|"]
    for p in SAFE_PREDS:
        d = res["preds"][p]
        lines.append(f"| {p} | {_f(d['recall'])} | {_f(d['precision'])} | {d['tp']}/{d['fp']}/{d['fn']} |")
    return "\n".join(lines)


def main():
    man = json.loads((ROOT / "dataset" / "trajectories" / "split_manifest.json").read_text(encoding="utf-8"))
    counts = man["meta"]["counts"]

    acc = {s: eval_accuracy(s) for s in ("val", "test")}
    safe = {s: eval_safety(s) for s in ("val", "test")}

    # 운영점/모델 선택 로그(val 기준) — test 전에 고정. 정확도=움직임 ADE 최소(배포 가능 예측기만),
    # 안전=recall 최대. 오라클(minADE@K·스테이션goal)은 후보에서 제외한다.
    def _pt(p):
        v = acc["val"]["preds"][p]["ade_moved"][0]
        return v if v == v else 1e9
    val_champ_acc = min(DEPLOYABLE_ACC, key=_pt)
    val_champ_safe = max(SAFE_PREDS, key=lambda p: (safe["val"]["preds"][p]["recall"][0]
                                                    if safe["val"]["preds"][p]["recall"][0] == safe["val"]["preds"][p]["recall"][0] else -1))
    selection = {"basis": "val", "chosen_accuracy": val_champ_acc, "chosen_safety_operating_point": val_champ_safe,
                 "val_counts_scenes": counts["val"], "note": "test 는 이 선택을 고정한 뒤 1회만 평가"}
    (ROOT / "docs" / "chanwoo" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "chanwoo" / "results" / "oppoint-selection.json").write_text(
        json.dumps({"selection": selection, "val_accuracy": acc["val"], "val_safety": safe["val"]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "docs" / "chanwoo" / "results" / "traj-split-eval.json").write_text(
        json.dumps({"manifest_meta": man["meta"], "accuracy": acc, "safety": safe, "selection": selection},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # 콘솔
    print(f"split scenes: {counts} · 선택(val): 정확도={val_champ_acc} · 안전운영점={val_champ_safe}")
    for s in ("val", "test"):
        print(f"\n[{s}] 정확도 (scene {acc[s]['n_scenes']} · win {acc[s]['n_windows']})\n" + _acc_table(acc[s]))
        print(f"[{s}] 안전 1.6s R={SAFE_R} (대상 {safe[s]['n_anti']} · 진입 {safe[s]['n_pos']})\n" + _safe_table(safe[s]))

    # docs 재생성
    _write_docs(man, acc, safe, selection)
    print("\n표를 docs/chanwoo/prediction-eval.md · prediction-safety-eval.md 에 기록. "
          "선택 로그 results/oppoint-selection.json.")


def _write_docs(man, acc, safe, selection):
    meta = man["meta"]
    head = (f"> 자동 생성: `train/eval_traj_split.py` (P0-1) · 데이터 `dataset/trajectories/`\n"
            f"> seed 단위 split — train {meta['counts']['train']} / val {meta['counts']['val']} / "
            f"test {meta['counts']['test']} scene · 관측 {OBS}(3.2s)/예측 {PRED}(4.8s)\n"
            f"> **val = 모델·운영점 선택 근거, test = 최종 1회.** CI = scene 단위 bootstrap 95% (B={B}).\n"
            f"> manifest: `dataset/trajectories/split_manifest.json` (재현: `python train/make_traj_split.py`)\n")

    acc_doc = ROOT / "docs" / "chanwoo" / "prediction-eval.md"
    acc_doc.write_text(
        "# 궤적 예측 정확도 — ADE/FDE (P0-1: split + scene-level CI)\n\n" + head +
        f"\n선택(val 기준): 정확도 대표 = **{selection['chosen_accuracy']}**\n\n"
        f"## val (선택 근거)\n\n{_acc_table(acc['val'])}\n\n"
        f"## test (최종 1회)\n\n{_acc_table(acc['test'])}\n\n"
        "## 읽는 법\n\n"
        "- 값은 `point [lo, hi]` = scene 단위 bootstrap 95% CI. 겹치면 차이가 유의하지 않을 수 있다.\n"
        "- **움직임** 열 = 예측 구간이 실제로 움직인 윈도우만(정지·지터 제외, 진짜 난이도).\n"
        "- 스테이션(goal)은 현재 목표를 아는 상한 베이스라인. 학습형(minADE@K)는 K모드 중 최선(멀티모달 상한).\n"
        "- test 는 val 선택을 고정한 뒤 1회만 평가했다(운영점 과적합 방지 — 감사 P0-1).\n",
        encoding="utf-8")

    safe_doc = ROOT / "docs" / "chanwoo" / "prediction-safety-eval.md"
    safe_doc.write_text(
        "# 궤적 예측 안전 지표 — 정지반경 진입 recall/precision (P0-1)\n\n" + head +
        f"> 라이브 제어 지평선 1.6s(4스텝) · 정지반경 R={SAFE_R}m. 선제 안전층이라 **recall 우선**.\n\n"
        f"선택(val 기준): 안전 운영점 = **{selection['chosen_safety_operating_point']}**\n\n"
        f"## val (선택 근거) — 대상 {safe['val']['n_anti']} · 실제 진입 {safe['val']['n_pos']}\n\n{_safe_table(safe['val'])}\n\n"
        f"## test (최종 1회) — 대상 {safe['test']['n_anti']} · 실제 진입 {safe['test']['n_pos']}\n\n{_safe_table(safe['test'])}\n\n"
        "## 읽는 법\n\n"
        "- \"지금 반경 **밖**의 사람이 1.6s 안에 반경 안으로 진입할지\"를 미리 맞혔나. 반경 안은 반응형 몫이라 제외.\n"
        "- **recall**=실제 진입 중 미리 잡은 비율(놓치면 충돌) · **precision**=경보 중 진짜 비율(낮으면 헛정지).\n"
        "- **최빈**=최상위 단일 모드 · **전모드**=K모드 합집합(보수적, recall↑). 이 둘이 운영점 후보다.\n"
        "- CI = scene 단위 bootstrap 95%. test 는 val 선택 고정 후 1회.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
