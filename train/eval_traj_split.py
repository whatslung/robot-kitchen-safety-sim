"""P0-1 — train/val/test 분리 평가 + scene-level 95% CI (설계 §3-5·§4·§5).

seed 단위 manifest split(train/val/test) 위에서 ADE/FDE·안전 진입 recall/precision 을
**scene 단위 bootstrap CI**와 함께 낸다. val = 모델·운영점 선택 근거, test = 최종 1회.
학습형 백본은 있는 대로 자동 비교(LSTM + Transformer). docs/chanwoo/prediction-eval.md(정확도)·
prediction-safety-eval.md(안전)를 재생성한다.

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
from trajectory.learned_predictor import (LearnedPredictor, build_transformer_net,  # noqa: E402
                                          build_cvae_net, K)
from trajectory.evaluator import ade, fde, enters_radius, entry_confusion      # noqa: E402
from trajectory.bootstrap import scene_bootstrap_ci                            # noqa: E402
from trajectory.sim_traj import load_windows, OBS, PRED                        # noqa: E402

STEP_DT = 0.4
SAFE_R = 3.1                 # 정지반경(SAFE.NOM_STOP)
SAFE_HORIZON_STEPS = 4       # 라이브 제어 1.6s (발표 대표 조건)
H_LIVE = 4                   # 라이브 1.6s(4스텝) — 안전 결정 지평선
B = 2000                     # bootstrap 반복
_TRAJ = ROOT / "training" / "traj_predictor"


def _lstm_weights():
    w = Path(os.environ.get("PREDICT_MODEL", str(_TRAJ / "model.pt")))
    if w.exists():
        return str(w)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=os.environ.get("PREDICT_MODEL_REPO", "chanubc/human-move-lstm"),
                           filename=os.environ.get("PREDICT_MODEL_FILE", "model.pt"))


def learned_predictors():
    """사용 가능한 학습형 백본 → {이름: LearnedPredictor}. 백본만 다르고 head·정규화는 동일."""
    out = {"LSTM": LearnedPredictor(weights_path=_lstm_weights(), device="cpu")}
    tf = _TRAJ / "model_transformer.pt"
    if tf.exists():
        out["Transformer"] = LearnedPredictor(net=build_transformer_net(), weights_path=str(tf), device="cpu")
    else:
        print(f"[eval] Transformer 가중치 없음({tf}) → 생략. `python train/train_traj_transformer.py`.")
    cv = _TRAJ / "model_cvae.pt"
    if cv.exists():
        out["CVAE"] = LearnedPredictor(net=build_cvae_net(), weights_path=str(cv), device="cpu")
    else:
        print(f"[eval] CVAE 가중치 없음({cv}) → 생략. `python train/train_traj_cvae.py`.")
    return out


def _steps_from_path(path):
    return [(STEP_DT * (i + 1), x, z, 0.0) for i, (x, z) in enumerate(path)]


def _ci(scene_lists, statistic):
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


BASE_ACC = ["등속(const-vel)", "칼만(Kalman)", "스테이션(goal)"]


def acc_names(learned):
    names = list(BASE_ACC)
    for k in learned:
        names += [f"학습형 {k}(최빈)", f"학습형 {k}(minADE@{K})"]
    return names


def deployable_acc(learned):
    """오라클(스테이션goal·minADE@K) 제외 — 배포 가능한 예측기만 '선택 대표' 후보."""
    return ["등속(const-vel)", "칼만(Kalman)"] + [f"학습형 {k}(최빈)" for k in learned]


# ── 정확도(ADE/FDE) ──────────────────────────────────────────────────────────
def eval_accuracy(split, learned):
    wins = load_windows(split)
    cv = ConstantVelocityPredictor(n_steps=PRED)
    kf = KalmanPredictor(n_steps=PRED)
    sh = StationHeuristicPredictor(n_steps=PRED)
    hists = [[(o[1], o[2]) for o in w.scene.agents[0].history] for w in wins]
    modes_by = {name: lp.predict_batch(hists) for name, lp in learned.items()}

    names = acc_names(learned)
    # 두 지평선을 함께 낸다: 4.8s(전체 12스텝) + 라이브 1.6s(4스텝, 안전 결정 지평선).
    fields = ["ade", "fde", "ade_m", "fde_m", "ade16", "fde16", "ade16_m", "fde16_m"]
    per = defaultdict(lambda: {p: {k: [] for k in fields} for p in names})

    def _af(steps, gt):
        return (ade(steps, gt), fde(steps, gt),
                ade(steps[:H_LIVE], gt[:H_LIVE]), fde(steps[:H_LIVE], gt[:H_LIVE]))

    def _af_min(mode_steps, gt):        # minADE@K: 모드 중 최선 (각 지평선 독립)
        return (min(ade(s, gt) for s in mode_steps), min(fde(s, gt) for s in mode_steps),
                min(ade(s[:H_LIVE], gt[:H_LIVE]) for s in mode_steps),
                min(fde(s[:H_LIVE], gt[:H_LIVE]) for s in mode_steps))

    for i, w in enumerate(wins):
        cv_s = cv.predict(w.scene).per_agent[0][0].steps
        kf_s = kf.predict(w.scene).per_agent[0][0].steps
        sh_s = sh.predict_steps(w.scene.agents[0], w.scene.now, w.scene.horizon, w.goal)
        vals = {"등속(const-vel)": _af(cv_s, w.gt), "칼만(Kalman)": _af(kf_s, w.gt),
                "스테이션(goal)": _af(sh_s, w.gt)}
        for name in learned:
            mode_steps = [_steps_from_path(m["path"]) for m in modes_by[name][i]]
            vals[f"학습형 {name}(최빈)"] = _af(mode_steps[0], w.gt)
            vals[f"학습형 {name}(minADE@{K})"] = _af_min(mode_steps, w.gt)
        for p, (a, f, a16, f16) in vals.items():
            d = per[w.scene_id][p]
            d["ade"].append(a); d["fde"].append(f); d["ade16"].append(a16); d["fde16"].append(f16)
            if w.moved:
                d["ade_m"].append(a); d["fde_m"].append(f)
                d["ade16_m"].append(a16); d["fde16_m"].append(f16)

    scenes = list(per.keys())
    out = {"split": split, "n_scenes": len(scenes), "n_windows": len(wins), "order": names, "preds": {}}
    for p in names:
        out["preds"][p] = {k.replace("_m", "_moved") if k.endswith("_m") else k:
                           _ci([per[s][p][k] for s in scenes], _mean_concat) for k in fields}
    return out


# ── 안전(진입 recall/precision, 라이브 1.6s) ─────────────────────────────────
def safe_names(learned):
    names = ["등속(const-vel)", "칼만(Kalman)"]
    for k in learned:
        names += [f"학습형 {k}(최빈)", f"학습형 {k}(전모드)"]
    return names


def _cur_dist(w):
    last = w.scene.agents[0].history[-1]
    return ((last[1] - w.robot[0]) ** 2 + (last[2] - w.robot[1]) ** 2) ** 0.5


def eval_safety(split, learned, R=SAFE_R, horizon_steps=SAFE_HORIZON_STEPS):
    wins = load_windows(split)
    cv = ConstantVelocityPredictor(n_steps=PRED)
    kf = KalmanPredictor(n_steps=PRED)
    hists = [[(o[1], o[2]) for o in w.scene.agents[0].history] for w in wins]
    modes_by = {name: lp.predict_batch(hists) for name, lp in learned.items()}

    def cut(path):
        return path[:horizon_steps]

    names = safe_names(learned)
    per = defaultdict(lambda: {p: [0, 0, 0] for p in names})   # scene → pred → [TP,FP,FN]
    n_anti = n_pos = 0
    for i, w in enumerate(wins):
        cur = _cur_dist(w)
        if cur < R:
            continue
        n_anti += 1
        gt_entry = enters_radius(cut([(x, z) for (_, x, z) in w.gt]), w.robot, R)
        if gt_entry:
            n_pos += 1
        cv_xz = cut([(x, z) for (_, x, z, _) in cv.predict(w.scene).per_agent[0][0].steps])
        kf_xz = cut([(x, z) for (_, x, z, _) in kf.predict(w.scene).per_agent[0][0].steps])
        pe = {"등속(const-vel)": enters_radius(cv_xz, w.robot, R),
              "칼만(Kalman)": enters_radius(kf_xz, w.robot, R)}
        for name in learned:
            modes = modes_by[name][i]
            pe[f"학습형 {name}(최빈)"] = enters_radius(cut(modes[0]["path"]), w.robot, R)
            pe[f"학습형 {name}(전모드)"] = any(enters_radius(cut(m["path"]), w.robot, R) for m in modes)
        for p in names:
            cell = entry_confusion(cur, gt_entry, pe[p], R)
            if cell == "TP": per[w.scene_id][p][0] += 1
            elif cell == "FP": per[w.scene_id][p][1] += 1
            elif cell == "FN": per[w.scene_id][p][2] += 1

    scenes = list(per.keys())
    out = {"split": split, "R": R, "horizon_steps": horizon_steps, "order": names,
           "n_scenes": len(scenes), "n_anti": n_anti, "n_pos": n_pos, "preds": {}}
    for p in names:
        rows = [per[s][p] for s in scenes]
        out["preds"][p] = {
            "tp": sum(r[0] for r in rows), "fp": sum(r[1] for r in rows), "fn": sum(r[2] for r in rows),
            "recall": _ci(rows, _recall), "precision": _ci(rows, _precision),
        }
    return out


# ── 출력 ──────────────────────────────────────────────────────────────────────
def _f(ci):
    p, lo, hi = ci
    return "n/a" if p != p else f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def _acc_table(res):
    lines = ["| 예측기 | ADE(m) 95%CI | FDE(m) 95%CI | ADE(움직임) | FDE(움직임) |",
             "|---|---|---|---|---|"]
    for p in res["order"]:
        d = res["preds"][p]
        lines.append(f"| {p} | {_f(d['ade'])} | {_f(d['fde'])} | {_f(d['ade_moved'])} | {_f(d['fde_moved'])} |")
    return "\n".join(lines)


def _acc16_table(res):
    """라이브 1.6s(안전 결정 지평선) 정확도 — 안전 recall/precision과 같은 지평선."""
    lines = ["| 예측기 | ADE@1.6s 95%CI | FDE@1.6s 95%CI | ADE@1.6s(움직임) | FDE@1.6s(움직임) |",
             "|---|---|---|---|---|"]
    for p in res["order"]:
        d = res["preds"][p]
        lines.append(f"| {p} | {_f(d['ade16'])} | {_f(d['fde16'])} | "
                     f"{_f(d['ade16_moved'])} | {_f(d['fde16_moved'])} |")
    return "\n".join(lines)


def _safe_table(res):
    lines = ["| 예측기 | recall 95%CI | precision 95%CI | (TP/FP/FN) |", "|---|---|---|---|"]
    for p in res["order"]:
        d = res["preds"][p]
        lines.append(f"| {p} | {_f(d['recall'])} | {_f(d['precision'])} | {d['tp']}/{d['fp']}/{d['fn']} |")
    return "\n".join(lines)


def main():
    man = json.loads((ROOT / "dataset" / "trajectories" / "split_manifest.json").read_text(encoding="utf-8"))
    counts = man["meta"]["counts"]
    learned = learned_predictors()

    acc = {s: eval_accuracy(s, learned) for s in ("val", "test")}
    safe = {s: eval_safety(s, learned) for s in ("val", "test")}

    # 선택(val 기준) — 정확도=움직임 ADE 최소(배포 가능만), 안전=recall 최대. test 전에 고정.
    def _pt(p):
        v = acc["val"]["preds"][p]["ade_moved"][0]
        return v if v == v else 1e9
    val_champ_acc = min(deployable_acc(learned), key=_pt)
    val_champ_safe = max(safe["val"]["order"],
                         key=lambda p: (safe["val"]["preds"][p]["recall"][0]
                                        if safe["val"]["preds"][p]["recall"][0] == safe["val"]["preds"][p]["recall"][0] else -1))
    selection = {"basis": "val", "chosen_accuracy": val_champ_acc,
                 "chosen_safety_operating_point": val_champ_safe,
                 "learned_backbones": list(learned), "val_counts_scenes": counts["val"],
                 "note": "test 는 이 선택을 고정한 뒤 1회만 평가"}
    (ROOT / "docs" / "chanwoo" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "chanwoo" / "results" / "oppoint-selection.json").write_text(
        json.dumps({"selection": selection, "val_accuracy": acc["val"], "val_safety": safe["val"]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "docs" / "chanwoo" / "results" / "traj-split-eval.json").write_text(
        json.dumps({"manifest_meta": man["meta"], "accuracy": acc, "safety": safe, "selection": selection},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"split scenes: {counts} · 백본: {list(learned)} · 선택(val): 정확도={val_champ_acc} · 안전={val_champ_safe}")
    for s in ("val", "test"):
        print(f"\n[{s}] 정확도 4.8s (scene {acc[s]['n_scenes']} · win {acc[s]['n_windows']})\n" + _acc_table(acc[s]))
        print(f"[{s}] 정확도 1.6s(라이브)\n" + _acc16_table(acc[s]))
        print(f"[{s}] 안전 1.6s R={SAFE_R} (대상 {safe[s]['n_anti']} · 진입 {safe[s]['n_pos']})\n" + _safe_table(safe[s]))

    _write_docs(man, acc, safe, selection, list(learned))
    print("\n표를 docs/chanwoo/prediction-eval.md · prediction-safety-eval.md 에 기록. "
          "선택 로그 results/oppoint-selection.json.")


def _write_docs(man, acc, safe, selection, backbones):
    meta = man["meta"]
    head = (f"> 자동 생성: `train/eval_traj_split.py` (P0-1) · 데이터 `dataset/trajectories/`\n"
            f"> seed 단위 split — train {meta['counts']['train']} / val {meta['counts']['val']} / "
            f"test {meta['counts']['test']} scene · 관측 {OBS}(3.2s)/예측 {PRED}(4.8s)\n"
            f"> **val = 모델·운영점 선택 근거, test = 최종 1회.** CI = scene 단위 bootstrap 95% (B={B}).\n"
            f"> 학습형 백본 비교: {', '.join(backbones)} — head·손실·정규화·split·eval 동일, **백본만 교체**\n"
            f"> (Trajectron++식 멀티모달+불확실성 출력 · ≠ Trajectron++ 실행). "
            f"manifest 재현: `python train/make_traj_split.py`\n")

    (ROOT / "docs" / "chanwoo" / "prediction-eval.md").write_text(
        "# 궤적 예측 정확도 — ADE/FDE (P0-1 split·CI · 백본 비교)\n\n" + head +
        f"\n선택(val 기준): 정확도 대표 = **{selection['chosen_accuracy']}**\n\n"
        "## 4.8s 지평선 (전체 예측 12스텝)\n\n"
        f"### val (선택 근거)\n\n{_acc_table(acc['val'])}\n\n"
        f"### test (최종 1회)\n\n{_acc_table(acc['test'])}\n\n"
        "## 1.6s 지평선 (라이브 제어 4스텝 — 안전 결정과 같은 지평선)\n\n"
        "> 로봇이 실제 정지·감속 판단에 쓰는 지평선. 4.8s ADE/FDE는 안전 기준으론 과대평가이므로,\n"
        "> **안전 논의는 이 1.6s 값과 아래 안전 recall/precision을 함께 본다.**\n\n"
        f"### val\n\n{_acc16_table(acc['val'])}\n\n"
        f"### test (최종 1회)\n\n{_acc16_table(acc['test'])}\n\n"
        "## 읽는 법\n\n"
        "- 값은 `point [lo, hi]` = scene 단위 bootstrap 95% CI. 겹치면 차이가 유의하지 않을 수 있다.\n"
        "- **움직임** 열 = 예측 구간이 실제로 움직인 윈도우만(정지·지터 제외, 진짜 난이도).\n"
        "- 스테이션(goal)=현재 목표를 아는 상한 · minADE@K=K모드 중 최선(멀티모달 상한, 오라클).\n"
        "- **LSTM vs Transformer**: 백본만 다르고 나머지 동일 → 차이는 백본에서 온다.\n"
        "- **안전 기준**: 단일 'ADE<X' 임계는 없다. 로봇 속도·제동·ISO 여유마진에 달림. 실무 기준은\n"
        "  1.6s FDE가 정지반경(공칭 1.2m)·마진보다 충분히 작을 것(대략 ≲0.3~0.5m) + 진입 recall↑.\n"
        "- test 는 val 선택을 고정한 뒤 1회만 평가(운영점 과적합 방지 — 감사 P0-1).\n",
        encoding="utf-8")

    (ROOT / "docs" / "chanwoo" / "prediction-safety-eval.md").write_text(
        "# 궤적 예측 안전 지표 — 정지반경 진입 recall/precision (P0-1 · 백본 비교)\n\n" + head +
        f"> 라이브 제어 지평선 1.6s(4스텝) · 정지반경 R={SAFE_R}m. 선제 안전층이라 **recall 우선**.\n\n"
        f"선택(val 기준): 안전 운영점 = **{selection['chosen_safety_operating_point']}**\n\n"
        f"## val (선택 근거) — 대상 {safe['val']['n_anti']} · 실제 진입 {safe['val']['n_pos']}\n\n{_safe_table(safe['val'])}\n\n"
        f"## test (최종 1회) — 대상 {safe['test']['n_anti']} · 실제 진입 {safe['test']['n_pos']}\n\n{_safe_table(safe['test'])}\n\n"
        "## 읽는 법\n\n"
        "- \"지금 반경 **밖**의 사람이 1.6s 안에 반경 안으로 진입할지\"를 미리 맞혔나. 반경 안은 반응형 몫이라 제외.\n"
        "- **recall**=실제 진입 중 미리 잡은 비율(놓치면 충돌) · **precision**=경보 중 진짜 비율(낮으면 헛정지).\n"
        "- **최빈**=최상위 단일 모드 · **전모드**=K모드 합집합(보수적, recall↑). 백본별로 둘 다 낸다.\n"
        "- CI = scene 단위 bootstrap 95%. test 는 val 선택 고정 후 1회.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
