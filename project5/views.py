import json, random, torch, numpy as np
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from django.http import JsonResponse, FileResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import render

from .mouse import MouseGridEnv
from .rlcore import (
    ARTIFACTS, rollout, rollout_ui, load_policy,
    make_pref_pairs_no_org, fit_bradley_terry, RewardNet, train_rlhf, has_movement
)

# in-memory store for pending pairs; saved prefs go to JSONL file
PREF_STORE = {}
PREF_FILE  = ARTIFACTS / "prefs.jsonl"

def index(request):
    return render(request, "project5/index.html")

# ---- Task 1: baseline training ----
@csrf_exempt
@require_http_methods(["POST"])
def run_demo(request):
    from .rlcore import train_reinforce
    _policy, _rets = train_reinforce(episodes=60, seed=0)
    return JsonResponse({"ok": True, "msg": "Baseline trained", "artifact": "policy_reinforce.pt"})

# ---- Task 2: train reward model (fake or real) ----
@csrf_exempt
@require_http_methods(["POST"])
def train_reward(request):
    use_real = False
    try:
        body = json.loads(request.body.decode() or "{}")
        use_real = bool(body.get("use_real", False))
    except Exception:
        pass

    if use_real:
        pairs = []
        if PREF_FILE.exists():
            with open(PREF_FILE, "r") as f:
                for line in f:
                    d = json.loads(line)
                    pairs.append((
                        np.array(d["o1"], dtype=np.float32),
                        np.array(d["o2"], dtype=np.float32),
                        int(d["y"])
                    ))
        if not pairs:
            return JsonResponse({"ok": False, "error": "No saved preferences yet. Collect some first."}, status=400)
        _ = fit_bradley_terry(pairs)
        return JsonResponse({"ok": True, "msg": f"Reward model trained from {len(pairs)} human pairs", "artifact": "reward_model.pt"})

    # fake user: prefer fewer organics
    env = MouseGridEnv()
    policy = load_policy("base")
    trajs = [rollout(env, policy) for _ in range(40)]
    pairs = make_pref_pairs_no_org(trajs)
    _ = fit_bradley_terry(pairs)
    return JsonResponse({"ok": True, "msg": f"Reward model trained on {len(pairs)} pairs (fake user)", "artifact": "reward_model.pt"})

# ---- Task 3: RLHF training ----
@csrf_exempt
@require_http_methods(["POST"])
def run_rlhf(request):
    reward_path = ARTIFACTS / "reward_model.pt"
    if not reward_path.exists():
        return JsonResponse({"ok": False, "error": "Reward model not trained. Train it first."}, status=400)
    policy_ref = load_policy("base")
    reward_model = RewardNet()
    reward_model.load_state_dict(torch.load(reward_path, map_location="cpu"))
    reward_model.eval()
    _ = train_rlhf(policy_ref, reward_model, steps=80)
    return JsonResponse({"ok": True, "msg": "RLHF training complete", "artifact": "policy_rlhf.pt"})

# ---- play endpoints (full episodes; stop on cheese/trap) ----
def rollout_view(request, kind: str):
    if kind not in {"base", "rlhf"}:
        return JsonResponse({"ok": False, "error": "unknown kind"}, status=400)
    env = MouseGridEnv()
    policy = load_policy(kind)
    traj = rollout_ui(env, policy)   # UI-friendly but still stops on done
    frames = traj["grids"].tolist()
    return JsonResponse({
        "ok": True,
        "kind": kind,
        "frames": frames,
        "num_frames": len(frames),
        "reward_sum": float(traj["rews"].sum()),
    })

# ---- artifacts + status ----
def artifact(request, name: str):
    path = ARTIFACTS / name
    if not path.exists(): raise Http404("not found")
    return FileResponse(open(path, "rb"), as_attachment=True, filename=name)

def status(request):
    out = []
    for name in ["policy_reinforce.pt", "reward_model.pt", "policy_rlhf.pt"]:
        p = ARTIFACTS / name
        if p.exists():
            stat = p.stat()
            out.append({
                "name": name, "exists": True, "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "download": f"./artifact/{name}",
            })
        else:
            out.append({"name": name, "exists": False, "size": 0, "mtime": None, "download": None})
    return JsonResponse({"ok": True, "artifacts": out})

# ---- human preference collection (optional but cool) ----
@require_http_methods(["GET"])
def pref_new(request):
    """Serve two complete episodes (A/B) that visibly move."""
    env = MouseGridEnv()
    policy = load_policy("base")

    # try several times so both sides have some movement
    for _ in range(12):
        t1 = rollout_ui(env, policy)
        t2 = rollout_ui(env, policy)
        A = t1["grids"].tolist(); B = t2["grids"].tolist()
        if has_movement(A) and has_movement(B):
            pid = str(uuid4()); PREF_STORE[pid] = {"o1": t1["obs"], "o2": t2["obs"]}
            return JsonResponse({"ok": True, "pair_id": pid, "A": A, "B": B})

    # fallback — still return something
    pid = str(uuid4()); PREF_STORE[pid] = {"o1": t1["obs"], "o2": t2["obs"]}
    return JsonResponse({"ok": True, "pair_id": pid, "A": A, "B": B})

@csrf_exempt
@require_http_methods(["POST"])
def pref_choose(request):
    try:
        body = json.loads(request.body.decode() or "{}")
        pid = body.get("pair_id"); choice = body.get("choice")  # "A" or "B"
        if pid not in PREF_STORE or choice not in ("A", "B"):
            return JsonResponse({"ok": False, "error": "bad id or choice"}, status=400)
        y = 1 if choice == "A" else 0
        with open(PREF_FILE, "a") as f:
            json.dump({"o1": PREF_STORE[pid]["o1"].tolist(),
                       "o2": PREF_STORE[pid]["o2"].tolist(),
                       "y": y}, f); f.write("\n")
        del PREF_STORE[pid]
        cnt = 0
        if PREF_FILE.exists():
            with open(PREF_FILE) as f: cnt = sum(1 for _ in f)
        return JsonResponse({"ok": True, "saved": True, "count": cnt})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)
