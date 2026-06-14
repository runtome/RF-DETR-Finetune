import glob
import os
import re


def get_next_trial_name(base_dir: str = "outputs", prefix: str = "trial") -> str:
    os.makedirs(base_dir, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    existing = []
    try:
        for d in os.listdir(base_dir):
            m = pattern.match(d)
            if m and os.path.isdir(os.path.join(base_dir, d)):
                existing.append(int(m.group(1)))
    except FileNotFoundError:
        pass
    n = (max(existing) + 1) if existing else 1
    return f"{prefix}_{n:02d}"


def find_latest_checkpoint(base_dir: str = "outputs") -> str | None:
    pattern = os.path.join(base_dir, "trial_*", "save_model", "*.pth")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)
