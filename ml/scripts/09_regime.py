import numpy as np

from odom_ml.data.build import HZ, load_labeled, manifest
from odom_ml.models.traction import TractionModel
from odom_ml import config as C

model = TractionModel.from_npz(C.ARTIFACTS_DIR / "models" / "traction_v1" / "traction.npz")
man = manifest(HZ)
bags = [r["bag_id"] for r in man["bags"] if r["duration"] >= 300.0][:40]
print("bags", len(bags))

allv = {k: [] for k in ("ref", "whl", "u", "acc", "whl_f", "whl_r", "model_only", "trust")}
gap_hist = []
for bid in bags:
    d = load_labeled(bid, HZ)
    t, u = d["t"], d["u"]
    ref, whl = d["speed"], d["v_wheel_mean"]
    a_ref = d["accel"]
    # model-only propagation (open loop with the measured u)
    v = float(np.nanmedian(ref[:20]))
    mo = np.empty_like(ref)
    for i in range(ref.size):
        if i:
            v = max(0.0, v + float(model.accel(u[i - 1], max(v, 0.0))) * 0.02)
        mo[i] = v
    allv["ref"].append(ref)
    allv["whl"].append(whl)
    allv["whl_f"].append(d["v_front"])
    allv["whl_r"].append(d["v_rear"])
    allv["u"].append(u)
    allv["acc"].append(a_ref)
    allv["model_only"].append(mo)
    # gap stats on the wheel mean
    m = ~np.isfinite(whl)
    if m.any():
        idx = np.flatnonzero(np.diff(np.concatenate([[0], m.view(np.int8), [0]])))
        for s, e in zip(idx[::2], idx[1::2]):
            gap_hist.append((e - s) * 0.02)

ref = np.concatenate(allv["ref"])
whl = np.concatenate(allv["whl"])
u = np.concatenate(allv["u"])
acc = np.concatenate(allv["acc"])
mo = np.concatenate(allv["model_only"])
wf = np.concatenate(allv["whl_f"])
wr = np.concatenate(allv["whl_r"])
n = ref.size
print("samples", n, "ref finite", np.isfinite(ref).mean())

ok = np.isfinite(ref) & (ref > 0.3)
okw = ok & np.isfinite(whl)
print("\n--- overall (v>0.3) ---")
print("wheel_mean rmse %.4f bias %+.4f" % (np.sqrt(np.mean((whl[okw] - ref[okw]) ** 2)), np.mean(whl[okw] - ref[okw])))
print("model_only rmse %.4f bias %+.4f" % (np.sqrt(np.mean((mo[ok] - ref[ok]) ** 2)), np.mean(mo[ok] - ref[ok])))
blend = np.where(np.isfinite(whl), whl, mo)
print("wheel+fill rmse %.4f" % np.sqrt(np.mean((blend[ok] - ref[ok]) ** 2)))

print("\n--- by accel regime (v>0.3) ---")
for name, m in [
    ("accel >0.3", ok & (acc > 0.3)),
    ("accel 0.1..0.3", ok & (acc > 0.1) & (acc <= 0.3)),
    ("coast |a|<0.1", ok & (np.abs(acc) <= 0.1)),
    ("brake -0.3..-0.1", ok & (acc < -0.1) & (acc >= -0.3)),
    ("brake <-0.3", ok & (acc < -0.3)),
]:
    if m.sum() < 100:
        continue
    w = whl[m] - ref[m]
    o = mo[m] - ref[m]
    print(
        "%-18s n=%8d wheel_rmse=%.4f bias=%+.4f  model_rmse=%.4f bias=%+.4f"
        % (name, m.sum(), np.sqrt(np.mean(w**2)), w.mean(), np.sqrt(np.mean(o**2)), o.mean())
    )

print("\n--- by |u| regime (v>0.3) ---")
for name, m in [
    ("u=0 coast", ok & (np.abs(u) < 0.5)),
    ("|u|<=3", ok & (np.abs(u) >= 0.5) & (np.abs(u) <= 3)),
    ("|u| 4..8", ok & (np.abs(u) > 3) & (np.abs(u) <= 8)),
    ("|u| 9..12", ok & (np.abs(u) > 8) & (np.abs(u) <= 12)),
    ("|u|>12", ok & (np.abs(u) > 12)),
]:
    if m.sum() < 100:
        continue
    w = whl[m] - ref[m]
    print("%-12s n=%8d wheel_rmse=%.4f bias=%+.4f  (mean v=%.2f)" % (name, m.sum(), np.sqrt(np.mean(w**2)), w.mean(), ref[m].mean()))

print("\n--- by speed (v>0.3) ---")
for lo, hi in [(0.3, 2), (2, 5), (5, 8), (8, 12), (12, 16)]:
    m = okw & (ref >= lo) & (ref < hi)
    if m.sum() < 100:
        continue
    w = whl[m] - ref[m]
    print("v %4.1f-%4.1f n=%8d wheel_rmse=%.4f bias=%+.4f  ratio=%.5f" % (lo, hi, m.sum(), np.sqrt(np.mean(w**2)), w.mean(), np.mean(whl[m]) / np.mean(ref[m])))

print("\n--- wheel availability ---")
print("finite whl frac %.4f  finite front %.4f finite rear %.4f" % (np.isfinite(whl).mean(), np.isfinite(wf).mean(), np.isfinite(wr).mean()))
if gap_hist:
    g = np.array(gap_hist)
    print("gaps: n=%d max=%.2fs p99=%.2fs total=%.0fs frac_time_in_gap=%.4f" % (g.size, g.max(), np.percentile(g, 99), g.sum(), g.sum() / (n * 0.02)))
    long_g = g[g > 0.15]
    print("gaps>0.15s: n=%d total=%.0fs max=%.2f" % (long_g.size, long_g.sum(), long_g.max() if long_g.size else 0))

print("\n--- front vs rear (where both) ---")
b = np.isfinite(wf) & np.isfinite(wr)
dd = (wf - wr)[b]
print("diff mean %+.4f std %.4f p99 %.4f" % (dd.mean(), dd.std(), np.percentile(np.abs(dd), 99)))
