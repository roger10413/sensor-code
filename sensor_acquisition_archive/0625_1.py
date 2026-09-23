"""
Process senior's c2.csv:
1. Raw time-frequency plot
2. Median-filtered time-frequency plot (removes cycle-slip glitches)
3. Sensitivity in g/Hz (mass inferred from frequency), using filtered data
Outputs timestamped folder with CSVs + separate figures.
"""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.stats import linregress
import os, datetime

CSV = "c2.csv"
MEDWIN = 501            # median window; cycle-slips are outliers -> median removes them
DYNO_TO_N = 100.0       # senior's Dyno column = N/100
N_PER_GF = 0.00980665   # 1 gf = 0.00980665 N

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = f"senior_out_{ts}"; os.makedirs(outdir, exist_ok=True)

df = pd.read_csv(CSV)
t = df["Time(s)"].values
f = df["Frequency(Hz)"].values
dyno = df["Dyno"].values
force_N = dyno * DYNO_TO_N
mass_g = force_N / N_PER_GF

f_filt = medfilt(f, kernel_size=MEDWIN)

# filter quality on a held-load window (force ~constant) so std reflects noise, not load change
held = (t >= 88) & (t < 90)
raw_held = f[held].std(); filt_held = f_filt[held].std()
print(f"Held-window raw std  = {raw_held:.1f} Hz")
print(f"Held-window filt std = {filt_held:.1f} Hz")

# Fig 1: raw
plt.figure(figsize=(11,5))
plt.plot(t, f, lw=0.4, color="#c0392b")
plt.xlabel("Time (s)"); plt.ylabel("Frequency (Hz)")
plt.title("Senior raw data: Time vs Frequency (with cycle-slip glitches)")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{outdir}/fig1_raw_time_freq.png", dpi=150); plt.close()

# Fig 2: filtered
plt.figure(figsize=(11,5))
plt.plot(t, f_filt, lw=0.7, color="#27ae60")
plt.xlabel("Time (s)"); plt.ylabel("Frequency (Hz)")
plt.title(f"Median-filtered (window={MEDWIN}): Time vs Frequency "
          f"(held-window std {raw_held:.0f}->{filt_held:.0f} Hz)")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{outdir}/fig2_filtered_time_freq.png", dpi=150); plt.close()

# Sensitivity: bin by mass (median) to get clean characteristic curve, avoids hysteresis overlap
mask = np.isfinite(f_filt) & np.isfinite(mass_g)
mg = mass_g[mask]; ff = f_filt[mask]
nbins = 60
bins = np.linspace(mg.min(), mg.max(), nbins+1)
idx = np.digitize(mg, bins)
bx, by = [], []
for b in range(1, nbins+1):
    mm = idx == b
    if mm.sum() > 50:
        bx.append(np.median(mg[mm])); by.append(np.median(ff[mm]))
bx = np.array(bx); by = np.array(by)

reg = linregress(bx, by)
S_HzPerG = reg.slope
f0 = reg.intercept
S_gPerHz = 1.0 / S_HzPerG          # <-- requested direction: g/Hz

# endpoint method (actual data ends, no extrapolation)
S_HzPerG_end = (by[-1]-by[0])/(bx[-1]-bx[0])
S_gPerHz_end = 1.0/S_HzPerG_end

print("\n=== Sensitivity (filtered) ===")
print(f"Fit     : {S_HzPerG:.4f} Hz/g -> {S_gPerHz:.4f} g/Hz  R^2={reg.rvalue**2:.5f}")
print(f"Endpoint: {S_HzPerG_end:.4f} Hz/g -> {S_gPerHz_end:.4f} g/Hz")
print(f"f0 = {f0:.0f} Hz")

# Fig 3: characteristic curve
plt.figure(figsize=(8,6))
plt.scatter(bx, by, s=18, color="#2980b9", label="Binned median (filtered)")
xx = np.linspace(bx.min(), bx.max(), 200)
plt.plot(xx, reg.slope*xx+reg.intercept, "r-",
         label=f"Fit: {S_gPerHz:.3f} g/Hz (R2={reg.rvalue**2:.4f})")
plt.xlabel("Equivalent mass (g)"); plt.ylabel("Frequency (Hz)")
plt.title("Senior characteristic curve (filtered) & sensitivity")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{outdir}/fig3_characteristic_curve.png", dpi=150); plt.close()

# CSV outputs
pd.DataFrame({
    "Time(s)": t, "Frequency_raw(Hz)": f, "Frequency_filtered(Hz)": f_filt,
    "Dyno": dyno, "Force(N)": force_N, "Mass_equiv(g)": mass_g,
}).to_csv(f"{outdir}/cleaned_data.csv", index=False)
pd.DataFrame({"Mass_g": bx, "Freq_filt_Hz": by}).to_csv(
    f"{outdir}/characteristic_curve.csv", index=False)
pd.DataFrame({
    "item": ["held_raw_std_Hz","held_filt_std_Hz","S_Hz_per_g_fit","S_g_per_Hz_fit",
             "R2_fit","S_Hz_per_g_endpoint","S_g_per_Hz_endpoint","f0_Hz"],
    "value": [raw_held, filt_held, S_HzPerG, S_gPerHz, reg.rvalue**2,
              S_HzPerG_end, S_gPerHz_end, f0],
}).to_csv(f"{outdir}/sensitivity_summary.csv", index=False)

print("\nOutput folder:", outdir)
print(sorted(os.listdir(outdir)))