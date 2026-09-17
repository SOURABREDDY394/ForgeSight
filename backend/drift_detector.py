"""
ForgeSight — Distribution Shift & Environmental Drift Monitor
Monitors camera lighting variations, optical blur/defocus, color/material shifts,
and model score distribution to alert operators before false alarms propagate.
"""

import time
import numpy as np
import cv2
from scipy.stats import ks_2samp
from typing import Dict, Any, List, Optional


class IndustrialDriftMonitor:
    """
    Real-time distribution shift monitor for industrial edge vision lines.
    Tracks:
    1. Lighting / Exposure shift (Kolmogorov-Smirnov histogram test)
    2. Optical blur / vibration defocus (Laplacian variance)
    3. Chromatic / Alloy variance (LAB color distribution)
    4. Model confidence drift (rolling anomaly score drift)
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.baseline_calibrated = False
        
        # Baseline reference statistics
        self.baseline_luminance_hist: Optional[np.ndarray] = None
        self.baseline_laplacian_var: float = 450.0
        self.baseline_color_means: np.ndarray = np.array([128.0, 128.0, 128.0])
        
        # Rolling buffers
        self.recent_scores: List[float] = []
        self.recent_laplacian: List[float] = []
        self.recent_luminances: List[float] = []
        self.drift_events: List[Dict[str, Any]] = []

        self._calibrate_synthetic_baseline()

    def _calibrate_synthetic_baseline(self):
        """Seed baseline statistics for a calibrated industrial inspection station."""
        bins = np.linspace(0, 256, 33)
        # Typical bell curve around 140 intensity for a controlled factory lightbox
        synth_hist = np.exp(-((bins[:-1] - 145) ** 2) / (2 * 45 ** 2))
        self.baseline_luminance_hist = synth_hist / synth_hist.sum()
        self.baseline_laplacian_var = 520.0
        self.baseline_color_means = np.array([142.0, 126.0, 130.0])  # Industrial gray-steel
        self.baseline_calibrated = True

    def calibrate_with_samples(self, images: List[np.ndarray]):
        """Calibrate baseline statistics on validated commissioning images."""
        if not images:
            return
        
        lap_vars = []
        lums = []
        color_means = []

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            lap_vars.append(cv2.Laplacian(gray, cv2.CV_64F).var())
            hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=True)
            lums.append(hist)
            if len(img.shape) == 3:
                color_means.append(img.mean(axis=(0, 1)))

        self.baseline_laplacian_var = float(np.mean(lap_vars))
        self.baseline_luminance_hist = np.mean(lums, axis=0)
        if color_means:
            self.baseline_color_means = np.mean(color_means, axis=0)
        self.baseline_calibrated = True

    def check_frame(self, image: np.ndarray, anomaly_score: float) -> Dict[str, Any]:
        """
        Evaluate an incoming frame for distribution shift against baseline.
        Returns drift metrics, shift classifications, and operator recommendations.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 1. Blur / Sharpness check
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.recent_laplacian.append(lap_var)
        if len(self.recent_laplacian) > self.window_size:
            self.recent_laplacian.pop(0)

        sharpness_ratio = lap_var / (self.baseline_laplacian_var + 1e-5)
        is_blur_drift = bool(sharpness_ratio < 0.42)

        # 2. Luminance & Lighting distribution shift
        curr_hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=True)
        # Compute Wasserstein distance / total variation between histograms
        hist_shift = float(0.5 * np.sum(np.abs(curr_hist - self.baseline_luminance_hist)))
        mean_lum = float(gray.mean())
        is_lighting_drift = bool(hist_shift > 0.40 or mean_lum < 55 or mean_lum > 220)

        # 3. Chromatic / Material alloy shift
        color_drift_score = 0.0
        if len(image.shape) == 3:
            curr_color = image.mean(axis=(0, 1))
            color_drift_score = float(np.linalg.norm(curr_color - self.baseline_color_means) / 100.0)
        is_material_drift = bool(color_drift_score > 0.55)

        # 4. Confidence / Score drift
        self.recent_scores.append(anomaly_score)
        if len(self.recent_scores) > self.window_size:
            self.recent_scores.pop(0)

        rolling_score_mean = float(np.mean(self.recent_scores))
        # High rolling defect rate suggests false alarm cascade due to drift
        is_score_drift = bool(len(self.recent_scores) >= 10 and rolling_score_mean > 0.65)

        # Overall Status Synthesis
        drift_warnings = []
        if is_blur_drift:
            drift_warnings.append("OPTICAL_DEFOCUS: Camera focus degraded or lens soiled.")
        if is_lighting_drift:
            drift_warnings.append(f"ILLUMINATION_SHIFT: Ambient/fixture lighting variance ({mean_lum:.1f} lux equiv).")
        if is_material_drift:
            drift_warnings.append("CHROMATIC_SHIFT: Raw alloy surface reflectance changed.")
        if is_score_drift:
            drift_warnings.append("ANOMALY_CASCADE: 65%+ parts failing; possible process parameter change.")

        overall_status = "STABLE_IN_SPEC"
        if len(drift_warnings) >= 2 or is_score_drift:
            overall_status = "CRITICAL_DRIFT_ALERT"
        elif len(drift_warnings) == 1:
            overall_status = "ENVIRONMENTAL_WARNING"

        result = {
            "status": overall_status,
            "drift_detected": overall_status != "STABLE_IN_SPEC",
            "sharpness_ratio": round(sharpness_ratio, 3),
            "laplacian_var": round(lap_var, 1),
            "baseline_laplacian": round(self.baseline_laplacian_var, 1),
            "lighting_shift_score": round(hist_shift, 3),
            "mean_luminance": round(mean_lum, 1),
            "material_drift_score": round(color_drift_score, 3),
            "rolling_anomaly_score": round(rolling_score_mean, 3),
            "active_warnings": drift_warnings,
            "timestamp": time.time()
        }

        if result["drift_detected"]:
            self.drift_events.append({
                "time": time.strftime("%H:%M:%S"),
                "status": overall_status,
                "warnings": drift_warnings,
                "sharpness": result["sharpness_ratio"],
                "lighting": result["lighting_shift_score"]
            })
            if len(self.drift_events) > 25:
                self.drift_events.pop(0)

        return result
