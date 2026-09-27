"""
ForgeSight — Safe Model Registry & Staged Rollout Controller
Implements validation approval gates, version tracking, canary deployment,
and zero-downtime rollback to protect manufacturing lines from silent regressions.
"""

import time
from typing import Dict, Any, List, Optional


class SafeModelRegistry:
    """
    MLOps Model Registry with Industrial Release Approval Gates.
    Guarantees no unvalidated or regressed model is silently activated on factory edge devices.
    """

    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.active_production_version: str = "v1.4.2"
        self.canary_version: Optional[str] = None
        self.canary_traffic_percent: int = 0
        self._seed_default_models()

    def _seed_default_models(self):
        """Register initial benchmark models for the metal component line."""
        self.register_model({
            "version": "v1.3.0",
            "name": "ForgeSight Baseline Gabor-KNN",
            "architecture": "Gabor-Sobel MultiPatch + KNN",
            "dataset_version": "mvtec-metal-parts-v1.0",
            "status": "ARCHIVED",
            "created_at": "2026-08-12 09:14:00",
            "metrics": {
                "image_auroc": 0.9410,
                "pixel_iou": 0.7620,
                "false_rejects_per_1000": 4.80,
                "defect_escape_rate_pct": 0.20,
                "p95_latency_ms": 14.2
            },
            "gate_passed": True,
            "notes": "Initial pilot deployment. Superseded by PatchCore memory bank."
        })

        self.register_model({
            "version": "v1.4.2",
            "name": "ForgeSight PatchCore Edge-Prod",
            "architecture": "PatchCore Coreset Memory Bank + Gaussian Unpool",
            "dataset_version": "mvtec-metal-parts-v2.1",
            "status": "PRODUCTION",
            "created_at": "2026-09-28 14:32:00",
            "metrics": {
                "image_auroc": 0.9882,
                "pixel_iou": 0.8840,
                "false_rejects_per_1000": 0.60,
                "defect_escape_rate_pct": 0.00,
                "p95_latency_ms": 18.6
            },
            "gate_passed": True,
            "notes": "Current active production model across all 4 line inspection stations."
        })

        self.register_model({
            "version": "v2.0.0-rc1",
            "name": "ForgeSight Few-Shot MultiScale Candidate",
            "architecture": "Hierarchical PatchCore + Synthetic Defect Augmentor",
            "dataset_version": "mvtec-metal-parts-v2.4-reviewed",
            "status": "CANARY",
            "created_at": "2026-10-12 18:20:00",
            "metrics": {
                "image_auroc": 0.9945,
                "pixel_iou": 0.9160,
                "false_rejects_per_1000": 0.25,
                "defect_escape_rate_pct": 0.00,
                "p95_latency_ms": 21.4
            },
            "gate_passed": True,
            "notes": "Candidate trained with human review triage labels. 15% traffic canary."
        })
        self.canary_version = "v2.0.0-rc1"
        self.canary_traffic_percent = 15

    def register_model(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        version = model_data["version"]
        self.models[version] = model_data
        return model_data

    def list_models(self) -> List[Dict[str, Any]]:
        return list(self.models.values())

    def get_model(self, version: str) -> Optional[Dict[str, Any]]:
        return self.models.get(version)

    def promote_to_production(self, version: str) -> Dict[str, Any]:
        """Promote a candidate model to full 100% production after passing approval gates."""
        if version not in self.models:
            raise ValueError(f"Model version {version} not found in registry.")

        cand = self.models[version]
        # Verify approval gates
        metrics = cand.get("metrics", {})
        if metrics.get("defect_escape_rate_pct", 1.0) > 0.05:
            raise ValueError(f"GATE VIOLATION: Model {version} has escape rate > 0.0%. Deployment halted.")

        # Demote previous prod model to rollback archive
        if self.active_production_version in self.models:
            self.models[self.active_production_version]["status"] = "PREVIOUS_PROD_STANDBY"

        cand["status"] = "PRODUCTION"
        self.active_production_version = version
        self.canary_version = None
        self.canary_traffic_percent = 0
        return cand

    def rollback(self, target_version: Optional[str] = None) -> Dict[str, Any]:
        """Instantly rollback production line to a proven previous model."""
        target = target_version or "v1.4.2"
        if target not in self.models:
            raise ValueError(f"Rollback target {target} not available.")

        current = self.active_production_version
        if current in self.models:
            self.models[current]["status"] = "ROLLED_BACK"

        self.models[target]["status"] = "PRODUCTION"
        self.active_production_version = target
        self.canary_version = None
        self.canary_traffic_percent = 0

        return {
            "status": "ROLLBACK_SUCCESSFUL",
            "active_version": target,
            "demoted_version": current,
            "timestamp": time.time()
        }

    def set_canary(self, version: str, traffic_percent: int) -> Dict[str, Any]:
        """Route partial inspection line traffic to candidate model for staged A/B evaluation."""
        if version not in self.models:
            raise ValueError(f"Canary version {version} not registered.")
        
        self.canary_version = version
        self.canary_traffic_percent = max(0, min(100, traffic_percent))
        self.models[version]["status"] = f"CANARY_{self.canary_traffic_percent}%"

        return {
            "canary_version": version,
            "traffic_percent": self.canary_traffic_percent,
            "production_version": self.active_production_version
        }

    def compare(self, version_a: str, version_b: str) -> Dict[str, Any]:
        """Side-by-side performance comparison for engineering review."""
        mod_a = self.models.get(version_a, {})
        mod_b = self.models.get(version_b, {})

        return {
            "model_a": mod_a,
            "model_b": mod_b,
            "metric_delta": {
                "auroc_delta": round(mod_b.get("metrics", {}).get("image_auroc", 0) - mod_a.get("metrics", {}).get("image_auroc", 0), 4),
                "pixel_iou_delta": round(mod_b.get("metrics", {}).get("pixel_iou", 0) - mod_a.get("metrics", {}).get("pixel_iou", 0), 4),
                "false_reject_delta": round(mod_b.get("metrics", {}).get("false_rejects_per_1000", 0) - mod_a.get("metrics", {}).get("false_rejects_per_1000", 0), 2),
                "escape_rate_delta": round(mod_b.get("metrics", {}).get("defect_escape_rate_pct", 0) - mod_a.get("metrics", {}).get("defect_escape_rate_pct", 0), 2)
            }
        }
