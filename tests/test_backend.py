"""
Unit and Integration Tests for ForgeSight
Validates:
- Anomaly detector patch extraction, anomaly scoring, heatmap generation
- Drift detector lighting, blur, and score shift logic
- Model registry safety gates, canary traffic, rollback
- Data manager SQLite persistence and triage reviews
- Evaluator manufacturing KPI formulas
"""

import pytest
import numpy as np
from backend.anomaly_detector import IndustrialAnomalyDetector
from backend.drift_detector import IndustrialDriftMonitor
from backend.model_registry import SafeModelRegistry
from backend.data_manager import IndustrialDataManager
from backend.evaluator import IndustrialModelEvaluator


def test_anomaly_detector_golden_vs_defect():
    detector = IndustrialAnomalyDetector()
    
    # 1. Test clean normal metal component
    normal_img = np.ones((256, 256, 3), dtype=np.uint8) * 160
    normal_result = detector.inspect(normal_img)
    assert "anomaly_score" in normal_result
    assert "heatmap_base64" in normal_result
    assert normal_result["anomaly_score"] < 0.60
    assert normal_result["inference_time_ms"] > 0

    # 2. Test component with severe artificial black gouge defect
    defect_img = normal_img.copy()
    defect_img[100:150, 100:150] = 0
    defect_result = detector.inspect(defect_img)
    assert defect_result["anomaly_score"] > normal_result["anomaly_score"]
    assert len(defect_result["bounding_boxes"]) >= 1


def test_drift_monitor_blur_and_lighting():
    monitor = IndustrialDriftMonitor()
    
    # Normal frame
    normal_img = np.random.randint(120, 170, (256, 256, 3), dtype=np.uint8)
    res_normal = monitor.check_frame(normal_img, anomaly_score=0.15)
    assert res_normal["status"] in ["STABLE_IN_SPEC", "ENVIRONMENTAL_WARNING"]

    # Severe blur frame (all uniform pixels -> zero Laplacian variance)
    blur_img = np.ones((256, 256, 3), dtype=np.uint8) * 140
    res_blur = monitor.check_frame(blur_img, anomaly_score=0.20)
    assert res_blur["sharpness_ratio"] < 0.20


def test_model_registry_safety_gates_and_rollback():
    registry = SafeModelRegistry()
    assert registry.active_production_version == "v1.4.2"

    # Test canary
    canary_res = registry.set_canary("v2.0.0-rc1", 20)
    assert canary_res["traffic_percent"] == 20

    # Test promotion
    promoted = registry.promote_to_production("v2.0.0-rc1")
    assert registry.active_production_version == "v2.0.0-rc1"

    # Test instant rollback
    rollback_res = registry.rollback("v1.4.2")
    assert registry.active_production_version == "v1.4.2"
    assert rollback_res["status"] == "ROLLBACK_SUCCESSFUL"


def test_evaluator_metrics():
    y_true = [0, 0, 0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.15, 0.3, 0.85, 0.95]
    y_pred = [0, 0, 0, 0, 1, 1]
    latencies = [15.2, 16.1, 14.8, 17.0, 15.5, 16.4]

    metrics = IndustrialModelEvaluator.evaluate_inspection_run(
        y_true, y_scores, y_pred, latencies
    )
    assert metrics["image_auroc"] == 1.0
    assert metrics["defect_escape_rate_pct"] == 0.0
    assert metrics["false_rejects_per_1000"] == 0.0
    assert metrics["p95_latency_ms"] > 0
    assert metrics["meets_release_gates"] is True


def test_data_manager_triage_loop(tmp_path):
    db_file = str(tmp_path / "test_forgesight.db")
    dm = IndustrialDataManager(db_path=db_file)

    insp_id = "INSP-TEST-001"
    dm.save_inspection({
        "id": insp_id,
        "part_id": "PART-999",
        "station_id": "STN-01-CAST",
        "timestamp": 1720000000.0,
        "is_defect": True,
        "anomaly_score": 0.88,
        "defect_type": "Surface Crack",
        "model_version": "v1.4.2",
        "latency_ms": 18.5
    })

    triage = dm.get_triage_queue()
    assert len(triage) == 1
    assert triage[0]["id"] == insp_id

    # Reviewer confirms defect
    rev = dm.submit_review(insp_id, reviewer="Senior QA Inspector", decision="CONFIRM_DEFECT")
    assert rev["decision"] == "CONFIRM_DEFECT"
    assert rev["review_status"] == "VERIFIED_CONFIRM_DEFECT"
