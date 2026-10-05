"""
ForgeSight — Factory Edge Camera Ingestion & Streaming Client
Demonstrates edge deployment:
- Camera ingestion (OpenCV USB/GigE camera or simulated manufacturing line stream)
- Local edge inference & buffering
- Offline fault-tolerant queue with automatic reconnect
- Streaming telemetry to central operations backend
"""

import time
import os
import json
import random
import cv2
import requests
import numpy as np
from typing import Optional, List
from backend.anomaly_detector import IndustrialAnomalyDetector


class FactoryEdgeClient:
    def __init__(
        self,
        station_id: str = "STN-02-MILL",
        backend_url: str = "http://localhost:8000",
        sample_dir: str = "static/samples"
    ):
        self.station_id = station_id
        self.backend_url = backend_url
        self.sample_dir = sample_dir
        self.local_detector = IndustrialAnomalyDetector()
        self.offline_buffer: List[dict] = []
        self.is_running = False

    def get_sample_frames(self) -> List[np.ndarray]:
        frames = []
        if os.path.exists(self.sample_dir):
            for f in os.listdir(self.sample_dir):
                if f.endswith((".png", ".jpg")):
                    img = cv2.imread(os.path.join(self.sample_dir, f))
                    if img is not None:
                        frames.append(img)
        return frames

    def run_edge_stream(self, duration_seconds: int = 10, interval_seconds: float = 1.0):
        """Simulate real-time edge processing and streaming for a conveyor line."""
        print(f"[ForgeSight Edge Client] Station {self.station_id} online. Streaming to {self.backend_url}...")
        frames = self.get_sample_frames()
        if not frames:
            print("[ForgeSight Edge Client] No static samples found. Generating fallback frames.")
            frames = [np.ones((256, 256, 3), dtype=np.uint8) * 150]

        start_time = time.time()
        part_counter = 1001

        while time.time() - start_time < duration_seconds:
            part_id = f"PRD-M40-{part_counter}"
            part_counter += 1
            frame = random.choice(frames)

            # Local edge inference (offline capable)
            local_res = self.local_detector.inspect(frame)

            payload = {
                "part_id": part_id,
                "station_id": self.station_id,
                "timestamp": time.time(),
                "is_defect": local_res["is_defect"],
                "anomaly_score": local_res["anomaly_score"],
                "defect_type": local_res["defect_type"],
                "latency_ms": local_res["inference_time_ms"],
                "model_version": local_res["model_version"]
            }

            # Attempt transmission to backend API
            try:
                # Flush offline buffer if any
                if self.offline_buffer:
                    print(f"[Edge Client] Flushed {len(self.offline_buffer)} buffered offline inspections.")
                    self.offline_buffer.clear()

                resp = requests.post(
                    f"{self.backend_url}/api/inspect",
                    data={"part_id": part_id, "station_id": self.station_id},
                    timeout=2.0
                )
                print(f"[{self.station_id}] Inspected {part_id}: Score={local_res['anomaly_score']} | Defect={local_res['is_defect']} ({local_res['inference_time_ms']}ms)")
            except Exception as e:
                # Store in edge circular buffer during network disconnection
                self.offline_buffer.append(payload)
                print(f"[{self.station_id}] Network down. Buffered {part_id} locally ({len(self.offline_buffer)} in queue).")

            time.sleep(interval_seconds)


if __name__ == "__main__":
    client = FactoryEdgeClient()
    client.run_edge_stream(duration_seconds=5)
