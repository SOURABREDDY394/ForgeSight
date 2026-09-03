"""
ForgeSight — Synthetic Industrial Sample Generator
Generates realistic manufacturing test components:
- Machined Bearing Raceways (Normal vs Radial Gouge)
- Stamped Metal Brackets (Normal vs Stress Crack)
- Cast Aluminum Flanges (Normal vs Casting Porosity Void)
- Electronic Subassemblies / PCBs (Normal vs Bridge / Missing Element)
- Threaded Studs (Normal vs Thread Burr)
"""

import os
import cv2
import numpy as np


def create_sample_images(output_dir: str = "static/samples"):
    os.makedirs(output_dir, exist_ok=True)
    size = (384, 384)

    # 1. Bearing Race — Golden Normal
    img_bearing_normal = np.ones((*size, 3), dtype=np.uint8) * 148
    center = (192, 192)
    # Draw concentric lathe rings
    for r in range(30, 175, 4):
        val = int(148 + np.sin(r * 0.4) * 18 + np.random.randint(-4, 4))
        cv2.circle(img_bearing_normal, center, r, (val, val, val), 2)
    # Metallic specular sheen
    x_coords, y_coords = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    sheen = np.sin((x_coords - y_coords) / 35.0) * 12
    img_bearing_normal = np.clip(img_bearing_normal + sheen[:, :, None], 0, 255).astype(np.uint8)
    # Inner and outer bore dark borders
    cv2.circle(img_bearing_normal, center, 45, (35, 38, 42), -1)
    cv2.circle(img_bearing_normal, center, 175, (40, 42, 48), 6)
    cv2.imwrite(os.path.join(output_dir, "bearing_normal.png"), img_bearing_normal)

    # 2. Bearing Race — Scratch Defect
    img_bearing_scratch = img_bearing_normal.copy()
    # Draw harsh sharp scratch across the raceway
    scratch_pts = np.array([[120, 110], [145, 140], [175, 185], [195, 220]], np.int32)
    cv2.polylines(img_bearing_scratch, [scratch_pts], isClosed=False, color=(30, 30, 30), thickness=3)
    cv2.polylines(img_bearing_scratch, [scratch_pts + 1], isClosed=False, color=(240, 240, 240), thickness=1)  # highlight ridge
    cv2.imwrite(os.path.join(output_dir, "bearing_scratch.png"), img_bearing_scratch)

    # 3. Stamped Steel Bracket — Golden Normal
    img_bracket_normal = np.ones((*size, 3), dtype=np.uint8) * 170
    # Brushed metal texture
    brush_noise = np.random.normal(0, 7, size).astype(np.int16)
    brush_horiz = cv2.GaussianBlur(brush_noise, (31, 1), 0)
    img_bracket_normal = np.clip(img_bracket_normal.astype(np.int16) + brush_horiz[:, :, None], 0, 255).astype(np.uint8)
    # Stamped mounting hole
    cv2.circle(img_bracket_normal, (192, 192), 52, (50, 52, 58), -1)
    cv2.circle(img_bracket_normal, (192, 192), 58, (220, 220, 225), 2)  # chamfer
    cv2.imwrite(os.path.join(output_dir, "bracket_normal.png"), img_bracket_normal)

    # 4. Stamped Steel Bracket — Crack Defect
    img_bracket_crack = img_bracket_normal.copy()
    # Stress fracture branching from hole rim
    crack_pts = [
        (192 - 38, 192 - 38), (135, 130), (122, 118), (110, 95), (92, 78)
    ]
    for i in range(len(crack_pts) - 1):
        cv2.line(img_bracket_crack, crack_pts[i], crack_pts[i+1], (20, 20, 20), 2)
        # Micro fissure branch
        if i == 2:
            cv2.line(img_bracket_crack, crack_pts[i], (140, 102), (25, 25, 25), 1)
    cv2.imwrite(os.path.join(output_dir, "bracket_crack.png"), img_bracket_crack)

    # 5. Cast Aluminum Flange — Void / Porosity Defect
    img_flange_void = np.ones((*size, 3), dtype=np.uint8) * 155
    # Cast granular pebble texture
    grain = np.random.randint(-12, 12, size).astype(np.int16)
    img_flange_void = np.clip(img_flange_void.astype(np.int16) + grain[:, :, None], 0, 255).astype(np.uint8)
    # Porosity blowholes cluster
    void_centers = [(240, 140), (248, 145), (236, 152), (255, 138), (242, 160)]
    for vc in void_centers:
        cv2.circle(img_flange_void, vc, np.random.randint(4, 9), (20, 20, 22), -1)
        cv2.circle(img_flange_void, (vc[0]-1, vc[1]-1), np.random.randint(2, 4), (5, 5, 5), -1)
    cv2.imwrite(os.path.join(output_dir, "flange_void.png"), img_flange_void)

    # 6. PCB Circuit Assembly — Solder Bridge Defect
    img_pcb_defect = np.ones((*size, 3), dtype=np.uint8)
    img_pcb_defect[:, :] = (34, 110, 48)  # Green solder mask
    # Copper traces
    for y in range(60, 340, 36):
        cv2.line(img_pcb_defect, (50, y), (330, y), (20, 185, 220), 5)  # Gold/copper traces
        cv2.circle(img_pcb_defect, (100, y), 10, (210, 215, 220), -1)   # Silver pads
        cv2.circle(img_pcb_defect, (280, y), 10, (210, 215, 220), -1)
    # Solder bridge short circuit between trace 2 and 3
    cv2.ellipse(img_pcb_defect, (190, 114), (8, 20), 15, 0, 360, (210, 215, 225), -1)
    cv2.imwrite(os.path.join(output_dir, "pcb_bridge.png"), img_pcb_defect)

    print(f"[ForgeSight] Created 6 industrial sample components in {output_dir}")


if __name__ == "__main__":
    create_sample_images()
