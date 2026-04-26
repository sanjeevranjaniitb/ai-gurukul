"""Generate 5 default avatar placeholder images for the catalog.

These are simple stylized face illustrations created with OpenCV.
Replace them with real HD face images for production use.
"""

import os
import cv2
import numpy as np

OUTPUT_DIR = "frontend/public/avatars"
SIZE = 512


def draw_face(img, skin, hair, eye, lip, bg):
    """Draw a simple stylized face."""
    h, w = img.shape[:2]
    img[:] = bg

    # Head (oval)
    cv2.ellipse(img, (w // 2, h // 2 + 20), (160, 200), 0, 0, 360, skin, -1)

    # Hair
    cv2.ellipse(img, (w // 2, h // 2 - 80), (170, 120), 0, 180, 360, hair, -1)

    # Eyes
    cv2.ellipse(img, (w // 2 - 55, h // 2 - 10), (22, 16), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (w // 2 + 55, h // 2 - 10), (22, 16), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (w // 2 - 55, h // 2 - 10), 10, eye, -1)
    cv2.circle(img, (w // 2 + 55, h // 2 - 10), 10, eye, -1)
    cv2.circle(img, (w // 2 - 53, h // 2 - 12), 4, (255, 255, 255), -1)
    cv2.circle(img, (w // 2 + 57, h // 2 - 12), 4, (255, 255, 255), -1)

    # Eyebrows
    cv2.line(img, (w // 2 - 75, h // 2 - 35), (w // 2 - 35, h // 2 - 40), hair, 3)
    cv2.line(img, (w // 2 + 35, h // 2 - 40), (w // 2 + 75, h // 2 - 35), hair, 3)

    # Nose
    cv2.line(img, (w // 2, h // 2 + 5), (w // 2 - 8, h // 2 + 35), skin, 2)
    cv2.line(img, (w // 2 - 8, h // 2 + 35), (w // 2 + 8, h // 2 + 35), skin, 2)

    # Mouth
    cv2.ellipse(img, (w // 2, h // 2 + 65), (35, 12), 0, 0, 180, lip, -1)
    cv2.ellipse(img, (w // 2, h // 2 + 65), (35, 8), 0, 180, 360, lip, -1)

    # Neck
    cv2.rectangle(img, (w // 2 - 40, h // 2 + 200), (w // 2 + 40, h), skin, -1)

    # Shoulders
    cv2.ellipse(img, (w // 2, h - 20), (200, 80), 0, 180, 360, (80, 80, 80), -1)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    avatars = [
        {
            "name": "professional_male",
            "label": "Professional Male",
            "skin": (180, 200, 220),
            "hair": (40, 30, 20),
            "eye": (80, 60, 40),
            "lip": (120, 130, 180),
            "bg": (220, 200, 180),
        },
        {
            "name": "professional_female",
            "label": "Professional Female",
            "skin": (190, 210, 230),
            "hair": (30, 20, 60),
            "eye": (100, 80, 50),
            "lip": (130, 120, 190),
            "bg": (200, 180, 210),
        },
        {
            "name": "young_male",
            "label": "Young Male",
            "skin": (160, 190, 210),
            "hair": (50, 40, 30),
            "eye": (120, 80, 40),
            "lip": (110, 120, 170),
            "bg": (210, 220, 200),
        },
        {
            "name": "young_female",
            "label": "Young Female",
            "skin": (185, 205, 225),
            "hair": (20, 15, 50),
            "eye": (90, 70, 45),
            "lip": (140, 130, 200),
            "bg": (230, 210, 220),
        },
        {
            "name": "elder_male",
            "label": "Elder Male",
            "skin": (170, 195, 215),
            "hair": (180, 180, 180),
            "eye": (70, 55, 35),
            "lip": (115, 125, 165),
            "bg": (195, 195, 195),
        },
    ]

    manifest = []
    for a in avatars:
        img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        draw_face(img, a["skin"], a["hair"], a["eye"], a["lip"], a["bg"])
        path = os.path.join(OUTPUT_DIR, f"{a['name']}.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        manifest.append({"name": a["name"], "label": a["label"], "file": f"/avatars/{a['name']}.jpg"})
        print(f"Created: {path}")

    print(f"\nGenerated {len(manifest)} default avatars in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
