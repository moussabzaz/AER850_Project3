import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pathlib import Path
from ultralytics import YOLO

# ----------------- GLOBAL CONFIG ----------------- #


BASE_DIR = Path(r"C:\Users\mouss\AER850_Project3_Data")

MOTHERBOARD_IMG = BASE_DIR / "motherboard_image.JPEG"
YOLO_DATA_YAML = BASE_DIR / "data" / "data.yaml"
EVAL_DIR = BASE_DIR / "data" / "evaluation"

RUN_NAME = "pcb_yolo11n_run"

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ----------------- STEP 1: PCB MASKING ----------------- #

def create_pcb_mask(
    img_path: Path = MOTHERBOARD_IMG,
    output_dir: Path = Path("step1_results"),
    show_figs: bool = True,
):
    """
    Reads the motherboard image, extracts the PCB using contour-based masking,
    and saves + optionally displays: edges, mask and extracted PCB.
    """
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")

    # Rotate to make the PCB upright (same idea as your friends')
    img_rot = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)

    # Strong blur to smooth out noise and small details
    blurred = cv2.GaussianBlur(img_rot, (41, 41), 3)

    # Grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding to separate foreground/background
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        51,
        7,
    )

    # Edge detection (Canny)
    edges = cv2.Canny(thresh, 40, 300)

    # Dilate edges to close gaps
    edges_dilated = cv2.dilate(edges, None, iterations=10)

    # Find all external contours
    contours, _ = cv2.findContours(
        edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("No contours found for PCB extraction")

    # Take the largest contour as the PCB
    biggest = max(contours, key=cv2.contourArea)

    # Build an empty mask and fill the largest contour
    mask = np.zeros_like(img_rot)
    cv2.drawContours(mask, [biggest], contourIdx=-1,
                     color=(255, 255, 255), thickness=cv2.FILLED)

    # Apply mask to the rotated image
    extracted = cv2.bitwise_and(img_rot, mask)

    # Make sure output dir exists
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save images
    cv2.imwrite(str(output_dir / "edges.png"), edges)
    cv2.imwrite(str(output_dir / "mask.png"), mask)
    cv2.imwrite(str(output_dir / "pcb_extracted.png"), extracted)

    if show_figs:
        # Convert to RGB for Matplotlib
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        extracted_rgb = cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(edges_rgb)
        plt.title("Edge Map")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask_rgb)
        plt.title("PCB Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(extracted_rgb)
        plt.title("Extracted PCB")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


# --------------- MAIN BLOCK --------------- #

if __name__ == "__main__":
    print("Running PCB masking...")
    create_pcb_mask()
    print("Done.")
