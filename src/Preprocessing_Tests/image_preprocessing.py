import cv2
import numpy as np
import os





import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product


# ==========================================
# 1. CONFIGURATION
# ==========================================

# Load images and test different CHALE preprocessing



classes = ["0", "1", "2", "3", "4"]
IMAGE_PATH = "../../Raw_Images/4/SevereG4 (32).png"

OUTPUT_DIR = "CLAHE_Experiments"

CLIP_LIMITS = [1.0, 2.0, 3.0, 4.0, 5.0]

TILE_GRID_SIZES = [
    (4, 4),
    (8, 8),
    (12, 12),
    (16, 16)
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. LOAD IMAGE
# ==========================================

image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

print("Image loaded:", image.shape)


# ==========================================
# 3. METRICS
# ==========================================

def calculate_metrics(image):

    contrast = np.std(image)

    mean_intensity = np.mean(image)

    min_value = np.min(image)

    max_value = np.max(image)

    # Entropy
    hist = cv2.calcHist(
        [image],
        [0],
        None,
        [256],
        [0, 256]
    )

    hist = hist / hist.sum()

    entropy = -np.sum(
        hist * np.log2(hist + 1e-10)
    )

    return {
        "Contrast": contrast,
        "Mean": mean_intensity,
        "Min": min_value,
        "Max": max_value,
        "Entropy": entropy
    }


# ==========================================
# 4. ORIGINAL IMAGE
# ==========================================

results = []

original_metrics = calculate_metrics(image)

results.append({
    "Configuration": "Original",
    "ClipLimit": None,
    "TileGridSize": None,
    **original_metrics
})

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "original.png"),
    image
)


# ==========================================
# 5. CLAHE EXPERIMENTS
# ==========================================

processed_images = {
    "Original": image
}

for clip_limit, tile_size in product(
    CLIP_LIMITS,
    TILE_GRID_SIZES
):

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_size
    )

    processed = clahe.apply(image)

    config_name = (
        f"CLAHE_clip{clip_limit}_"
        f"tile{tile_size[0]}x{tile_size[1]}"
    )

    # Save processed image
    output_path = os.path.join(
        OUTPUT_DIR,
        config_name + ".png"
    )

    cv2.imwrite(output_path, processed)

    processed_images[config_name] = processed

    # Calculate metrics
    metrics = calculate_metrics(processed)

    results.append({
        "Configuration": config_name,
        "ClipLimit": clip_limit,
        "TileGridSize": str(tile_size),
        **metrics
    })

    print("Processed:", config_name)


# ==========================================
# 6. SAVE METRICS
# ==========================================

df = pd.DataFrame(results)

df.to_csv(
    os.path.join(OUTPUT_DIR, "metrics.csv"),
    index=False
)

print("\nMetrics:")
print(df.to_string(index=False))


# ==========================================
# 7. VISUAL COMPARISON
# ==========================================

n_images = len(processed_images)

n_cols = 4
n_rows = int(np.ceil(n_images / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(16, 4 * n_rows)
)

axes = np.array(axes).flatten()

for ax, (name, img) in zip(
    axes,
    processed_images.items()
):

    ax.imshow(img, cmap="gray", vmin=0, vmax=255)

    ax.set_title(name, fontsize=9)

    ax.axis("off")

for ax in axes[n_images:]:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "comparison.png"),
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nExperiments completed!")
print("Results saved in:", OUTPUT_DIR)
