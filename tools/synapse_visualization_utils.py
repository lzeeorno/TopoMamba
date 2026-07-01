import numpy as np


SYNAPSE_ORGAN_COLORS = {
    1: (255, 16, 16),     # Aorta
    2: (0, 240, 24),      # Gallbladder
    3: (20, 40, 255),     # Left kidney
    4: (255, 242, 0),     # Right kidney
    5: (255, 0, 240),     # Liver
    6: (0, 229, 240),     # Pancreas
    7: (255, 138, 0),     # Spleen
    8: (138, 22, 255),    # Stomach
}


def normalize_gray_image(image):
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    image_min = float(np.min(image))
    image_max = float(np.max(image))
    if 0.0 <= image_min and image_max <= 1.0:
        scaled = image * 255.0
    elif image_max > image_min:
        scaled = (image - image_min) / (image_max - image_min) * 255.0
    else:
        scaled = np.zeros_like(image, dtype=np.float32)

    return np.clip(scaled, 0, 255).astype(np.uint8)


def colorize_label_mask(labels):
    labels = np.asarray(labels).astype(np.uint8)
    rgb = np.zeros(labels.shape + (3,), dtype=np.uint8)

    for label, color in SYNAPSE_ORGAN_COLORS.items():
        rgb[labels == label] = color

    return rgb


def overlay_label_mask(image, labels, alpha=0.72):
    gray = normalize_gray_image(image)
    output = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
    labels = np.asarray(labels).astype(np.uint8)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    for label, color in SYNAPSE_ORGAN_COLORS.items():
        mask = labels == label
        if not np.any(mask):
            continue
        color_arr = np.asarray(color, dtype=np.float32)
        output[mask] = (1.0 - alpha) * output[mask] + alpha * color_arr

    return np.clip(output, 0, 255).astype(np.uint8)
