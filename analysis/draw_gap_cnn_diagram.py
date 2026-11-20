"""Editable diagram of the CNN used by the CARLA gap controllers."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def draw_cnn_diagram():
    """Return a five-stage figure with text and shapes available for SVG editing."""
    navy = "#1B365D"
    teal = "#007F78"
    slate = "#465C73"
    border = "#ADC0D2"
    figure = plt.figure(figsize=(14.6, 2.65), facecolor="white")
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set(xlim=(0, 14.6), ylim=(0, 2.65), aspect="equal")
    axis.axis("off")
    axis.text(
        0.55, 2.25, "CNN gap estimator", color=navy, fontsize=20,
        fontfamily="DejaVu Sans", fontweight="bold", va="center",
    )

    # The live camera and saved RGB frames both have 224 × 224 pixels.
    stages = (
        (2.40, "RGB input", "B × 3 × 224 × 224", "#F6F9FC", border),
        (3.05, "ResNet-18 backbone", "Global average pooling", "#EEF5FB", border),
        (1.65, "Features", "B × 512", "#F6F9FC", border),
        (2.10, "Linear(512, 1)", "", "#F6F9FC", border),
        (2.29, "Predicted gap", "B × 1 · metres", "#E8F5F1", teal),
    )
    left, bottom, height, gap = 0.55, 0.64, 1.07, 0.49
    center_y = bottom + height / 2
    for index, (width, title, detail, fill, edge) in enumerate(stages):
        axis.add_patch(FancyBboxPatch(
            (left, bottom), width, height, boxstyle="round,pad=0.03,rounding_size=0.13",
            facecolor=fill, edgecolor=edge, linewidth=1.15,
        ))
        label_style = {"ha": "center", "va": "center", "fontfamily": "DejaVu Sans"}
        axis.text(
            left + width / 2, center_y + (0.15 if detail else 0.0), title,
            color=navy, fontsize=13.25, fontweight="bold", **label_style,
        )
        if detail:
            axis.text(
                left + width / 2, center_y - 0.19, detail,
                color=slate, fontsize=10.9, **label_style,
            )
        if index + 1 < len(stages):
            right = left + width
            axis.add_patch(FancyArrowPatch(
                (right + 0.09, center_y), (right + gap - 0.09, center_y),
                arrowstyle="-|>", shrinkA=0, shrinkB=0, mutation_scale=11,
                color=teal, linewidth=1.5,
            ))
        left += width + gap
    return figure
