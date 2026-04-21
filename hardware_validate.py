import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Helper functions for plotting
# =========================================================
def build_binary_signal(intervals, t):
    signal = np.zeros_like(t, dtype=int)
    for start, end, label in intervals:
        label = label.strip().lower()

        # Only motor vibration intervals should be 1
        # Spaces mean the motor is off, so they remain 0
        if label in ["dot", "dash"]:
            signal[(t >= start) & (t < end)] = 1
    return signal


def plot_aligned(test_intervals, theoretical_intervals, title, dt=0.01):
    t_end = max(
        max(end for _, end, _ in test_intervals),
        max(end for _, end, _ in theoretical_intervals)
    )
    t = np.arange(0, t_end + dt, dt)

    signal_test = build_binary_signal(test_intervals, t)
    signal_theoretical = build_binary_signal(theoretical_intervals, t)

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axs[0].step(t, signal_test, where='post', linewidth=2)
    axs[0].set_ylim(-0.2, 1.2)
    axs[0].set_ylabel("Test")
    axs[0].set_title(f"{title} - Test (Audio)")
    axs[0].grid(True)

    axs[1].step(t, signal_theoretical, where='post', linewidth=2)
    axs[1].set_ylim(-0.2, 1.2)
    axs[1].set_ylabel("Theoretical")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_title(f"{title} - Theoretical")
    axs[1].grid(True)

    plt.tight_layout()

    # Save figure to current directory using title as filename
    filename = title.replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")


# =========================================================
# New interval-based accuracy calculation
# =========================================================
THEORETICAL_DURATION = {
    "dot": 0.8,
    "dash": 2.0,
    "letter space": 3.0,
    "word space": 6.0
}

def normalize_label(label):
    return label.strip().lower()


def compute_interval_accuracy(test_intervals, phrase_name=""):
    """
    test_intervals format:
    [
        (start, end, label),
        ...
    ]
    """
    interval_accuracies = []

    print(f"\n===== {phrase_name} Interval Accuracy =====")

    for i, (start, end, label) in enumerate(test_intervals, start=1):
        label_norm = normalize_label(label)
        measured_duration = end - start
        theoretical_duration = THEORETICAL_DURATION[label_norm]

        error_ratio = abs(measured_duration - theoretical_duration) / theoretical_duration
        interval_accuracy = 1 - error_ratio

        # Clamp to zero if the error is too large
        interval_accuracy = max(interval_accuracy, 0)

        interval_accuracies.append(interval_accuracy)

        print(
            f"Interval {i:2d}: {label:12s} | "
            f"Measured = {measured_duration:5.2f} s | "
            f"Theoretical = {theoretical_duration:4.1f} s | "
            f"Accuracy = {interval_accuracy * 100:6.2f}%"
        )

    overall_accuracy = np.mean(interval_accuracies)

    print(f"\n{phrase_name} Overall Accuracy = {overall_accuracy * 100:.2f}%")
    return overall_accuracy


# =========================================================
# 1. HI
# =========================================================
hi_test_intervals = [
    (0.00, 0.82, "dot"),
    (1.82, 2.65, "dot"),
    (3.71, 4.52, "dot"),
    (5.54, 6.55, "dot"),
    (7.55, 10.54, "letter space"),
    (10.54, 11.44, "dot"),
    (12.42, 13.35, "dot"),
    (13.35, 16.25, "letter space"),
]

hi_theoretical_intervals = [
    (0.00, 0.80, "dot"),
    (1.80, 2.60, "dot"),
    (3.60, 4.40, "dot"),
    (5.40, 6.20, "dot"),
    (7.20, 10.20, "letter space"),
    (10.20, 11.00, "dot"),
    (12.00, 12.80, "dot"),
    (12.80, 15.80, "letter space"),
]


# =========================================================
# 2. HOW ARE YOU
# =========================================================
how_are_you_test_intervals = [
    (0.00, 0.64, "dot"),
    (1.69, 2.52, "dot"),
    (3.60, 4.42, "dot"),
    (5.54, 6.34, "dot"),
    (7.34, 10.40, "letter space"),
    (10.40, 12.43, "dash"),
    (13.30, 15.30, "dash"),
    (16.60, 18.50, "dash"),
    (19.50, 22.52, "letter space"),
    (22.52, 23.43, "dot"),
    (24.51, 26.45, "dash"),
    (27.32, 29.52, "dash"),
    (30.52, 36.83, "word space"),
    (36.83, 37.72, "dot"),
    (38.72, 40.64, "dash"),
    (41.64, 44.50, "letter space"),
    (44.50, 45.62, "dot"),
    (46.50, 48.54, "dash"),
    (49.82, 50.62, "dot"),
    (51.62, 54.72, "letter space"),
    (54.72, 55.52, "dot"),
    (56.52, 62.81, "word space"),
    (62.81, 64.64, "dash"),
    (65.58, 66.52, "dot"),
    (67.50, 69.40, "dash"),
    (70.21, 72.52, "dash"),
    (73.52, 76.72, "letter space"),
    (76.72, 78.92, "dash"),
    (79.73, 81.74, "dash"),
    (83.12, 85.04, "dash"),
    (86.04, 88.88, "letter space"),
    (88.88, 89.92, "dot"),
    (90.74, 91.76, "dot"),
    (93.62, 94.82, "dash"),
    (95.82, 98.82, "letter space"),
]

how_are_you_theoretical_intervals = [
    (0.00, 0.80, "dot"),
    (1.80, 2.60, "dot"),
    (3.60, 4.40, "dot"),
    (5.40, 6.20, "dot"),
    (7.20, 10.20, "letter space"),
    (10.20, 12.20, "dash"),
    (13.20, 15.20, "dash"),
    (16.20, 18.20, "dash"),
    (19.20, 22.20, "letter space"),
    (22.20, 23.00, "dot"),
    (24.00, 26.00, "dash"),
    (27.00, 29.00, "dash"),
    (30.00, 36.00, "word space"),
    (36.00, 36.80, "dot"),
    (37.80, 39.80, "dash"),
    (40.80, 43.80, "letter space"),
    (43.80, 44.60, "dot"),
    (45.60, 47.60, "dash"),
    (48.60, 49.40, "dot"),
    (50.40, 53.40, "letter space"),
    (53.40, 54.20, "dot"),
    (55.20, 61.20, "word space"),
    (61.20, 63.20, "dash"),
    (64.20, 65.00, "dot"),
    (66.00, 68.00, "dash"),
    (69.00, 71.00, "dash"),
    (72.00, 75.00, "letter space"),
    (75.00, 77.00, "dash"),
    (78.00, 80.00, "dash"),
    (81.00, 83.00, "dash"),
    (84.00, 87.00, "letter space"),
    (87.00, 87.80, "dot"),
    (88.80, 89.60, "dot"),
    (90.60, 92.60, "dash"),
    (93.60, 96.60, "letter space"),
]


# =========================================================
# Run plotting — saves figures to current directory
# =========================================================
plot_aligned(hi_test_intervals, hi_theoretical_intervals, "Hi")
plot_aligned(how_are_you_test_intervals, how_are_you_theoretical_intervals, "How_Are_You")

# =========================================================
# Run new interval-based accuracy
# =========================================================
hi_accuracy = compute_interval_accuracy(hi_test_intervals, "Hi")
how_are_you_accuracy = compute_interval_accuracy(how_are_you_test_intervals, "How Are You")