"""Assignment 03: summarize a telemetry ward's systolic readings.

Run from the assignment directory with the project environment active:

    python3 analysis.py
"""

import numpy as np


def load_readings(filename):
    """Return (patient_ids, monitors, hour_columns, readings) from the supplied CSV.

    readings is a 2D array of integers: one row per patient, one column per
    monitored hour, in the order the header lists them.
    """
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    header = lines[0].strip().split(",")
    rows = [line.strip().split(",") for line in lines[1:] if line.strip()]

    patient_ids = np.array([row[0] for row in rows])
    monitors = np.array([row[1] for row in rows])
    hour_columns = np.array(header[2:])
    readings = np.array([row[2:] for row in rows]).astype(int)
    return patient_ids, monitors, hour_columns, readings


def main():
    patient_ids, monitors, hour_columns, readings = load_readings("data/bp_readings.csv")
    print(f"Loaded {readings.shape[0]} patients x {readings.shape[1]} hours")

    patient_means = readings.mean(axis=1)
    monitor_names = sorted(set(monitors))

    monitor_averages = np.array([
        patient_means[monitors == monitor].mean()
        for monitor in monitor_names
    ])

    highest_patient_index = patient_means.argmax()
    hour_means = readings.mean(axis=0)
    peak_hour_index = hour_means.argmax()
    high_monitor_index = monitor_averages.argmax()
    high_monitor = monitor_names[high_monitor_index]
    other_monitors = monitors != high_monitor

    results = {
        "patients": len(patient_ids),
        "readings": readings.size,
        "mean_sbp": readings.mean(),
        "sd_sbp": readings.std(),
        "min_sbp": readings.min(),
        "max_sbp": readings.max(),
        "stage2_patients": (patient_means >= 140).sum(),
        "highest_patient": patient_ids[highest_patient_index],
        "highest_patient_mean": patient_means[highest_patient_index],
        "peak_hour_column": hour_columns[peak_hour_index],
        "peak_hour_mean": hour_means[peak_hour_index],
        "high_monitor": high_monitor,
        "monitor_offset": (
            patient_means[monitors == high_monitor].mean()
            - patient_means[other_monitors].mean()
        ),
        "stage2_other_monitors": (
            patient_means[other_monitors] >= 140
        ).sum(),
    }

    with open("output/vitals_summary.txt", "w", encoding="utf-8") as file:
        for key, value in results.items():
            file.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()
