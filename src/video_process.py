
import cv2
import numpy as np


def extract_weld_timestamps(video_path, detection_threshold=1750):
    """
    Detect welds based on color and return a list of timestamps (in seconds).

    :param video_path: Path to the video file.
    :param detection_threshold: Minimum number of brown pixels to qualify as weld.
    :return: List of timestamps in seconds where welds are detected.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamps = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create annular region mask
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        radius_outer = int(frame.shape[0] * 0.47)
        radius_inner = int(frame.shape[0] * 0.30)
        cv2.circle(mask, center, radius_outer, 255, -1)
        cv2.circle(mask, center, radius_inner, 0, -1)

        # Threshold for brown hue
        lower_brown = np.array([5, 40, 40])
        upper_brown = np.array([30, 255, 255])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        combined_mask = cv2.bitwise_and(brown_mask, brown_mask, mask=mask)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        brown_pixels = cv2.countNonZero(cleaned)

        if brown_pixels > detection_threshold:
            time_sec = frame_count / fps
            if not timestamps or abs(time_sec - timestamps[-1]) > 0.01:  # avoid tight duplicates
                timestamps.append(round(time_sec, 3))

        frame_count += 1

    cap.release()
    return timestamps


def filter_and_average_timestamps(timestamps, min_gap=0.04):
    """
    Group close timestamps into a single averaged timestamp.

    :param timestamps: List of timestamps (floats, in seconds), sorted or unsorted.
    :param min_gap: Maximum time (in seconds) between adjacent timestamps to consider them part of the same weld.
    :return: List of averaged timestamps, one per detected weld group.
    """
    if not timestamps:
        return []

    timestamps = sorted(timestamps)
    groups = []
    current_group = [timestamps[0]]

    for i in range(1, len(timestamps)):
        if timestamps[i] - current_group[-1] <= min_gap:
            current_group.append(timestamps[i])
        else:
            groups.append(current_group)
            current_group = [timestamps[i]]

    groups.append(current_group)
    averaged = [round(sum(group) / len(group), 3) for group in groups if len(group) >= 2]
    return averaged


def estimate_speeds(timestamps, segment_length=12.0, max_speed_threshold=10.0):
    """
    Estimate speed between consecutive weld timestamps, skipping unrealistic fast intervals.

    :param timestamps: List of filtered weld timestamps (in seconds).
    :param segment_length: Distance between welds in meters.
    :param max_speed_threshold: Maximum allowed speed (m/s). Intervals above this will be skipped.
    :return: List of speeds (m/s) between valid weld pairs.
    """
    speeds = []
    i = 1
    while i < len(timestamps):
        delta_time = timestamps[i] - timestamps[i - 1]
        if delta_time > 0:
            speed = segment_length / delta_time
            if speed <= max_speed_threshold:
                speeds.append(round(speed, 2))
                i += 1
            else:
                # skip this timestamp and try the next one
                i += 1
        else:
            i += 1
    return speeds


# Example usage
if __name__ == "__main__":
    video_file = "videos/2024-06-24_11_41_27 (1).mp4"
    raw_times = extract_weld_timestamps(video_file)
    clean_times = filter_and_average_timestamps(raw_times)
    speeds = estimate_speeds(clean_times)
    print(raw_times)
    print("Filtered Weld timestamps (seconds):", clean_times)
    print("Estimated Speeds (m/s):", speeds)
