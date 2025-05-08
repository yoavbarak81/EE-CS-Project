import numpy as np
import pandas as pd


def add_synthetic_noise(file_path, output_file, noise_level=0.1):
    # Load the data
    data = pd.read_csv(file_path)

    # Define the columns to add noise to
    accel_columns = ['Accel x[m/s^2]', 'Accel y[m/s^2]', 'Accel z[m/s^2]']
    spin_columns = ['Spin x[degrees/sec]', 'Spin y[degrees/sec]', 'Spin z[degrees/sec]']

    # Add Gaussian noise to the acceleration and spin data
    for col in accel_columns + spin_columns:
        noise = np.random.normal(0, noise_level, size=len(data))
        data[col] += noise

    # Save the noisy data to a new CSV file
    data.to_csv(output_file, index=False)
    print(f"Noisy data saved to {output_file}")


def calculate_mse(file1, file2, columns):
    """
    Calculate the Mean Squared Error (MSE) between two CSV files for specified columns.

    :param file1: Path to the first CSV file
    :param file2: Path to the second CSV file
    :param columns: List of columns to compare (e.g., ['x', 'y', 'z', 'roll', 'pitch', 'yaw'])
    :return: Dictionary of MSE values for each column
    """
    # Load the CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Ensure the DataFrames have the same length
    if len(df1) != len(df2):
        raise ValueError("The two files must have the same number of rows.")

    # Calculate MSE for each column
    mse_results = {}
    for col in columns:
        squared_diff = (df1[col] - df2[col]) ** 2
        mse = squared_diff.mean()
        mse_results[col] = mse

    return mse_results


def moving_average_filter(file, columns, window_size, output_file):
    """
    Apply a moving average filter with padding to smooth the data and save to a new CSV file.

    :param file: Path to a CSV file.
    :param columns: List of columns to apply the moving average filter.
    :param window_size: Size of the moving window.
    :param output_file: Path to save the filtered data.
    """
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file)

    # Apply moving average filter with padding to each column
    for col in columns:
        if col in data.columns:
            # Convert the column to a NumPy array for easier manipulation
            column_data = data[col].values

            # Pad the data at the beginning and end
            pad_size = window_size // 2
            padded_data = np.pad(column_data, (pad_size, pad_size), mode='edge')

            # Apply the rolling window and calculate the mean
            smoothed_data = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')

            # Assign the smoothed data back to the column
            data[col] = smoothed_data
        else:
            print(f"Warning: Column '{col}' not found in the file.")

    # Save the filtered data to a new CSV file
    data.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")


def compare_start_end_locations(clean_file, predict_file, start_location_clean=None, end_location_clean=None):
    # Load clean data if start or end locations are not provided
    if start_location_clean is None or end_location_clean is None:
        clean_data = pd.read_csv(clean_file)
        start_location_clean = clean_data[['x', 'y', 'z']].iloc[1].values
        end_location_clean = clean_data[['x', 'y', 'z']].iloc[-1].values

    # Load predict data
    predict_data = pd.read_csv(predict_file)
    start_location_predict = predict_data[['x', 'y', 'z']].iloc[1].values
    end_location_predict = predict_data[['x', 'y', 'z']].iloc[-1].values

    # Calculate errors
    start_error = np.abs(start_location_predict - start_location_clean)
    end_error = np.abs(end_location_predict - end_location_clean)

    # Return results
    return [start_error.tolist(), end_error.tolist()]

def validate_video_vs_imu_speeds(self, segment_length=12.0, show_plot=True):
    """
    Compare speeds estimated from video welds with speeds from IMU-based trajectory.
    Print statistics and optionally show plot.

    :param segment_length: Assumed distance between welds (meters)
    :param show_plot: Whether to display a comparison plot
    """
    if self.positions is None or not self.video_speeds:
        print("Missing data for validation.")
        return

    imu_times = self.acceleration_data['Time'].dropna().astype('datetime64[ns]').astype(np.int64) / 1e9
    imu_positions = np.linalg.norm(self.positions, axis=1)

    imu_estimated_speeds = []
    video_times = [t for t, _ in self.video_speeds]

    for i in range(1, len(video_times)):
        t0, t1 = video_times[i - 1], video_times[i]
        dt = t1 - t0
        if dt <= 0:
            continue

        # distance between those two times according to IMU
        p0 = np.interp(t0, imu_times, imu_positions)
        p1 = np.interp(t1, imu_times, imu_positions)
        dist = p1 - p0
        speed = dist / dt
        imu_estimated_speeds.append((t1, speed))

    # compare
    video_speeds_only = [s for _, s in self.video_speeds[1:]]  # skip first (we're comparing deltas)
    imu_speeds_only = [s for _, s in imu_estimated_speeds]

    if len(imu_speeds_only) != len(video_speeds_only):
        print("Warning: mismatch in lengths between video and IMU speeds")

    diffs = np.array(imu_speeds_only) - np.array(video_speeds_only)
    mae = np.mean(np.abs(diffs))
    rmse = np.sqrt(np.mean(diffs ** 2))
    max_err = np.max(np.abs(diffs))

    print(f"\nValidation Results (IMU vs Video):")
    print(f"  MAE  : {mae:.3f} m/s")
    print(f"  RMSE : {rmse:.3f} m/s")
    print(f"  Max error: {max_err:.3f} m/s")

    if show_plot:
        import matplotlib.pyplot as plt
        x_axis = [round(t, 1) for t, _ in imu_estimated_speeds]
        plt.plot(x_axis, imu_speeds_only, label='IMU estimated', marker='o')
        plt.plot(x_axis, video_speeds_only, label='Video (ground truth)', marker='x')
        plt.title("Speed Comparison: IMU vs Video")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (m/s)")
        plt.legend()
        plt.grid(True)
        plt.show()
