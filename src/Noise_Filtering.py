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


import pandas as pd
import numpy as np


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
