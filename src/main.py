import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Noise_Filtering


class TrajectoryAnalyzer:
    def __init__(self, initial_velocity=(0.0, 0.0, 0.0), initial_orientation=(0.0, 0.0, 0.0)):
        self.initial_velocity = np.array(initial_velocity)
        self.initial_orientation = np.array(initial_orientation)
        self.acceleration_data = None
        self.time_intervals = None
        self.positions = None
        self.orientations = [self.initial_orientation]

    def convert_g_to_mps2(df, columns):
        """
        Converts acceleration values from G to m/s^2.

        :param df: pandas DataFrame
        :param columns: list of column names to convert
        """
        G = 9.80665  # 1G in m/s²
        df[columns] = df[columns] * G

    def load_data(self, file_path):
        try:
            # Load CSV using comma separator
            data = pd.read_csv(file_path, sep=',')

            # Clean column names
            data.columns = data.columns.str.strip()

            # Debug print
            print("Cleaned column names:", data.columns.tolist())

            # Select only the relevant columns
            required_cols = ['Chiptime', 'ax(g)', 'ay(g)', 'az(g)', 'wx(deg/s)', 'wy(deg/s)', 'wz(deg/s)']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Found only: {data.columns.tolist()}")

            data = data[required_cols]
            data.columns = ['Time', 'ax_g', 'ay_g', 'az_g', 'wx', 'wy', 'wz']

            # Convert acceleration from G to m/s²
            G = 9.80665
            data['ax'] = data['ax_g'] * G
            data['ay'] = data['ay_g'] * G
            data['az'] = data['az_g'] * G
            data.drop(columns=['ax_g', 'ay_g', 'az_g'], inplace=True)

            # Format Time column: convert last colon to dot
            data['Time'] = data['Time'].astype(str).str.strip()
            data['Time'] = data['Time'].str.replace(r'(?<=\d{2}:\d{2}:\d{2}):', '.', regex=True)
            data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S.%f', errors='coerce')

            if data['Time'].isnull().any():
                bad_times = data.loc[data['Time'].isnull(), 'Time'].head(3).tolist()
                raise ValueError(f"Some time values could not be parsed. Examples: {bad_times}")

            # Compute time intervals
            self.time_intervals = data['Time'].diff().dt.total_seconds().fillna(0).values
            self.time_intervals[0] = self.time_intervals[1:].mean()
            self.time_intervals = np.insert(self.time_intervals, 0, 0)

            # Add initial zero row for calculation
            zero_row = pd.DataFrame([[pd.NaT, 0, 0, 0, 0, 0, 0]], columns=['Time', 'ax', 'ay', 'az', 'wx', 'wy', 'wz'])
            self.acceleration_data = pd.concat([zero_row, data], ignore_index=True)

        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    def calculate_trajectory(self):
        if self.acceleration_data is None:
            raise RuntimeError("Acceleration data not loaded. Use load_data() first.")

        accel = self.acceleration_data[['ax', 'ay', 'az']].values
        spin = self.acceleration_data[['wx', 'wy', 'wz']].values * np.pi / 180  # Convert to radians/sec

        position = np.zeros((len(accel), 3))
        velocity = np.zeros((len(accel), 3))
        velocity[0] = self.initial_velocity
        self.orientations = [self.initial_orientation]

        for i in range(1, len(accel)):
            dt = self.time_intervals[i]  # Use actual time interval

            # Update orientation
            angular_velocity = spin[i] * dt
            orientation = angular_velocity

            # Calculate rotation matrix
            R = rotation_matrix(*orientation)

            # Rotate accumulated velocity to the new orientation
            velocity[i - 1] = R @ velocity[i - 1]

            # Rotate acceleration to global frame
            rotated_accel = R @ accel[i]

            # Integrate acceleration for velocity
            velocity[i] = velocity[i - 1] + rotated_accel * dt

            # Integrate velocity for position
            position[i] = position[i - 1] + velocity[i] * dt

            # Save orientation
            if i < len(self.orientations):
                self.orientations[i] = orientation.copy()
            else:
                self.orientations.append(orientation.copy())

        self.positions = position

    def save_trajectory(self, output_file):
        if self.positions is None or self.orientations is None:
            raise RuntimeError("Trajectory or orientations not calculated. Use calculate_trajectory() first.")

        trajectory_df = pd.DataFrame(self.positions, columns=['x', 'y', 'z'])
        orientation_df = pd.DataFrame(self.orientations, columns=['roll', 'pitch', 'yaw'])
        time_df = pd.DataFrame(self.acceleration_data['Time'].values, columns=['Time'])

        result_df = pd.concat([time_df, trajectory_df, orientation_df], axis=1)
        result_df.to_csv(output_file, index=False)
        print(f"Trajectory and orientations saved to {output_file}")

    def plot_trajectory(self):
        """
        Plot the 3D trajectory with time markers.
        """
        if self.positions is None:
            raise RuntimeError("Trajectory not calculated. Use calculate_trajectory() first.")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the 3D trajectory
        ax.plot(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2], label="Trajectory", lw=2)

        # # Mark points every second with timestamps
        # time_values = self.acceleration_data['Time'].dt.strftime('%H:%M:%S:%f').values  # הצגת שעה, דקה ושנייה בלבד
        # for i in range(0, len(self.positions), 10):  # מניחים בערך 10 מדידות בשנייה, ניתן לשנות בהתאם
        #     ax.scatter(self.positions[i, 0], self.positions[i, 1], self.positions[i, 2], color='red', s=50)
        #     ax.text(self.positions[i, 0], self.positions[i, 1], self.positions[i, 2], time_values[i], fontsize=9,
        #             color='black')

        # Set labels and title
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")
        ax.set_title("3D Calculated Trajectory with Time Markers")

        # Add grid and legend
        ax.grid(True)
        ax.legend()
        plt.show()

    def run(self, file_path, output_file):
        self.load_data(file_path)
        self.calculate_trajectory()
        self.save_trajectory(output_file)
        self.plot_trajectory()


def rotation_matrix(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    return Rz @ Ry @ Rx


if __name__ == "__main__":
    # File paths
    clean_file = 'Routes/22042025/20250422130242.csv'
    noise_file = 'circle_with_noise.csv'
    clean_file_output = 'clean_file_output.csv'
    noise_file_output = 'noise_file_output.csv'

    # Initialize TrajectoryAnalyzer
    analyzer = TrajectoryAnalyzer(initial_velocity=(0.0, 0.0, 0.0), initial_orientation=(0.0, 0.0, 0.0))

    # Process clean file
    analyzer.run(clean_file, clean_file_output)

    # Add synthetic noise to the clean file and save as noise_file
    Noise_Filtering.add_synthetic_noise(clean_file, noise_file, noise_level=0.01)

    # Process noise file
    analyzer.run(noise_file, noise_file_output)

    # Columns to compare
    columns_to_compare = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

    # Calculate Mean Squared Error (MSE) for each column
    mse_noise_results = Noise_Filtering.calculate_mse(clean_file_output, noise_file_output, columns_to_compare)

    # Print MSE results
    print("Mean Squared Error (MSE) Results:")
    for col in columns_to_compare:
        mse_before = mse_noise_results[col]
        print(f"  Column: {col}")
        print(f"    MSE : {mse_before:.6f}")

    # Compare start and end locations
    result = Noise_Filtering.compare_start_end_locations(clean_file_output, noise_file_output, None, None)
    print("\nStart and End Location Errors:")
    print(f"  Start Location Error: {result[0]}")
    print(f"  End Location Error: {result[1]}")

