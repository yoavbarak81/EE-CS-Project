import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from datetime import datetime

class TrajectoryAnalyzer:
    def __init__(self, initial_velocity=(0.0, 0.0, 0.0), initial_orientation=(0.0, 0.0, 0.0)):
        self.initial_velocity = np.array(initial_velocity)
        self.initial_orientation = np.array(initial_orientation)
        self.acceleration_data = None
        self.time_intervals = None
        self.positions = None
        self.orientations = [self.initial_orientation]

    def load_data(self, file_path):
        try:
            data = pd.read_csv(file_path)

            required_columns = ['Time', 'Accel x[m/s^2]', 'Accel y[m/s^2]', 'Accel z[m/s^2]',
                                'Spin x[degrees/sec]', 'Spin y[degrees/sec]', 'Spin z[degrees/sec]']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"File must contain {required_columns} columns.")

            # Convert timestamps to datetime objects
            data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S:%f')

            # Compute time intervals in seconds
            self.time_intervals = data['Time'].diff().dt.total_seconds().fillna(0).values
            self.time_intervals[0] = self.time_intervals[1:].mean()
            self.time_intervals = np.insert(self.time_intervals, 0, 0)
            self.acceleration_data = data[['Time', 'Accel x[m/s^2]', 'Accel y[m/s^2]', 'Accel z[m/s^2]',
                                           'Spin x[degrees/sec]', 'Spin y[degrees/sec]', 'Spin z[degrees/sec]']].copy()
            zero_row = pd.DataFrame([[pd.NaT, 0, 0, 0, 0, 0, 0]], columns=self.acceleration_data.columns)
            self.acceleration_data = pd.concat([zero_row, self.acceleration_data], ignore_index=True)

            self.acceleration_data.rename(columns={
                'Accel x[m/s^2]': 'ax',
                'Accel y[m/s^2]': 'ay',
                'Accel z[m/s^2]': 'az',
                'Spin x[degrees/sec]': 'wx',
                'Spin y[degrees/sec]': 'wy',
                'Spin z[degrees/sec]': 'wz'
            }, inplace=True)

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
        orientation = np.array(self.initial_orientation)

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


# Example usage
file_path = 'ACCdd.csv'  # Replace with the actual path
output_file = "trajectory_output.csv"

analyzer = TrajectoryAnalyzer(initial_velocity=(0.0, 0.0, 0.0), initial_orientation=(0.0, 0.0, 0.0))
analyzer.load_data(file_path)
analyzer.calculate_trajectory()
analyzer.save_trajectory(output_file)
analyzer.plot_trajectory()
