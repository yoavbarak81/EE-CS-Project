import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TrajectoryAnalyzer:
    def __init__(self, initial_velocity=(0.0, 0.0, 0.0), initial_orientation=(0.0, 0.0, 0.0)):
        """
        Initialize the TrajectoryAnalyzer with an initial velocity and orientation.

        :param initial_velocity: Tuple of initial velocities (vx, vy, vz) in m/s
        :param initial_orientation: Tuple of initial orientation (roll, pitch, yaw) in radians
        """
        self.initial_velocity = np.array(initial_velocity)
        self.initial_orientation = np.array(initial_orientation)
        self.acceleration_data = None
        self.time_intervals = None
        self.positions = None
        self.orientations = [self.initial_orientation]

    def load_data(self, file_path):
        """
        Load acceleration and angular velocity data from a CSV file.

        :param file_path: Path to the file containing acceleration and angular velocity data
        The file must have columns: Time, Accel x[m/s^2], Accel y[m/s^2], Accel z[m/s^2],
        Spin x[degrees/sec], Spin y[degrees/sec], Spin z[degrees/sec]
        """
        try:
            data = pd.read_csv(file_path)

            required_columns = ['Time', 'Accel x[m/s^2]', 'Accel y[m/s^2]', 'Accel z[m/s^2]',
                                'Spin x[degrees/sec]', 'Spin y[degrees/sec]', 'Spin z[degrees/sec]']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"File must contain {required_columns} columns.")

            # Convert time to seconds based on specific format in the file
            data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S:%f').dt.hour * 3600 + \
                           pd.to_datetime(data['Time'], format='%H:%M:%S:%f').dt.minute * 60 + \
                           pd.to_datetime(data['Time'], format='%H:%M:%S:%f').dt.second + \
                           pd.to_datetime(data['Time'], format='%H:%M:%S:%f').dt.microsecond / 1e6

            self.acceleration_data = data[['Time', 'Accel x[m/s^2]', 'Accel y[m/s^2]', 'Accel z[m/s^2]',
                                           'Spin x[degrees/sec]', 'Spin y[degrees/sec]', 'Spin z[degrees/sec]']].copy()

            self.acceleration_data.rename(columns={
                'Accel x[m/s^2]': 'ax',
                'Accel y[m/s^2]': 'ay',
                'Accel z[m/s^2]': 'az',
                'Spin x[degrees/sec]': 'wx',
                'Spin y[degrees/sec]': 'wy',
                'Spin z[degrees/sec]': 'wz'
            }, inplace=True)

            self.time_intervals = np.diff(self.acceleration_data['Time'].values, prepend=0)
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    def calculate_trajectory(self):
        """
        Calculate the trajectory and orientation based on acceleration and angular velocity data.
        """
        if self.acceleration_data is None:
            raise RuntimeError("Acceleration data not loaded. Use load_data() first.")

        velocities = [self.initial_velocity]
        positions = [np.zeros(3)]
        orientations = [self.initial_orientation]

        for i in range(1, len(self.acceleration_data)):
            dt = self.time_intervals[i]
            accel = self.acceleration_data.iloc[i][['ax', 'ay', 'az']].values
            angular_vel = np.radians(self.acceleration_data.iloc[i][['wx', 'wy', 'wz']].values)  # Convert degrees/sec to radians/sec

            # Update orientation
            new_orientation = orientations[-1] + angular_vel * dt
            orientations.append(new_orientation)

            # Update velocity (assuming acceleration is in the local frame)
            new_velocity = velocities[-1] + accel * dt
            velocities.append(new_velocity)

            # Update position
            new_position = positions[-1] + new_velocity * dt
            positions.append(new_position)

        self.positions = np.array(positions)
        self.orientations = np.array(orientations)

    def normalize_time_intervals(self):
        """
        Normalize time intervals to ensure consistent integration.
        """
        if self.time_intervals is not None:
            self.time_intervals[self.time_intervals < 0] = 0  # Remove negative intervals

    def save_trajectory(self, output_file):
        """
        Save the calculated trajectory and orientations to a file.

        :param output_file: Path to the output file (CSV format)
        """
        if self.positions is None or self.orientations is None:
            raise RuntimeError("Trajectory or orientations not calculated. Use calculate_trajectory() first.")

        trajectory_df = pd.DataFrame(self.positions, columns=['x', 'y', 'z'])
        orientation_df = pd.DataFrame(self.orientations, columns=['roll', 'pitch', 'yaw'])
        result_df = pd.concat([trajectory_df, orientation_df], axis=1)
        result_df.to_csv(output_file, index=False)
        print(f"Trajectory and orientations saved to {output_file}")

    def validate_circular_motion(self, radius_tolerance=0.1):
        """
        Validate if the calculated trajectory approximates a circular motion.

        :param radius_tolerance: Allowed deviation from the mean radius
        """
        if self.positions is None:
            raise RuntimeError("Trajectory not calculated. Use calculate_trajectory() first.")

        # Compute distances from origin
        distances = np.sqrt(self.positions[:, 0]**2 + self.positions[:, 1]**2)
        mean_radius = np.mean(distances)
        max_deviation = np.max(np.abs(distances - mean_radius))

        # Analyze deviation patterns to identify inconsistencies
        deviation_indices = np.where(np.abs(distances - mean_radius) > radius_tolerance)[0]
        if len(deviation_indices) > 0:
            print("Inconsistent points:")
            for idx in deviation_indices:
                print(f"Index: {idx}, Deviation: {distances[idx] - mean_radius:.3f} m")

        is_circular = max_deviation <= radius_tolerance

        print(f"Mean Radius: {mean_radius:.2f} m")
        print(f"Max Deviation: {max_deviation:.2f} m")
        print(f"Circular Motion Validation: {'PASSED' if is_circular else 'FAILED'}")

        return is_circular

    def plot_trajectory(self):
        """
        Plot the trajectory to visually verify the motion.
        """
        if self.positions is None:
            raise RuntimeError("Trajectory not calculated. Use calculate_trajectory() first.")

        plt.figure(figsize=(6, 6))
        plt.plot(self.positions[:, 0], self.positions[:, 1], label="Trajectory")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Calculated Trajectory")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Example usage
    file_path = "corrected_circular_motion.csv"  # Replace with your file path
    output_file = "trajectory_output.csv"  # Replace with your desired output path

    analyzer = TrajectoryAnalyzer(initial_velocity=(0.0, 0.0, 0.0), initial_orientation=(0.0, 0.0, 0.0))
    analyzer.load_data(file_path)
    analyzer.normalize_time_intervals()  # Normalize time intervals
    analyzer.calculate_trajectory()
    analyzer.save_trajectory(output_file)
    analyzer.validate_circular_motion(radius_tolerance=0.1)
    analyzer.plot_trajectory()
