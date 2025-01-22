import numpy as np
import pandas as pd
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

            # Assign indices as time units (1, 2, 3, ...)
            data['Time'] = np.arange(1, len(data) + 1)

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

            self.goto = np.diff(self.acceleration_data['Time'].values, prepend=0)

        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    # def calculate_trajectory(self):
    #     """
    #     Calculate the trajectory based on acceleration data using cumulative sum for integration.
    #     """
    #     if self.acceleration_data is None:
    #         raise RuntimeError("Acceleration data not loaded. Use load_data() first.")
    #
    #     # Extract time intervals
    #     dt = np.mean(np.diff(self.acceleration_data['Time'].values))  # Calculate average time step
    #
    #     # Extract acceleration data
    #     accel_x = self.acceleration_data['ax'].values
    #     accel_y = self.acceleration_data['ay'].values
    #     accel_z = self.acceleration_data['az'].values
    #
    #     # Integrate acceleration to get velocity
    #     velocity_x = np.cumsum(accel_x) * dt
    #     velocity_y = np.cumsum(accel_y) * dt
    #     velocity_z = np.cumsum(accel_z) * dt
    #
    #     # Integrate velocity to get position
    #     position_x = np.cumsum(velocity_x) * dt
    #     position_y = np.cumsum(velocity_y) * dt
    #     position_z = np.cumsum(velocity_z) * dt
    #
    #     # Combine positions into an array
    #     self.positions = np.vstack((position_x, position_y, position_z)).T

    def calculate_trajectory(self):
        """
        Calculate the trajectory using acceleration and angular velocity data.
        """
        if self.acceleration_data is None:
            raise RuntimeError("Acceleration data not loaded. Use load_data() first.")

        # Extract time intervals
        dt = np.mean(np.diff(self.acceleration_data['Time'].values))  # Average time step

        # Extract acceleration and angular velocity data
        accel_data = self.acceleration_data[['ax', 'ay', 'az']].values
        angular_velocity_data = np.radians(self.acceleration_data[['wx', 'wy', 'wz']].values)  # Convert to radians/sec

        # Initialize positions, velocities, and orientation
        velocities = [self.initial_velocity]
        positions = [np.zeros(3)]
        orientations = [self.initial_orientation]

        for i in range(1, len(accel_data)):
            # Calculate new orientation
            angular_vel = angular_velocity_data[i]
            new_orientation = orientations[-1] + angular_vel * dt  # Update roll, pitch, yaw
            orientations.append(new_orientation)

            # Rotate acceleration to global frame using the new orientation
            roll, pitch, yaw = new_orientation
            rotation_matrix = self.get_rotation_matrix(roll, pitch, yaw)
            accel_global = rotation_matrix @ accel_data[i]

            # Update velocity
            new_velocity = velocities[-1] + accel_global * dt
            velocities.append(new_velocity)

            # Update position
            avg_velocity = (velocities[-1] + velocities[-2]) / 2
            new_position = positions[-1] + avg_velocity * dt
            positions.append(new_position)

        self.positions = np.array(positions)
        self.orientations = np.array(orientations)

    def get_rotation_matrix(self, roll, pitch, yaw):
        """
        Compute the rotation matrix from roll, pitch, and yaw angles.
        """
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx

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

        # Add the time column to the output
        trajectory_df = pd.DataFrame(self.positions, columns=['x', 'y', 'z'])
        orientation_df = pd.DataFrame(self.orientations, columns=['roll', 'pitch', 'yaw'])
        time_df = pd.DataFrame(self.acceleration_data['Time'].values, columns=['Time'])

        # Concatenate the time, position, and orientation data
        result_df = pd.concat([time_df, trajectory_df, orientation_df], axis=1)
        result_df.to_csv(output_file, index=False)
        print(f"Trajectory and orientations saved to {output_file}")

    def plot_trajectory(self):
        """
        Plot the 3D trajectory to visually verify the motion.
        """
        if self.positions is None:
            raise RuntimeError("Trajectory not calculated. Use calculate_trajectory() first.")

        # Create a 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the 3D trajectory
        ax.plot(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2], label="Trajectory", lw=2)

        # Set labels and title
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")
        ax.set_title("3D Calculated Trajectory")

        # Add grid and legend
        ax.grid(True)
        ax.legend()

        # Show the plot
        plt.show()


# Example usage
file_path = 'square_motion_corrected.csv'  # Replace with the path to your file
output_file = "trajectory_output.csv"  # Replace with your desired output path

analyzer = TrajectoryAnalyzer(initial_velocity=(0.0, 0.0, 0.0), initial_orientation=(0.0, 0.0, 0.0))
analyzer.load_data(file_path)
analyzer.normalize_time_intervals()  # Normalize time intervals
analyzer.calculate_trajectory()
analyzer.save_trajectory(output_file)
analyzer.plot_trajectory()
