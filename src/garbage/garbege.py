import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TrajectoryAnalyzer:
    def __init__(self, initial_velocity=(0.0, 0.0, 0.0), initial_orientation=(0.0, 0.0, 0.0)):
        """
        Initialize the TrajectoryAnalyzer with an initial velocity and orientation.
        """
        self.initial_velocity = np.array(initial_velocity)
        self.initial_orientation = np.array(initial_orientation)
        self.acceleration_data = None
        self.time_intervals = None
        self.positions = None
        self.orientations = [self.initial_orientation]

    def load_data(self, file_path):
        """
        Load acceleration data from a CSV file.

        :param file_path: Path to the file containing acceleration data
        """
        try:
            data = pd.read_csv(file_path)

            # Verify that the necessary columns exist
            required_columns = ['linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"File must contain {required_columns} columns.")

            # Simulate time intervals assuming a constant sampling rate of 100ms (0.1s)
            data['Time'] = np.arange(0, len(data) * 0.1, 0.1)

            self.acceleration_data = data[['linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z']].copy()
            self.acceleration_data.rename(columns={
                'linear_acceleration.x': 'ax',
                'linear_acceleration.y': 'ay',
                'linear_acceleration.z': 'az'
            }, inplace=True)

            # Calculate time intervals based on the sampling rate (100ms)
            self.time_intervals = np.diff(data['Time'].values, prepend=0)
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    def calculate_trajectory(self):
        """
        Calculate the trajectory based on acceleration data.
        """
        if self.acceleration_data is None:
            raise RuntimeError("Acceleration data not loaded. Use load_data() first.")

        velocities = [self.initial_velocity]
        positions = [np.zeros(3)]

        for i in range(1, len(self.acceleration_data)):
            dt = self.time_intervals[i]
            accel = self.acceleration_data.iloc[i][['ax', 'ay', 'az']].values

            # Update velocity based on acceleration
            new_velocity = velocities[-1] + accel * dt
            velocities.append(new_velocity)

            # Update position based on velocity
            new_position = positions[-1] + new_velocity * dt
            positions.append(new_position)

        self.positions = np.array(positions)

    def save_trajectory(self, output_file):
        """
        Save the calculated trajectory to a file.
        """
        if self.positions is None:
            raise RuntimeError("Trajectory not calculated. Use calculate_trajectory() first.")

        trajectory_df = pd.DataFrame(self.positions, columns=['x', 'y', 'z'])
        trajectory_df.to_csv(output_file, index=False)
        print(f"Trajectory saved to {output_file}")

    def plot_trajectory(self):
        """
        Plot the trajectory to visually verify the motion.
        """
        if self.positions is None:
            raise RuntimeError("Trajectory not calculated. Use calculate_trajectory() first.")

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2], label="Trajectory")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")
        ax.set_title("Calculated Trajectory")
        ax.legend()
        plt.show()

if __name__ == "__main__":
    # Example usage
    file_path = "HIMU-2024-12-24_15-29-17.csv"  # Replace with your file path
    output_file = "trajectory_output.csv"  # Replace with your desired output path

    analyzer = TrajectoryAnalyzer(initial_velocity=(0.0, 0.0, 0.0), initial_orientation=(0.0, 0.0, 0.0))
    analyzer.load_data(file_path)
    analyzer.calculate_trajectory()
    analyzer.save_trajectory(output_file)
    analyzer.plot_trajectory()


'''
Developer: Sumit Gadhiya
Date: 28.07.2024
Topic: 3D Trajectory Estimate using Dead Reckoning Algorithm

This script provides an algorithm for estimating the true 3D trajectory in space from IMU data using the Dead Reckoning(DR) Algorithm method.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load the data from .csv file
file_path = r"C:\Users\spptl\OneDrive\Desktop\Algoritham\q000000b6.csv"
data = pd.read_csv (file_path, delimiter= ';')
#print(data.head())


### Dead Reckoning- Simple Extrapolation of Movement from raw IMU

# Assuming a constant sampling rate
sampling_rate = 25  # Hz
dt = 1 / sampling_rate  # Time interval

# Integrate acceleration to get velocity
velocity_x = np.cumsum(accel_x) * dt
velocity_y = np.cumsum(accel_y) * dt
velocity_z = np.cumsum(accel_z) * dt

# Integrate velocity to get position
position_x = np.cumsum(velocity_x) * dt
position_y = np.cumsum(velocity_y) * dt
position_z = np.cumsum(velocity_z) * dt

# Plot Velocity Data
plt.figure(figsize=(12, 8))

# Plot Velocity X Axis
plt.subplot(3, 1, 1)
plt.plot(velocity_x, label='Velocity X')
plt.xlabel('Sample Index')
plt.ylabel('Velocity (m/s)')
plt.legend()

# Plot Velocity Y Axis
plt.subplot(3, 1, 2)
plt.plot(velocity_y, label='Velocity Y', color='orange')
plt.xlabel('Sample Index')
plt.ylabel('Velocity (m/s)')
plt.legend()

# Plot Velocity Z Axis
plt.subplot(3, 1, 3)
plt.plot(velocity_z, label='Velocity Z', color='green')
plt.xlabel('Sample Index')
plt.ylabel('Velocity (m/s)')
plt.legend()

plt.tight_layout()
plt.show()


# Plot the estimated positions
plt.figure(figsize=(12, 8))

# Plot Position X Axis
plt.subplot(3, 1, 1)
plt.plot(position_x, label='Position X')
plt.xlabel('Sample Index')
plt.ylabel('Position (m)')
plt.legend()

# Plot Position Y Axis
plt.subplot(3, 1, 2)
plt.plot(position_y, label='Position Y', color='orange')
plt.xlabel('Sample Index')
plt.ylabel('Position (m)')
plt.legend()

# Plot Position Z Axis
plt.subplot(3, 1, 3)
plt.plot(position_z, label='Position Z', color='green')
plt.xlabel('Sample Index')
plt.ylabel('Position (m)')
plt.legend()

plt.tight_layout()
plt.show()

# Visualization of 3D Trajectory

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(position_x, position_y, position_z, label='3D Trajectory')
ax.set_xlabel('Position X (m)')
ax.set_ylabel('Position Y (m)')
ax.set_zlabel('Position Z (m)')
ax.legend()

plt.title('3D Trajectory using Dead reckoning')
plt.show()









