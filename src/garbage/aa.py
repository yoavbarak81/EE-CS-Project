import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
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

    def calculate_trajectory(self):
        if self.acceleration_data is None:
            raise RuntimeError("Acceleration data not loaded. Use load_data() first.")

        dt = np.mean(np.diff(self.acceleration_data['Time'].values))  # Average time step

        accel = self.acceleration_data[['ax', 'ay', 'az']].values
        spin = self.acceleration_data[['wx', 'wy', 'wz']].values * np.pi / 180  # Convert to radians/sec

        position = np.zeros((len(accel), 3))
        velocity = np.zeros((len(accel), 3))
        orientation = np.array(self.initial_orientation)

        for i in range(1, len(accel)):
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

            # Save orientation for debugging/analysis
            self.orientations.append(orientation.copy())

        self.positions = position

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


def rotation_matrix(roll, pitch, yaw):
    """
    Calculate the rotation matrix based on roll, pitch, and yaw (in radians).
    """
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
file_path = 'ACC_2024-12-22_12_53_17.csv'  # Replace with the path to your file
output_file = "trajectory_output.csv"  # Replace with your desired output path

analyzer = TrajectoryAnalyzer(initial_velocity=(0.0, 0.0, 0.0), initial_orientation=(0.0, 0.0, 0.0))
analyzer.load_data(file_path)
analyzer.normalize_time_intervals()  # Normalize time intervals
analyzer.calculate_trajectory()
analyzer.save_trajectory(output_file)
analyzer.plot_trajectory()

####################################################################
import numpy as np
import pandas as pd
import folium
from folium.plugins import AntPath
import matplotlib.pyplot as plt
from pyproj import Geod

class GeoTrajectoryPlotter:
    def __init__(self, initial_coordinates):
        """
        Initialize the GeoTrajectoryPlotter with initial GPS coordinates.

        :param initial_coordinates: Tuple (latitude, longitude) in degrees
        """
        self.initial_coordinates = initial_coordinates
        self.trajectory_data = None

    def load_trajectory(self, file_path):
        """
        Load trajectory data from a CSV file.

        :param file_path: Path to the CSV file containing trajectory data
        The file must have columns: Time, x, y, z
        """
        try:
            self.trajectory_data = pd.read_csv(file_path)

            required_columns = ['Time', 'x', 'y', 'z']
            if not all(col in self.trajectory_data.columns for col in required_columns):
                raise ValueError(f"File must contain {required_columns} columns.")

            # Extract only the time part (HH:MM:SS.ffffff) from the full datetime string
            self.trajectory_data['Time'] = self.trajectory_data['Time'].str.extract(r'(\d{2}:\d{2}:\d{2}\.\d{6})')[0]

            # Convert to proper datetime format
            self.trajectory_data['Time'] = pd.to_datetime(self.trajectory_data['Time'], format='%H:%M:%S.%f')

        except Exception as e:
            raise RuntimeError(f"Failed to load trajectory data: {e}")

    def convert_to_geo_coordinates(self):
        """
        Convert the trajectory positions (x, y) to geographical coordinates.
        """
        if self.trajectory_data is None:
            raise RuntimeError("Trajectory data not loaded. Use load_trajectory() first.")

        geod = Geod(ellps="WGS84")  # Geodetic model for Earth distances
        latitudes = []
        longitudes = []

        # Initial trajectory starting point
        lat0, lon0 = self.initial_coordinates
        latitudes.append(lat0)
        longitudes.append(lon0)

        for i in range(1, len(self.trajectory_data)):
            dx = self.trajectory_data.loc[i, 'x']
            dy = self.trajectory_data.loc[i, 'y']

            # Compute new geographic coordinates based on displacement (meters)
            lon_new, lat_new, _ = geod.fwd(lon0, lat0, np.arctan2(dy, dx) * 180 / np.pi, np.sqrt(dx**2 + dy**2))

            latitudes.append(lat_new)
            longitudes.append(lon_new)

        self.trajectory_data['Latitude'] = latitudes
        self.trajectory_data['Longitude'] = longitudes

    def plot_on_map(self, output_html="trajectory_map.html", highlight_times=None):
        """
        Plot the trajectory on an interactive map and save as an HTML file.

        :param output_html: Output file to save the map
        :param highlight_times: List of timestamps (HH:MM:SS) to highlight
        """
        if self.trajectory_data is None:
            raise RuntimeError("Trajectory data not loaded. Use load_trajectory() first.")

        # Create map centered at initial location
        lat0, lon0 = self.initial_coordinates
        map_ = folium.Map(location=[lat0, lon0], zoom_start=15)

        # Create trajectory path
        path = list(zip(self.trajectory_data['Latitude'], self.trajectory_data['Longitude']))
        AntPath(path, color="blue", weight=5, delay=1000).add_to(map_)

        # Add start point
        folium.Marker(location=[lat0, lon0], popup="Start", icon=folium.Icon(color="green")).add_to(map_)

        # Add end point
        folium.Marker(location=path[-1], popup="End", icon=folium.Icon(color="red")).add_to(map_)

        # Add highlighted time points
        if highlight_times:
            highlight_df = self.trajectory_data[self.trajectory_data['Time'].dt.strftime('%H:%M:%S').isin(highlight_times)]
            for _, row in highlight_df.iterrows():
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=f"Time: {row['Time'].strftime('%H:%M:%S')}",
                    icon=folium.Icon(color="purple")
                ).add_to(map_)

        # Save the map as an HTML file
        map_.save(output_html)
        print(f"Map saved to {output_html}")

    def plot_trajectory_on_2D_map(self, highlight_times=None):
        """
        Plot the trajectory on a simple 2D map using Matplotlib.

        :param highlight_times: List of timestamps (HH:MM:SS) to highlight
        """
        if self.trajectory_data is None:
            raise RuntimeError("Trajectory data not loaded. Use load_trajectory() first.")

        plt.figure(figsize=(8, 6))
        plt.plot(self.trajectory_data['Longitude'], self.trajectory_data['Latitude'], marker='o', color='b', linestyle='-')

        # Mark start and end points
        plt.scatter(self.trajectory_data['Longitude'].iloc[0], self.trajectory_data['Latitude'].iloc[0], color='green', s=100, label="Start")
        plt.scatter(self.trajectory_data['Longitude'].iloc[-1], self.trajectory_data['Latitude'].iloc[-1], color='red', s=100, label="End")

        # Highlight specific time points
        if highlight_times:
            highlight_df = self.trajectory_data[self.trajectory_data['Time'].dt.strftime('%H:%M:%S').isin(highlight_times)]
            plt.scatter(highlight_df['Longitude'], highlight_df['Latitude'], color='purple', s=100, label="Highlighted Times")

            # Add time labels to highlighted points
            for _, row in highlight_df.iterrows():
                plt.text(row['Longitude'], row['Latitude'], row['Time'].strftime('%H:%M:%S'), fontsize=9, color='black')

        # Labels and grid
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("2D Map of Trajectory")
        plt.legend()
        plt.grid()
        plt.show()


# Example Usage
initial_coords = (32.0853, 34.7818)  # Tel Aviv
trajectory_file = "trajectory_output.csv"  # Your trajectory file
highlight_times = ["12:53:26"]  # Example timestamps to highlight

plotter = GeoTrajectoryPlotter(initial_coords)
plotter.load_trajectory(trajectory_file)
plotter.convert_to_geo_coordinates()
plotter.plot_on_map("trajectory_map.html", highlight_times)  # Create an interactive map
plotter.plot_trajectory_on_2D_map(highlight_times)  # Create a 2D trajectory plot
