import numpy as np
import pandas as pd
import folium
from folium.plugins import AntPath
import matplotlib.pyplot as plt
from pyproj import Geod
from branca.element import Template, MacroElement

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
                    popup=f"Time: {row['Time'].strftime('%H:%M:%S')}<br>({row['Latitude']:.6f}, {row['Longitude']:.6f})",
                    icon=folium.Icon(color="purple")
                ).add_to(map_)

        # JavaScript for displaying clicked coordinates
        click_js = """
            function onMapClick(e) {
                alert("Coordinates: " + e.latlng.lat + ", " + e.latlng.lng);
            }
            map.on('click', onMapClick);
        """
        map_.get_root().html.add_child(folium.Element(f'<script>{click_js}</script>'))

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
