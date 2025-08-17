# Roads Folder

This folder is designed to store CSV trajectory files that contain movement data in meters.

## File Format Expected

CSV files should have the following format:
```csv
Time,x,y,z
1900-01-01 11:56:11.312661,-0.009872917773836561,0.002258203815137040,0.0
1900-01-01 11:56:11.412971,-0.028530285613670500,0.001012073879396740,-0.001
...
```

Where:
- **Time**: Timestamp of the measurement
- **x**: X coordinate in meters
- **y**: Y coordinate in meters  
- **z**: Z coordinate in meters (optional, used for 3D data)

## Usage

1. Upload CSV files through the web interface
2. Files are automatically stored in this virtual folder structure
3. Load any previously uploaded file to plot its trajectory
4. The web application will convert meter coordinates to latitude/longitude based on your set start point

## Notes

- Files are stored in browser localStorage for persistence
- Physical copies of uploaded files are not automatically saved to this folder
- To manually save files here, copy them from your local system after upload 