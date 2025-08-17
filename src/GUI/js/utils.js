export function parseCSV(csvData) {
  if (!csvData || typeof csvData !== "string") {
    return [];
  }
  const lines = csvData.trim().split("\n");
  const data = [];

  // Determine if the first line is a header (if it contains any alphabetic characters)
  const probablyHasHeader = lines[0] && /[a-zA-Z]/.test(lines[0]);
  const startLine = probablyHasHeader ? 1 : 0;

  for (let i = startLine; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line === "") continue;

    const values = line.split(",");

    // Expect at least x, y in columns 1, 2 (0-indexed)
    if (values.length < 3) continue;

    const x = parseFloat(values[1]);
    const y = parseFloat(values[2]);

    if (isNaN(x) || isNaN(y) || x === 0 || y === 0) {
      continue;
    }

    const row = {
      timestamp: values[0] || `Row ${i + 1}`,
      x: x,
      y: y,
      z: values[3] ? parseFloat(values[3]) : 0,
      roll: values[4] ? parseFloat(values[4]) : 0,
      pitch: values[5] ? parseFloat(values[5]) : 0,
      yaw: values[6] ? parseFloat(values[6]) : 0,
    };

    // Final check for NaNs in optional fields
    row.z = isNaN(row.z) ? 0 : row.z;
    row.roll = isNaN(row.roll) ? 0 : row.roll;
    row.pitch = isNaN(row.pitch) ? 0 : row.pitch;
    row.yaw = isNaN(row.yaw) ? 0 : row.yaw;

    data.push(row);
  }

  if (lines.length > 0 && data.length === 0 && startLine < lines.length) {
    alert(
      "Could not parse any valid data points from the CSV. Please check the file format and ensure x/y values are non-zero."
    );
  }

  return data;
}

export function metersToLatLng(startLat, startLng, meterX, meterY) {
  // Approximate conversion: 1 degree latitude ≈ 111,111 meters
  // 1 degree longitude ≈ 111,111 * cos(latitude) meters
  const latDegreeInMeters = 111111;
  const lngDegreeInMeters = 111111 * Math.cos((startLat * Math.PI) / 180);

  const deltaLat = meterY / latDegreeInMeters;
  const deltaLng = meterX / lngDegreeInMeters;

  return {
    lat: startLat + deltaLat,
    lng: startLng + deltaLng,
  };
}

export function isValidCoordinate(lat, lng) {
  return (
    !isNaN(lat) &&
    !isNaN(lng) &&
    lat >= -90 &&
    lat <= 90 &&
    lng >= -180 &&
    lng <= 180
  );
}

export function formatFileSize(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

export function getLocationInfo(lat, lng) {
  // Simple location detection based on coordinates
  if (lat >= 24 && lat <= 49 && lng >= -125 && lng <= -66) {
    return "United States";
  } else if (lat >= 45 && lat <= 71 && lng >= -141 && lng <= -52) {
    return "Canada";
  } else if (lat >= 35 && lat <= 71 && lng >= -10 && lng <= 40) {
    return "Europe";
  } else if (lat >= -35 && lat <= 37 && lng >= -18 && lng <= 51) {
    return "Africa";
  } else if (lat >= -50 && lat <= 81 && lng >= 26 && lng <= 180) {
    return "Asia";
  } else if (lat >= -47 && lat <= -10 && lng >= 113 && lng <= 154) {
    return "Australia";
  } else if (lat >= -56 && lat <= 13 && lng >= -82 && lng <= -35) {
    return "South America";
  } else if (lat > 66) {
    return "Arctic Region";
  } else if (lat < -66) {
    return "Antarctica";
  } else if (lng >= -180 && lng <= -30 && lat >= 15 && lat <= 72) {
    return "North America";
  } else {
    return "Ocean or Remote Area";
  }
}
