import * as state from "./state.js";
import * as utils from "./utils.js";
import {
  showTrajectoryStats,
  showNoteEditor,
  createPopupContent,
} from "./ui.js";

export function initializeMap() {
  const map = L.map("map").setView([32.0943, 34.9551], 13);
  state.setMap(map);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution:
      '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19,
  }).addTo(map);

  /* map.on("click", function (e) {
    const lat = e.latlng.lat.toFixed(6);
    const lng = e.latlng.lng.toFixed(6);

    document.getElementById(
      "clickedCoords"
    ).innerHTML = `<strong>Clicked Location:</strong> Latitude: ${lat}, Longitude: ${lng}`;
    addMarker(lat, lng, `Clicked Location<br>Lat: ${lat}<br>Lng: ${lng}`);

    document.getElementById("latitude").value = lat;
    document.getElementById("longitude").value = lng;

    if (!document.getElementById("startLat").value) {
      document.getElementById("startLat").value = lat;
    }
    if (!document.getElementById("startLng").value) {
      document.getElementById("startLng").value = lng;
    }
  });*/
}

export function addMarker(lat, lng, popupContent) {
  const marker = L.marker([lat, lng]).addTo(state.map);
  marker.bindPopup(popupContent);
  state.markers.push(marker);
  marker.openPopup();
}

export function showLocation(lat, lng) {
  state.map.setView([lat, lng], 10);
  const locationInfo = utils.getLocationInfo(lat, lng);
  addMarker(
    lat,
    lng,
    `<h3>Your Location</h3><strong>Coordinates:</strong><br>Latitude: ${lat}<br>Longitude: ${lng}<br><strong>Info:</strong> ${locationInfo}`
  );
  document.getElementById(
    "clickedCoords"
  ).innerHTML = `<strong>Showing Location:</strong> Latitude: ${lat}, Longitude: ${lng}`;
}

export function setStartPoint(lat, lng) {
  state.setStartPoint({ lat, lng });

  if (state.startMarker) {
    state.map.removeLayer(state.startMarker);
  }

  const newStartMarker = L.marker([lat, lng], {
    icon: L.divIcon({
      className: "start-marker",
      html: '<div style="background-color: #fdcb6e; border: 3px solid white; border-radius: 50%; width: 20px; height: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.5);"></div>',
      iconSize: [20, 20],
      iconAnchor: [10, 10],
    }),
  }).addTo(state.map);
  state.setStartMarker(newStartMarker);

  state.startMarker.bindPopup(
    `<h3>Start Point</h3><strong>Coordinates:</strong><br>Latitude: ${lat}<br>Longitude: ${lng}`
  );
  state.map.setView([lat, lng], 15);
  document.getElementById(
    "clickedCoords"
  ).innerHTML = `<strong>Start Point Set:</strong> Latitude: ${lat}, Longitude: ${lng}`;
}

export function clearAllMarkers() {
  state.markers.forEach((marker) => {
    state.map.removeLayer(marker);
  });
  state.setMarkers([]);

  if (state.startMarker) {
    state.map.removeLayer(state.startMarker);
    state.setStartMarker(null);
    state.setStartPoint(null);
  }
}

export function clearTrajectory() {
  if (state.trajectoryPath) {
    state.map.removeLayer(state.trajectoryPath);
    state.setTrajectoryPath(null);
  }
  if (state.endMarker) {
    state.map.removeLayer(state.endMarker);
    state.setEndMarker(null);
  }
  state.trajectoryPoints.forEach((point) => {
    state.map.removeLayer(point);
  });
  state.setTrajectoryPoints([]);
  document.getElementById("trajectoryInfo").style.display = "none";
  state.setTrajectoryData([]);
}

export function plotTrajectory(csvData) {
  clearTrajectory();

  const startLat = parseFloat(document.getElementById("startLat").value);
  const startLng = parseFloat(document.getElementById("startLng").value);

  if (isNaN(startLat) || isNaN(startLng)) {
    alert("Please enter valid starting latitude and longitude.");
    return;
  }

  try {
    const data = utils.parseCSV(csvData);
    state.setTrajectoryData(data);

    if (
      !state.trajectoryData ||
      !Array.isArray(state.trajectoryData) ||
      state.trajectoryData.length === 0
    ) {
      console.warn("Plotting aborted: No valid trajectory data was parsed.");
      return;
    }

    const latlngs = state.trajectoryData.map((row) => {
      return utils.metersToLatLng(startLat, startLng, row.x, row.y);
    });

    const newPath = L.polyline(latlngs, {
      color: "#00b894",
      weight: 4,
      opacity: 0.8,
      dashArray: "10, 5",
    }).addTo(state.map);
    state.setTrajectoryPath(newPath);

    const points = [];
    latlngs.forEach((latlng, index) => {
      const dataPoint = state.trajectoryData[index];
      const timestamp = dataPoint.timestamp;
      const note = state.notes[timestamp];

      const hasNote = note && (note.text || note.image);

      const point = L.circleMarker(latlng, {
        radius: 5,
        fillColor: hasNote ? "#f1c40f" : "#00b894",
        color: "#fff",
        weight: 1,
        opacity: 1,
        fillOpacity: 0.8,
      }).addTo(state.map);

      point.bindPopup(createPopupContent(note));

      point.on("mouseover", function (e) {
        this.openPopup();
      });
      point.on("mouseout", function (e) {
        this.closePopup();
      });

      point.on("click", () => {
        showNoteEditor(timestamp, latlng);
      });

      points.push(point);
    });
    state.setTrajectoryPoints(points);

    const endCoord = latlngs[latlngs.length - 1];
    const newEndMarker = L.marker(endCoord, {
      icon: L.divIcon({
        className: "end-marker",
        html: '<div style="background-color: #ff6b6b; border: 3px solid white; border-radius: 50%; width: 20px; height: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.5);"></div>',
        iconSize: [20, 20],
        iconAnchor: [10, 10],
      }),
    }).addTo(state.map);
    state.setEndMarker(newEndMarker);

    const endData = state.trajectoryData[state.trajectoryData.length - 1];
    state.endMarker.bindPopup(`<h3>End Point</h3>
                       <strong>Coordinates:</strong><br>
                       Latitude: ${endCoord.lat.toFixed(6)}<br>
                       Longitude: ${endCoord.lng.toFixed(6)}<br>
                       <strong>Final Position:</strong><br>
                       X: ${endData.x.toFixed(3)}m<br>
                       Y: ${endData.y.toFixed(3)}m<br>
                       <strong>File:</strong> ${state.currentFile.name}`);

    const bounds = L.latLngBounds(latlngs);
    state.map.fitBounds(bounds, { padding: [20, 20] });

    showTrajectoryStats();
  } catch (error) {
    alert("Error processing trajectory data: " + error.message);
  }
}
