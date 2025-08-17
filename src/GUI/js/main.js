import * as state from "./state.js";
import * as map from "./map.js";
import * as ui from "./ui.js";
import * as fileManager from "./fileManager.js";
import * as utils from "./utils.js";

document.addEventListener("DOMContentLoaded", function () {
  map.initializeMap();
  setupEventListeners();
  fileManager.loadUploadedFiles();
  fileManager.loadNotes();

  const initialLat = 32.0943;
  const initialLng = 34.9551;
  document.getElementById("startLat").value = initialLat;
  document.getElementById("startLng").value = initialLng;
  map.setStartPoint(initialLat, initialLng);

  fetch("roads/output_with_angles.csv")
    .then((resp) => {
      if (!resp.ok) throw new Error("Default CSV not found");
      return resp.text();
    })
    .then((text) => {
      const fileData = {
        id: "default-csv",
        name: "output_with_angles.csv",
        size: text.length,
        uploadDate: new Date().toLocaleString(),
        content: text,
      };
      fileManager.saveFileToRoads(fileData, true);
      state.setCurrentFile(fileData);
      ui.updateFileInfo(fileData);
      ui.displayUploadedFiles();
      map.plotTrajectory(text);
      document.getElementById("clickedCoords").innerHTML =
        "<strong>Loaded default CSV:</strong> output_with_angles.csv";
    })
    .catch((err) => console.warn("Default CSV not found:", err.message));
});

function setupEventListeners() {
  ui.setupUIEventListeners();

  document
    .getElementById("setStartPoint")
    .addEventListener("click", function () {
      const lat = parseFloat(document.getElementById("startLat").value);
      const lng = parseFloat(document.getElementById("startLng").value);
      if (utils.isValidCoordinate(lat, lng)) {
        map.setStartPoint(lat, lng);
      } else {
        alert(
          "Please enter valid start point coordinates.\nLatitude: -90 to 90\nLongitude: -180 to 180"
        );
      }
    });

  document.getElementById("csvFile").addEventListener("change", function (e) {
    const file = e.target.files[0];
    if (file) {
      fileManager.handleFileUpload(file);
    }
  });

  document
    .getElementById("plotTrajectory")
    .addEventListener("click", function () {
      if (!state.currentFile) {
        alert("Please upload a CSV file first.");
        return;
      }
      if (!state.startPoint) {
        alert("Please set a start point first.");
        return;
      }
      map.plotTrajectory(state.currentFile.content);
    });

  document
    .getElementById("clearTrajectory")
    .addEventListener("click", map.clearTrajectory);
  document
    .getElementById("clearAllFiles")
    .addEventListener("click", fileManager.clearAllFiles);

  // Enter key support
  document.getElementById("startLat").addEventListener("keypress", (e) => {
    if (e.key === "Enter") document.getElementById("setStartPoint").click();
  });
  document.getElementById("startLng").addEventListener("keypress", (e) => {
    if (e.key === "Enter") document.getElementById("setStartPoint").click();
  });
}
