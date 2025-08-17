export let map;
export let markers = [];
export let trajectoryData = [];
export let trajectoryPath = null;
export let startMarker = null;
export let endMarker = null;
export let startPoint = null;
export let uploadedFiles = [];
export let currentFile = null;
export let trajectoryPoints = [];
export let notes = {};

export function setMap(newMap) {
  map = newMap;
}

export function setMarkers(newMarkers) {
  markers = newMarkers;
}

export function setTrajectoryData(newTrajectoryData) {
  trajectoryData = newTrajectoryData;
}

export function setTrajectoryPath(newTrajectoryPath) {
  trajectoryPath = newTrajectoryPath;
}

export function setStartMarker(newStartMarker) {
  startMarker = newStartMarker;
}

export function setEndMarker(newEndMarker) {
  endMarker = newEndMarker;
}

export function setStartPoint(newStartPoint) {
  startPoint = newStartPoint;
}

export function setUploadedFiles(newUploadedFiles) {
  uploadedFiles = newUploadedFiles;
}

export function setCurrentFile(newCurrentFile) {
  currentFile = newCurrentFile;
}

export function setTrajectoryPoints(newTrajectoryPoints) {
  trajectoryPoints = newTrajectoryPoints;
}

export function setNotes(newNotes) {
  notes = newNotes;
}
