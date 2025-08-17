import * as state from "./state.js";
import { displayUploadedFiles, updateFileInfo } from "./ui.js";

export function saveNotes() {
  localStorage.setItem("trajectoryNotes", JSON.stringify(state.notes));
}

export function loadNotes() {
  const savedNotes = localStorage.getItem("trajectoryNotes");
  if (savedNotes) {
    state.setNotes(JSON.parse(savedNotes));
  }
}

export function handleFileUpload(file) {
  if (
    !file.name.toLowerCase().endsWith(".csv") &&
    !file.name.toLowerCase().endsWith(".txt")
  ) {
    alert("Please upload a CSV or TXT file.");
    return;
  }

  const reader = new FileReader();
  reader.onload = function (e) {
    const content = e.target.result;
    const fileData = {
      id: Date.now().toString(),
      name: file.name,
      size: file.size,
      uploadDate: new Date().toLocaleString(),
      content: content,
    };

    saveFileToRoads(fileData);
    state.setCurrentFile(fileData);
    updateFileInfo(fileData);
    displayUploadedFiles();

    alert(`File "${file.name}" uploaded successfully to roads folder!`);
  };

  reader.onerror = function () {
    alert("Error reading file. Please try again.");
  };

  reader.readAsText(file);
}

export function saveFileToRoads(fileData, force = false) {
  const existingFiles = JSON.parse(localStorage.getItem("roadsFiles") || "[]");
  const existingIndex = existingFiles.findIndex(
    (f) => f.name === fileData.name
  );

  if (existingIndex !== -1) {
    if (
      force ||
      confirm(`File "${fileData.name}" already exists. Replace it?`)
    ) {
      existingFiles[existingIndex] = fileData;
    } else {
      return;
    }
  } else {
    existingFiles.push(fileData);
  }

  localStorage.setItem("roadsFiles", JSON.stringify(existingFiles));
  state.setUploadedFiles(existingFiles);
}

export function loadUploadedFiles() {
  const files = JSON.parse(localStorage.getItem("roadsFiles") || "[]");
  state.setUploadedFiles(files);
  displayUploadedFiles();
}

export function loadFile(fileId) {
  const file = state.uploadedFiles.find((f) => f.id === fileId);
  if (file) {
    state.setCurrentFile(file);
    updateFileInfo(file);
    document.getElementById(
      "clickedCoords"
    ).innerHTML = `<strong>Loaded File:</strong> ${file.name}`;
    alert(`File "${file.name}" loaded and ready for plotting!`);
  }
}

export function deleteFile(fileId) {
  if (confirm("Are you sure you want to delete this file?")) {
    const newFiles = state.uploadedFiles.filter((f) => f.id !== fileId);
    state.setUploadedFiles(newFiles);
    localStorage.setItem("roadsFiles", JSON.stringify(newFiles));

    if (state.currentFile && state.currentFile.id === fileId) {
      state.setCurrentFile(null);
      document.getElementById("fileName").textContent = "No file selected";
      document.getElementById("fileSize").textContent = "";
    }

    displayUploadedFiles();
  }
}

export function clearAllFiles() {
  state.setUploadedFiles([]);
  localStorage.removeItem("roadsFiles");
  state.setCurrentFile(null);
  document.getElementById("fileName").textContent = "No file selected";
  document.getElementById("fileSize").textContent = "";
  displayUploadedFiles();
  alert("All files cleared from roads folder!");
}
