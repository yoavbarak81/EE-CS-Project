import * as state from "./state.js";
import { formatFileSize } from "./utils.js";
import { loadFile, deleteFile, saveNotes } from "./fileManager.js";

let currentTimestamp = null;
let currentImage = null;

export function showNoteEditor(timestamp) {
  currentTimestamp = timestamp;
  currentImage = null;

  const modal = document.getElementById("noteEditor");
  const timestampEl = document.getElementById("noteTimestamp");
  const noteText = document.getElementById("noteText");
  const imagePreview = document.getElementById("noteImagePreview");
  const removeImageButton = document.getElementById("removeNoteImageButton");

  timestampEl.textContent = timestamp;
  const note = state.notes[timestamp] || {};
  noteText.value = note.text || "";

  if (note.image) {
    imagePreview.src = note.image;
    imagePreview.style.display = "block";
    removeImageButton.style.display = "block";
    currentImage = note.image;
  } else {
    imagePreview.src = "";
    imagePreview.style.display = "none";
    removeImageButton.style.display = "none";
  }

  modal.style.display = "block";
}

export function hideNoteEditor() {
  const modal = document.getElementById("noteEditor");
  modal.style.display = "none";
}

function saveCurrentNote() {
  const noteText = document.getElementById("noteText").value;
  const existingNote = state.notes[currentTimestamp] || {};

  const newNote = {
    text: noteText.trim(),
    image: currentImage || existingNote.image,
  };

  if (newNote.text || newNote.image) {
    state.notes[currentTimestamp] = newNote;
  } else {
    delete state.notes[currentTimestamp];
  }

  saveNotes();
  updateMarkerAppearance(currentTimestamp);
  hideNoteEditor();
}

function deleteCurrentNote() {
  delete state.notes[currentTimestamp];
  saveNotes();
  updateMarkerAppearance(currentTimestamp);
  hideNoteEditor();
}

function handleImageSelection(event) {
  const file = event.target.files[0];
  if (file && file.type.startsWith("image/")) {
    const reader = new FileReader();
    reader.onload = (e) => {
      currentImage = e.target.result;
      const imagePreview = document.getElementById("noteImagePreview");
      imagePreview.src = currentImage;
      imagePreview.style.display = "block";
      document.getElementById("removeNoteImageButton").style.display = "block";
    };
    reader.readAsDataURL(file);
  }
}

function removeSelectedImage() {
  currentImage = null;
  const imagePreview = document.getElementById("noteImagePreview");
  const imageInput = document.getElementById("noteImageInput");
  const removeImageButton = document.getElementById("removeNoteImageButton");

  imagePreview.src = "";
  imagePreview.style.display = "none";
  imageInput.value = ""; // Reset file input
  removeImageButton.style.display = "none";

  // If there was a saved image, we need to reflect its removal on save
  const note = state.notes[currentTimestamp];
  if (note) {
    delete note.image;
  }
}

function updateMarkerAppearance(timestamp) {
  const pointIndex = state.trajectoryData.findIndex(
    (p) => p.timestamp === timestamp
  );
  if (pointIndex !== -1) {
    const pointMarker = state.trajectoryPoints[pointIndex];
    if (pointMarker) {
      const note = state.notes[timestamp];
      const hasNote = note && (note.text || note.image);
      pointMarker.setStyle({
        fillColor: hasNote ? "#f1c40f" : "#00b894",
      });
      const popupContent = createPopupContent(note);
      pointMarker.bindPopup(popupContent);
    }
  }
}

export function setupUIEventListeners() {
  document
    .querySelector("#noteEditor .close-button")
    .addEventListener("click", hideNoteEditor);
  document
    .getElementById("saveNoteButton")
    .addEventListener("click", saveCurrentNote);
  document
    .getElementById("deleteNoteButton")
    .addEventListener("click", deleteCurrentNote);
  document
    .getElementById("selectImageButton")
    .addEventListener("click", () =>
      document.getElementById("noteImageInput").click()
    );
  document
    .getElementById("noteImageInput")
    .addEventListener("change", handleImageSelection);
  document
    .getElementById("removeNoteImageButton")
    .addEventListener("click", removeSelectedImage);

  window.addEventListener("click", (event) => {
    const modal = document.getElementById("noteEditor");
    if (event.target == modal) {
      hideNoteEditor();
    }
  });
}

export function createPopupContent(note) {
  if (!note) return "No note for this point.";

  let content = "";
  if (note.text) {
    content += `<p>${note.text}</p>`;
  }
  if (note.image) {
    content += `<img src="${note.image}" alt="Note image" style="max-width:150px; max-height:150px; margin-top:5px;">`;
  }
  return content || "Note saved (no text or image).";
}

export function updateFileInfo(fileData) {
  document.getElementById("fileName").textContent = fileData.name;
  document.getElementById("fileSize").textContent = formatFileSize(
    fileData.size
  );
}

export function displayUploadedFiles() {
  const filesList = document.getElementById("filesList");

  if (state.uploadedFiles.length === 0) {
    filesList.innerHTML = "<p class='no-files'>No files uploaded yet</p>";
    return;
  }

  filesList.innerHTML = state.uploadedFiles
    .map(
      (file) => `
    <div class="file-item" id="file-item-${file.id}">
      <div class="file-details">
        <div class="file-name">${file.name}</div>
        <div class="file-meta">
          ${formatFileSize(file.size)} • Uploaded: ${file.uploadDate}
        </div>
      </div>
      <div class="file-actions">
        <button class="load-btn" data-file-id="${file.id}">Load</button>
        <button class="delete-btn" data-file-id="${file.id}">Delete</button>
      </div>
    </div>
  `
    )
    .join("");

  // Add event listeners to the new buttons
  document.querySelectorAll(".load-btn").forEach((button) => {
    button.addEventListener("click", (e) => {
      loadFile(e.target.dataset.fileId);
    });
  });

  document.querySelectorAll(".delete-btn").forEach((button) => {
    button.addEventListener("click", (e) => {
      deleteFile(e.target.dataset.fileId);
    });
  });
}

export function showTrajectoryStats() {
  try {
    if (!state.trajectoryData || state.trajectoryData.length === 0) {
      document.getElementById("trajectoryInfo").style.display = "none";
      return;
    }

    const pointCount = state.trajectoryData.length;
    const startData = state.trajectoryData[0];
    const endData = state.trajectoryData[pointCount - 1];

    if (!startData || !endData) {
      console.error(
        "Trajectory data contains invalid entries.",
        state.trajectoryData
      );
      alert("Error in trajectory data. Could not generate stats.");
      return;
    }

    let totalDistance = 0;
    for (let i = 1; i < state.trajectoryData.length; i++) {
      const prev = state.trajectoryData[i - 1];
      const curr = state.trajectoryData[i];
      const dx = curr.x - prev.x;
      const dy = curr.y - prev.y;
      totalDistance += Math.sqrt(dx * dx + dy * dy);
    }

    const displacement = Math.sqrt(
      Math.pow(endData.x - startData.x, 2) +
        Math.pow(endData.y - startData.y, 2)
    );

    const statsHTML = `
      <strong>File:</strong> ${state.currentFile.name}<br>
      <strong>Data Points:</strong> ${pointCount}<br>
      <strong>Start Time:</strong> ${startData.timestamp}<br>
      <strong>End Time:</strong> ${endData.timestamp}<br>
      <strong>Total Distance:</strong> ${totalDistance.toFixed(3)} meters<br>
      <strong>Displacement:</strong> ${displacement.toFixed(3)} meters<br>
      <strong>Start Position (x,y):</strong> (${startData.x.toFixed(
        3
      )}, ${startData.y.toFixed(3)}) meters<br>
      <strong>End Position (x,y):</strong> (${endData.x.toFixed(
        3
      )}, ${endData.y.toFixed(3)}) meters<br>
      <strong>Final Roll/Pitch/Yaw:</strong> (${endData.roll.toFixed(
        3
      )}, ${endData.pitch.toFixed(3)}, ${endData.yaw.toFixed(3)})
    `;

    document.getElementById("trajectoryStats").innerHTML = statsHTML;
    document.getElementById("trajectoryInfo").style.display = "block";
  } catch (error) {
    console.error("Error in showTrajectoryStats:", error);
    alert(`Error generating stats: ${error.message}`);
  }
}
