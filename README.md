# EE-CS-Project
Final project for a degree in Electrical Engineering and Computer Science Noam Aburbeh and Yoav Barak

# Submarine Pipeline Trajectory Mapping System

## Project Overview
This project aims to develop an innovative solution for tracking the trajectory of a small submarine within closed pipelines in environments where traditional navigation systems like GPS are ineffective. The solution integrates Inertial Navigation Systems (INS) with optical navigation to achieve precise trajectory mapping. This effort is in collaboration with iPIPE Ltd., with the guidance of Dr. Menashe Rajuan.

## Problem Definition
Organizations managing underground or underwater pipelines, such as those in the water, oil, and gas sectors, face significant challenges in identifying cracks or leaks. Tracking the trajectory of inspection submarines in GPS-denied environments is crucial for accurate defect detection and maintenance.

## Proposed Solution
We are developing a **hybrid system** combining:
1. **Inertial Sensors (IMU)** to measure motion and orientation.
2. **Optical Navigation** using a camera to identify pipeline features for drift correction.

Key components:
- **Hardware:** High-definition cameras, IMU sensors, and an embedded processor.
- **Software:** Python for data fusion and OpenCV for image processing.
- **Processing Unit:** NVIDIA Jetson for real-time computations.

The hybrid approach ensures accuracy and adaptability without relying on electromagnetic signals or pre-installed markers.

## Evaluation Plan
The system will undergo:
1. **Real-World Testing:** Utilizing Hagihon’s pipeline infrastructure.
2. **Simulations:** Testing accuracy under various pipeline scenarios with predefined benchmarks.

Expected accuracy: ±10 cm deviation per 100 meters.

## Development Timeline
- **First 2 Months:** Hardware calibration and offline trajectory calculations.
- **January 2025:** MVP delivery—an offline prototype for mapping trajectories.
- **Future Goals:** Real-time mapping with advanced error correction (contingent on overcoming computational challenges).

## Collaborators
- **Team Members:**
  - Yoav Barak ([yoav.barak@mail.huji.ac.il](mailto:yoav.barak@mail.huji.ac.il))
  - Noam Aburbeh ([noam.aburbeh@mail.huji.ac.il](mailto:noam.aburbeh@mail.huji.ac.il))
- **Advisor:** Dr. Menashe Rajuan
- **Partner Organization:** iPIPE Ltd.
- **Industry Partner:** Hagihon ([hagihon.co.il](https://www.hagihon.co.il))

## Repository Structure
