# 🏸 G1432 — Badminton AI: Footwork Detection & Movement Analysis

An AI-powered computer vision project for analyzing badminton player movement and identifying footwork patterns on a singles badminton court using **YOLOv8** and video-based movement analysis.

## 🚀 Project Overview

Badminton requires fast and accurate movement across different areas of the court. This project aims to use computer vision to automatically detect badminton players from video footage, track their movement, analyze their movement patterns, and classify their position into one of **six court zones/corners**.

The system processes badminton video footage and converts player movement into meaningful court-position information.

## 🎯 Objectives

The main objectives of this project are:

- Detect badminton players from video frames.
- Track player movement across the court.
- Analyze movement patterns during gameplay.
- Divide the singles court into six movement zones.
- Classify the player's position based on detected location.
- Generate movement-related data for further performance analysis.

## 🧠 System Architecture

```text
Badminton Video
       ↓
Frame Extraction
       ↓
Annotated Dataset
       ↓
YOLOv8 Training
       ↓
Trained YOLOv8 Model
       ↓
New Video
       ↓
Player Detection
       ↓
Player Tracking
       ↓
Movement Analysis
       ↓
6-Corner Classification
       ↓
Performance Output