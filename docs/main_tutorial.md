# AgenticLab Human Main Tutorial

## Goals
After reading this tutorial, you should be able to:
1. Understand the code structure of AgenticLab on the H1 humanoid robot upper body.
2. Understand the purpose of the main modules.
3. Install the main modules: voice-to-text, text-to-plan, perception, GraspNet, and X5 communication dependencies.
4. Run the voice-to-text generation script.
5. Use remote camera capture to collect RGB-D images.
6. Run GraspNet in a local environment to generate grasp poses.
7. Run a simple X5 arm control script through the FastAPI framework.
8. Run a simple pick-and-place pipeline on the H1 humanoid robot.

## Main Modules
1. Voice-to-text and text-to-plan generation.
2. Planning with VLMs and Fast Downward.
3. X5 robot control and communication framework.
4. Camera perception and utilities.
5. Grasp generation and GraspNet environment setup.
6. YOLO detection and fine-tuning with new object classes.

## Main Code Architecture

## Module Documents
1. Voice-to-text and voice-to-plan generation.
2. Planning with VLMs and Fast Downward.
3. X5 robot control and communication framework: [X5/client_server.md](X5/client_server.md).
4. Camera perception and utilities.
5. Grasp generation and GraspNet environment setup: [grasp/grasp_client_server.md](grasp/grasp_client_server.md).
6. YOLO detection and fine-tuning: [perception/yolo_train.md](perception/yolo_train.md).
7. Pipeline execution: [pipeline/execution_without_planning.md](pipeline/execution_without_planning.md).

## Installation
### 1. Install the AgenticLab repository
### 2. Install voice-to-text dependencies
### 3. Install VLM calling and Fast Downward dependencies
### 4. Install camera dependencies
### 5. Install FastAPI dependencies
### 6. Install GraspNet dependencies

## Basic Operation
### 1. Run the voice-to-text generation script
### 2. Capture RGB-D images from the remote camera
### 3. Generate grasp poses with GraspNet in the local environment
### 4. Run a simple X5 arm control script through FastAPI
### 5. Run a simple pick-and-place script on X5
