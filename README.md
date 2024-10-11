# Chess_CV

We propose to develop a computer vision system for Project A: Board Game Tracking, as outlined in the project descriptions. The goal is to create a system capable of tracking and analyzing the moves made in a physical board game, such as Chess or Go, using a mounted webcam to capture the game in real time. The system will transcribe game moves, detect illegal moves, and identify individual game pieces robustly, even under challenging conditions like occlusions or varying camera placements. The system should display its results on a GUI to show that it is working correctly \\
      
## Methodology
### Image Acquisition \& Preprocessing
- Use a webcam to capture live video footage of the board game.
- Implement preprocessing techniques such as background subtraction and edge detection to isolate the board and pieces from the background.

### Piece Identification and Tracking
- Utilise image segmentation and object detection techniques to identify and classify different game pieces
- Apply algorithms such as Histogram of Oriented Gradients (HOG) and Convolutional Neural Networks (CNNs) for piece classification.
- Track piece movements using optical flow and Kalman filters to handle occlusions and ensure consistent tracking even when pieces are partially obscured.

### Occlusion Handling
- Develop a strategy to deal with hand occlusions by integrating temporal data. We will use a motion history image (MHI) approach to differentiate between hand movements and piece movements.

### Move Validation
- Implement a game logic system that checks for illegal moves based on the rules of the game (e.g., chess rules) and alerts the player in real-time.

## Timeline  
- (Week 9) Research and initial project setup, literature review on tracking algorithms.
- (Week 10) Implementation of image acquisition, preprocessing, and piece identification modules.
- (Week 11) Develop tracking algorithms and handle occlusions and implement move validation logic and conduct robustness testing.
- (Week 12) Evaluation, benchmarking, and report preparation.
- (week 13) Finalise project deliverables and presentation.

## Roles
### Nick
- UI
- Data acquisition and preprocessing
- Occlusion handling

### Mark
- Move validation
- Occlusion handling

### Jenny
- Piece identification and tracking
- Occlusion handling
