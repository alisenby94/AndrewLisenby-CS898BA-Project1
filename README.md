# HW1: CS 898BA – Image Analysis and Computer Vision

Andrew Lisenby, B.S.

Wichita State University

## Assignment

Purpose: To apply basic image analysis and processing techniques.

Situation: You are working at Meta, and your supervisor’s supervisor walks in, gasping for air. You don’t often interact with him, so this visit is strange. He opens a folder on his personal computer and shows you an image he thinks is an alien, captured by his doorbell camera. You are not convinced and think he is just seeing things. He came to see you because he heard you took an image analysis course in college and wants you to clean up the image so it is easier to make out what is in it.

![Grey alien captured on camera drinking a vintage 25oz Foster's Lager on his way to visit bigfoot.](assets/HW1_IMG_CS898BA.png)

## Description

This repository contains the required files for CS 898 - Image Analysis and Computer Vision (Wichita State University), including an AI_Log file to track AI usage (which I am deliberately avoiding), multi-step image processing output and scripts, and a fun hello_world animation just to add something more pragmatic to the repository.

## Setup & Installation:
**Linux instructions:**
```
cd <RepoDir>
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

**Windows Instructions:**

[Follow this tutorial and then see above instructions.](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview)

## Usage and Execution:
**Running hello_world.py:**

```
cd <RepoDir>/part01
python3 hello_world.py
```

**Running full pipeline:**
```
# Full pipeline execution (time consuming)
bash ./run_full_pipeline.sh

# Run isolated by calling python files individually from project root.
# For example:
python3 part02/basic_statistics.py
```


## TODO:
- [x] Add project title, author, course
- [x] Include brief project overview and "alien image" context
- [x] Add **Setup & Installation** section
- [ ] Add **Usage/Execution** section [IP]
- [x] **Results - Part 2**:
  - Image statistics table for original channels
  - Show/save examples of greyscale/binary/color spaces + normalized RGB
  - Affine transformations summary + sample images
  - Gaussian blur discussion + examples for different sigma values
- [ ] **Results - Part 3**:
  - Edge detection analysis (pros/cons + which worked best)
  - Embed 6 random 5-image comparison plots
  - Summary of total images generated
- [x] Link to `AI_Log.md` and key AI-assisted parts
- [ ] Add **Discussion/Conclusions** (key learnings, challenges, observations)
- [ ] Include table of contents for navigation
- [ ] Add badges (Python, OpenCV) and license if applicable
- [ ] Clean up: remove full assignment text or move to separate file; keep only relevant summary
- [ ] Ensure all images/plots are linked properly and render in GitHub
- [ ] Create run_full_pipeline.sh
- [ ] Refactor decision_making method to reduce time complexity of future assignments [AWT downtime]
- [ ] Make more coffee [IP]