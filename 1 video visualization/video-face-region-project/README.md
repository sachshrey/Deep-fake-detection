# Video Face Region Project

## Overview
The Video Face Region Project is designed to extract frames from video files and process these frames to isolate important facial regions. The project focuses on both real and fake videos, allowing for a comparative analysis of facial features.

## Project Structure
```
video-face-region-project
├── src
│   ├── extract_frames.py
│   ├── process_face_regions.py
│   └── utils
│       └── __init__.py
├── data
│   ├── videos
│   │   ├── real
│   │   └── fake
│   ├── frames
│   │   ├── real
│   │   └── fake
│   └── processed_frames
│       ├── real
│       └── fake
├── requirements.txt
└── README.md
```

## Installation
To set up the project, clone the repository and install the required dependencies. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage
1. **Extract Frames**: Run the `extract_frames.py` script to extract frames from the real and fake video files. The extracted frames will be saved in the `data/frames/real` and `data/frames/fake` directories.

   ```bash
   python src/extract_frames.py
   ```

2. **Process Face Regions**: After extracting the frames, run the `process_face_regions.py` script to process the frames and isolate important facial regions such as eyes, mouth, nose, ears, jawline, and face border. The processed images will be saved in the `data/processed_frames/real` and `data/processed_frames/fake` directories.

   ```bash
   python src/process_face_regions.py
   ```

## Scripts
- **extract_frames.py**: This script handles the extraction of frames from the provided video files and saves them into the respective folders.
- **process_face_regions.py**: This script processes the extracted frames to isolate and save important facial regions.
- **utils/__init__.py**: This file contains utility functions that assist in frame extraction and face region processing.

## Data Directories
- **data/videos/real**: Contains real video files.
- **data/videos/fake**: Contains fake video files.
- **data/frames/real**: Stores the extracted frames from the real video.
- **data/frames/fake**: Stores the extracted frames from the fake video.
- **data/processed_frames/real**: Contains processed images with only the important facial regions from the real video frames.
- **data/processed_frames/fake**: Contains processed images with only the important facial regions from the fake video frames.

## Requirements
The project requires the following libraries:
- OpenCV
- NumPy
- Other libraries as specified in `requirements.txt`

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.