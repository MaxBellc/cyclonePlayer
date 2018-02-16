# cyclonePlayer


This is a program that can determine if a green LED is turned on or off based on it's color using OpenCV.  It uses the HSV color space in order to seperate brightness from color.

To use this file, you need a video file.  Sintax is like this:
python3 greenDetector myVideo.mp4

Press p to pause and escape to exit

To make this faster, try resizing the frames to be smaller - there will be less pixles to go through.  
You can also try skipping frames to do less computations.
You can reduce the number of operations also.

