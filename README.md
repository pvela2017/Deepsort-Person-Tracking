# Deepsort-Person-Tracking

This work proposes a person identification and tracking method based on Yolact, Deepsort and color feature extraction for an agricultural robot using a single depth camera. First, Yolact identification method was used to detect all the people in the frame, then their bounding box were processed to extract the main color of theirs clothes. Thus, the target person can be identified and the remaining ones discarded. Finally, the Deepsort algorithm was used to track the target person around the image frame. Experimental results show that in an indoor environment, with controlled light intensity and no occlusion of the target, it is possible to track accurately a person.

## How To Run
```
python object_tracker5.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/test2.mp4 --top_color blue --bottom_color blue
```
## Results
[![Demo](https://img.youtube.com/vi/N2NRBbaBJrM/0.jpg)](https://www.youtube.com/watch?v=N2NRBbaBJrM)

You can find further information here: 

<br />
Project based on:
https://github.com/theAIGuysCode/yolov4-deepsort
