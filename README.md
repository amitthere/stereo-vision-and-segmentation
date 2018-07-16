
# CSE573 - Computer Vision and Image Processing - PA2

## Stereo Vision and Image Segmentation

The code implements the requirements mentioned in PA2.pdf

* PA2-Part1.py

This implements 1.1 and 1.2 from the assignment. Disparity Maps are generated from provided images
using block sizes of 3 and 9. Back-projection is used to do consistency checks with the given 
ground-truth. Mean squared error for all the cases is also calculated.

* PA2-Part3.py

Disparity Maps are generated using Dynamic Programming algorithm given in 1.3 reference.

* PA2-Part4.py

view3 is generated using the provided images and their ground-truths.

* PA2-Part5.py

Mean Shift Segmentation is implemented and run on the given Butterfly image.
