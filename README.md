# Python Image Fuse Tool  
A command line argument wrapper for `cv2.addWeighted()` to fuse images

## Background
I found myself wanting to fuse images I made with my Neural Style Transfer application a fair amount, so I decided to make a command line tool to control all of the parameters to `cv2.addWeighted()` and some image post-processing (i.e. scaling the final image) to speed up my workflow.

## Requirements
This tool works with Python 2.7.X and Python 3.X. The only external packages necessary are Numpy and OpenCV.

Run `pip install requirements.txt` to install required versions of packages.

## Using the tool
```
python image_fuse.py -h

usage: image_fuse.py [-h] [--w1 WEIGHT1] [--w2 WEIGHT2] [--g GAMMA]
                     [--fs SCALE_FACTOR] [--s]
                     img1_path img2_path output_name

Wraps OpenCV's addWeighted()

positional arguments:
  img1_path          Path to first image
  img2_path          Path to second image
  output_name        Path/name of output image

optional arguments:
  -h, --help         show this help message and exit
  --w1 WEIGHT1       Weight for first image
  --w2 WEIGHT2       Weight for second image
  --g GAMMA          Amount added to every pixel in final image
  --fs SCALE_FACTOR  Factor to scale final image
  --s                Flag for whether or not to show the image

```
