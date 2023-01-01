# Codebook

Implementation of paper - [Real-time foreground–background segmentation using codebook model](https://www.sciencedirect.com/science/article/pii/S1077201405000057)

## Description

There are 3 files, and the purpose of each file is shown below:

| file                      | purpose                                                                                                    |
| :------------------------ | :--------------------------------------------------------------------------------------------------------- |
| Codebook.py               | Implementation of Codebook algorithm.                                                                      |
| BackgroundSubtractorCB.py | Make an object that API is similar to other BackgroundSubtractors in cv2.                                  |
| BS_test.py                | Make a comparison for KNN, MOG2, and Codebook. It will show the mask that generates by those 3 algorithms. |

## Usage

```shell
usage: BS_test.py [-h] [-path PATH]

optional arguments:
    -h, --help show this help message and exit
    -path PATH the video path you want to input. (default: pedestrians.avi)

example:
    python3 BS_test.py ./pedestrians.avi
```
