# gest
Hand gestures as an input device

![example](https://raw.githubusercontent.com/bm371613/gest/master/images/example.gif)

## Why
For health related reasons, I had to stop using a mouse and a keyboard. [Talon](https://talonvoice.com/) allowed me to type with my voice and move the cursor with my eyes. This project was started to complement this setup with hand gestures.

## Development status
The project is in an early stage of development. I use it on daily basis, so it should be good enough for some.

What is implemented:
- pinching gesture recognition, in one hand orientation
- heatmap output, separate for left and right hand, indicating pinched point position
- demo for testing recognition models
- example script for simulating mouse clicks and scrolling
- scripts for producing and reviewing training data

### Bias
The gesture recognition model was trained on images of my hands, taken with my hardware in my working environment, so it is probably heavily biased.
I hope people who want to use it, but recognition quality prevents them from it, would capture some images of their hands using included tooling and donate it to the project, so that over time it works well for everyone.

## Installation

Use Python 3.6, 3.7 or 3.8 and [in a virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) run

`pip install gest`

If you clone this repository, you can get the exact versions of required libraries that I am using with [Poetry](https://python-poetry.org/)

`poetry install`

## Walkthrough

### Demo

First check how the included model works for you. Run

`python -m gest.demo`

and see if it recognizes your gestures as here:

![demo](https://raw.githubusercontent.com/bm371613/gest/master/images/demo.gif)

If you have multiple cameras, you can pick one like

`python -m gest.demo --camera 2`

Camera numbers are not necessarily consecutive.
Two cameras may be accessible as 0 and 2.
This option is supported by other commands as well.

### Example script

In the presentation on top I am running

`python -m gest.examples.two_handed_scroll_and_click`

It only acts if it detects both hands pinching and based on their relative position:
- double clicks if you cross your hands
- scrolls up or down if your hands pinch at different heights
- left clicks if your hands (almost) touch
- right clicks if your hands are on the same height, but not close horizontally (this action is delayed by a fraction of a second to prevent accidental use)

### Controlling CPU load

For everyday use, you don't want to dedicate too much resources to gesture recognition. You can control it by setting `OMP_NUM_THREADS`, as in

`OMP_NUM_THREADS=2 python -m gest.examples.two_handed_scroll_and_click`

Try different values to find balance between responsiveness and CPU load.

## Custom scripts

The demo and example scripts serve two additional purposes:
they can be used as templates for custom scripts
and they define the public API for the purpose of semantic versioning.

## Training data annotation

### Capturing

`python -m gest.annotation.capture --countdown 5 data_directory`

will help you create annotated images.
Once you start automatic annotation (press `a` to start/stop) it will ask you to pinch a given point with your left or right hand, or to not pinch ("background").

You will have 5 seconds before the image is captured (the `--countdown`).

You will also see the last annotated image for quick review. It can be deleted with `d`.

### Reviewing

`python -m gest.annotation.review --time 1 data_directory closed_pinch_left`

will let you review all images annotated as left hand pinch in `data_directory`, showing you each for 1 second if you start/stop automatic advancing with `a`. Otherwise you can go to the next/previous image with `n`/`p`. Delete incorrectly annotated images with `d`.

You should also review `closed_pinch_right` and `background`.

### Annotation guidelines
It makes sense to annotate realistic training data that the model performs poorly on, like if
- it mistakenly detects a pinch when you pick up the phone,
- it doesn't detect pinching when you wear a skin colored shirt.

If it performs poorly overall, it's good to capture the images in many short sessions, with different lighting, clothes, background, camera angle.

The point isn't though to look for tricky cases or stretch the definition of a pinching gesture to include a different hand orientation (eg. with pinching fingers pointing towards the camera).

### Donating annotated data
Contact me [b.marcinkowski@leomail.pl](mailto:b.marcinkowski@leomail.pl)
