# Amlogic S905x Human Detection

Human detection on Amlogic S905x devices using MS COCO 2017 datasets. Datasets were filtered using fiftyone library so that it only contains 'person' class. Model used is SSD MobileNetV2 and YOLOv4-tiny. The detection result can be viewed to website on this [repository](https://github.com/faldeus0092/tugas-akhir-cctv).

  

# Installation
This guide assumes you have installed armbian to your machine. If you haven't, refer to [this repository](https://github.com/ophub/amlogic-s9xxx-armbian)

## Building the Model

Follow this step if you want to train the model from scratch. Otherwise, use the trained model provided on ssd and yolo folder

### Downloading datasets

  Use [fiftyone](https://docs.voxel51.com/user_guide/export_datasets.html#basic-recipe) to download datasets of certain class. Refer to [fiftyone_coco.ipynb](https://github.com/faldeus0092/amlogic-s905x-human-detection/blob/main/fiftyone_coco.ipynb) and adjust the path to your own machine.

### Training the model
Model are trained on google colab. Use techzizou's tutorial for [MobileNetV2](https://techzizou.com/training-an-ssd-model-for-a-custom-object-using-tensorflow-2-x/) and [YOLOv4-tiny](https://techzizou.com/train-a-custom-yolov4-tiny-object-detector-using-google(-colab-tutorial-for-beginners/ ). The train configs used to train the model on this repository are [this file](https://github.com/faldeus0092/amlogic-s905x-human-detection/blob/main/ssd/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config) for MobileNet V2 and [this file] for [YOLOv4-tiny](https://github.com/faldeus0092/amlogic-s905x-human-detection/blob/main/yolo/yolov4-tiny-custom-.cfg).

## Running the inference from RTSP stream
Clone or download this repo to your Amlogic S905x device
To run human detection inference on s905x devices:
1. Adjust the path to saved model & weights
	- https://github.com/faldeus0092/amlogic-s905x-human-detection/blob/9d5dfaf3e7ea9425c748130037c9a118c9412156/main.py#L22-L24
2. Run the program, example ```main.py --model yolo --api http://localhost:5000/api/footage --url rtsp://KCKS:majuteru5@10.15.40.48:554/Streaming/Channels/1101 --cctv 2```. You can adjust the ```--model``` to ```yolo``` or ```mobilenet```. ```--api``` used for API endpoint that [this website](https://github.com/faldeus0092/tugas-akhir-cctv) is hosted on (adjust to your IP accordingly). ```url``` used for RTSP URL. ```--cctv``` used for CCTV number registered on database. 
	- https://github.com/faldeus0092/amlogic-s905x-human-detection/blob/9d5dfaf3e7ea9425c748130037c9a118c9412156/main.py#L28
3. The result can be seen on website (adjust the IP according to your host) ```http://localhost:5000/video_feed/[cctv_id]```


## To Do

Improving the model performance by using [this repository](https://github.com/ARM-software/armnn)
