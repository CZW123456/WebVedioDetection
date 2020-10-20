# WebVedieoDetection

A video object detection project that takes the video which is uploaded by the user and does object
detection based on [Yolo v3](https://github.com/eriklindernoren/PyTorch-YOLOv3).

**How to install**
> git clone git@github.com:CZW123456/WebVedioDetection.git
>
> pip install -r requirement.txt
> 
> cd weights
>
> bash download_weights.sh

**How to use** 
> python app.py --ip xxxx --port xxxx
>
> Open the web broswer and type in the address, say "http://0.0.0.0:1111" to enter the homepage.
>
> Follow the instruction presented in the homepage

Argument "ip" is the IP address where the service is launched.

**How it works**
+ First of all, the user should upload a video encoded with **H.264** to the server according to the instruction in the home page. If the vedio is not encoded with **H.264**, the broswer may not illustrate it properly.
+ [Yolo v3](https://github.com/eriklindernoren/PyTorch-YOLOv3) is utilized to perform video object detection based on the vedio user uploaded. Each frame is processed individually.
+ I currently do not perform advanced signal processing to the audio signal of the uploaded video. The audio signal is extracted and stored in the server for future processing. 
+ The stored audio is also utilized to mux with processed vedio, yielding a complete video file and it is encoded with **H.264** to guarantee it can be illustrated in all main-stream web broser such Chrome or IE Explore.
+ I also provide the download feature which the user can optionally choose to download the processed vedio.

**Note**
+ Pre-trained weights for [Yolo v3](https://github.com/eriklindernoren/PyTorch-YOLOv3) is utilized. Weights download script is in the /weights directory. 
+ Web service is based on [Flask](https://github.com/technext/Flusk), a light-weighted web framework for Python.
+ Uploaded vedios of any resolution can be processed by the server since [Yolo v3](https://github.com/eriklindernoren/PyTorch-YOLOv3) is a fully-convolution model without fully-connected layers.
+ We provide some example vedios in the /samples directory with which user can test the service.
+ I have found **significant acceleration** when the pipeline is implemented in a **batch processing** way. Take the vedio in "/sample/1603117268720758.mp4" as example, almost **2x acceleration** is achivable compared with frame-by-frame processing fashion with single NVIDIA 1080Ti GPU.


**To Do**
+ Using **multiprocessing toolkit** in Pytorch to slide the uploaded video into multiple chunks and process each chunk in parallel.
+ Add some audio signal processing features like denoising, fading-in, fading-out etc.

**Reference**
+ Minimal implementation of [Yolo v3](https://github.com/eriklindernoren/PyTorch-YOLOv3).
+ I referred to this [repo](https://github.com/boblef/auto_transcript) for the website design.