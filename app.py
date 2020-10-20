from flask import Flask, render_template, request, send_file
from models import *
from utils.utils import *
from utils.datasets import *
import os
import torch
import torchvision.transforms as transforms
import cv2
import subprocess
from pydub import AudioSegment
import ffmpeg
import torch.multiprocessing as mp
import argparse


app = Flask(__name__, static_folder='static')

# Path for user temporary uploaded video file
UPLOADED_DIR = "static/uploaded"
if not os.path.isdir(UPLOADED_DIR):
    os.makedirs(UPLOADED_DIR)
UPLOADED_BASE = os.path.join(UPLOADED_DIR, "tmp.mp4")


# Path for model configuration and pre-trained weight for YOLO model
model_def = "config/yolov3.cfg"
weights_path = "weights/yolov3.weights"

# Some hyper-parameter used in YOLO model
class_path = "config/coco.names"
conf_thres = 0.8
nms_thres = 0.4
batch_size = 1
img_size = 416

# # Batch size in processing incoming video stream
# num_processor = mp.cpu_count()
# num_process = num_processor // 2

# Get actual number of video frames, cv2.CAP_PROP_FRAME_COUNT may be inaccurate and lead to some tricky problems
video = cv2.VideoCapture(UPLOADED_BASE)
num_frames = 0
while 1:
    ret, _ = video.read()
    if ret == False:
        break
    num_frames += 1

# Directory path for processed video in the server
result_dir = os.path.join(os.getcwd(), "static", "detected")
if os.path.isdir(result_dir) == False:
    os.makedirs(result_dir)

# Specify device where computing happens and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet(model_def, img_size=img_size).to(device)
if weights_path.endswith(".weights"):
    model.load_darknet_weights(weights_path)
else:
    model.load_state_dict(torch.load(weights_path))

# Set in evaluation mode
model.eval()

# Extracts class labels from file
classes = load_classes(class_path)

# Colors of BBox for different classes
COLORS = np.random.uniform(255, 125, size=(len(classes), 3))
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')


@app.route('/upload_file', methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        f = request.files['file']
        f.save(UPLOADED_BASE)
        return render_template('dashboard.html', result_status=False)


@app.route('/process', methods=["GET", "POST"])
def process():
    if request.method == "POST":
        status = request.form["user_request"]
        if status == "top":
            return render_template("index.html")
        elif status == "processing_singleprocess_single" or status == "processing_singleprocess_batch":
            if status == "processing_singleprocess_single":
                bs = 1
            else:
                bs = 32
            # extract meta data from the video stream
            probe = ffmpeg.probe(UPLOADED_BASE)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            # extract audio of the video and store them in the disk as .wav file
            audio = AudioSegment.from_file(UPLOADED_BASE)
            audio.export(os.path.join(result_dir, "tmp_audio.mp3"), format="mp3")
            # input stream
            process1 = (
                ffmpeg
                    .input(UPLOADED_BASE)
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24', vf="fps=25")
                    .run_async(pipe_stdout=True)
            )
            # output stream
            process2 = (
                ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
                    .output(os.path.join(result_dir, "video_detected.mp4"), pix_fmt='yuv420p')
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
            )
            # loop through a batch of frames: first recover RGB image from input bytes and do some processing
            # and then dump it to bytes
            while 1:
                in_bytes = process1.stdout.read(bs * width * height * 3)
                if not in_bytes:
                    break
                in_frame_array = (
                    np.frombuffer(in_bytes, np.uint8)
                )
                num_frame_contain = len(in_frame_array) // (width * height * 3)
                in_frame = in_frame_array.reshape((num_frame_contain, height, width, 3))
                batch_input_imgs = torch.zeros(size=(bs, 3, img_size, img_size)).to(device)
                for i in range(num_frame_contain):
                    img = transforms.ToTensor()(in_frame[i])
                    # Pad to square resolution
                    img, _ = pad_to_square(img, 0)
                    # Resize
                    img = resize(img, img_size)
                    batch_input_imgs[i] = img
                # Get detections
                with torch.no_grad():
                    batch_detections = model.forward(batch_input_imgs)
                    batch_detections = non_max_suppression(batch_detections, conf_thres, nms_thres)
                # Add boundingBox
                for frame_idx, detection in enumerate(batch_detections):
                    if detection is not None:
                        # Rescale boxes to original image
                        detection = rescale_boxes(detection, img_size, [height, width])
                        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                            label = "{}: {:.2f}%".format(classes[int(cls_pred.item())], cls_conf.item() * 100)
                            y = y1 - 15 if y1 - 15 > 15 else y1 + 15
                            cv2.rectangle(in_frame[frame_idx], (x1, y1), (x2, y2), color=COLORS[int(cls_pred.item())])
                            cv2.putText(in_frame[frame_idx], label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        COLORS[int(cls_pred.item())], 2)
                process2.stdin.write(
                    in_frame
                        .astype(np.uint8)
                        .tobytes()
                )
            # close video stream
            process2.stdin.close()
            process1.wait()
            process2.wait()
            process1.stdout.close()
            # add audio
            cmd = "ffmpeg -i static/detected/video_detected.mp4 -i static/detected/tmp_audio.mp3 -c:v copy -c:a aac -strict experimental -y static/detected/video_detected_with_audio.mp4"
            subprocess.run(cmd, shell=True)
            result_status = True
            # release GPU memory
            torch.cuda.empty_cache()
            return render_template("dashboard.html", result_status=result_status)

@app.route('/download/', methods=['GET', 'POST'])
def download():
    print("download")
    return send_file("static/detected/video_detected_with_audio.mp4", as_attachment=True)


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1111)
    args = parser.parse_args()
    ip = "0.0.0.0"
    # ip = '162.105.85.250'
    app.run(host=args.ip, port=args.port, debug=True)
