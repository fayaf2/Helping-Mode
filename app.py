from flask import Flask, render_template, request, jsonify
from inference import get_roboflow_model
import supervision as sv
import cv2
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv8 model
model_id = "yolov8n-640"
model = get_roboflow_model(model_id=model_id)

# Supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


# Route to process frames from client


@app.route('/process_frame', methods=['POST'])
def process_frame():
    file = request.files['frame']
    if not file:
        return jsonify({'error': 'No frame provided'}), 400

    # Convert the frame to an OpenCV image
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Run YOLO inference
    results = model.infer(frame)[0]
    detections_list = results.predictions

    object_names = []

    if detections_list:
        boxes = np.array([
            [det.x - det.width / 2, det.y - det.height / 2, det.x + det.width / 2, det.y + det.height / 2]
            for det in detections_list
        ])
        class_ids = np.array([det.class_id for det in detections_list])
        confidences = np.array([det.confidence for det in detections_list])
        object_names = [det.class_name for det in detections_list]

        detections = sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)

        # Annotate frame (if needed for debugging)
        frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections)

    return jsonify({'objects': ', '.join(object_names)})


# Home page
@app.route('/')

def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, port=5050)
