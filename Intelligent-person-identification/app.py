import os
import cv2
from flask import Flask, Response, request, jsonify, send_from_directory
from flask_cors import CORS
from modules.tracking_module import RealTimeTracker

app = Flask(__name__, static_folder='frontend', template_folder='frontend')
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

tracker_system = RealTimeTracker(model_path="yolo11n.pt")
detection_logs = []

# Configurable User Blacklist Rules
active_blacklist = {
    "mask": True,
    "tattoo": False,
    "spectacles": False,
    "wristband": False,
    "bag": False
}

# --- FRONTEND ROUTING ---
@app.route('/')
def serve_frontend():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory('frontend', path)

# --- CONFIGURABLE BLACKLIST ENDPOINTS ---
@app.route('/api/blacklist', methods=['GET', 'POST'])
def manage_blacklist():
    global active_blacklist
    if request.method == 'POST':
        active_blacklist = request.json.get('blacklist', active_blacklist)
        return jsonify({"status": "updated", "blacklist": active_blacklist})
    return jsonify({"blacklist": active_blacklist})

# --- FLEXIBLE NATURAL LANGUAGE SEARCH ---
@app.route('/api/search', methods=['POST'])
def search_person():
    raw_query = request.json.get('query', '').lower()
    
    # Filter out natural language filler words
    stop_words = {"person", "people", "man", "woman", "guy", "with", "wearing", "a", "in", "and", "the", "carrying", "shirt", "pant", "color"}
    query_tokens = [word for word in raw_query.split() if word not in stop_words]
    
    matches = []
    
    for log in detection_logs:
        attr = log["attributes"]
        searchable_text = f"shirt:{attr.get('shirt_color')} pant:{attr.get('pant_color')} hair:{attr.get('hair_color')} mask:{attr.get('mask')} maskcolor:{attr.get('mask_color')} bag:{attr.get('accessory')}".lower()
        
        if query_tokens and any(token in searchable_text for token in query_tokens):
            matches.append({
                "timestamp": f"{log['timestamp']}s",
                "person_id": log["track_id"],
                "shirt": attr.get("shirt_color"),
                "pant": attr.get("pant_color"),
                "hair": attr.get("hair_color"),
                "accessory": attr.get("accessory"),
                "mask": attr.get("mask")
            })

    unique_matches = list({m['person_id']: m for m in matches}.values())
    return jsonify({"query": raw_query, "results": unique_matches})

# --- DYNAMIC ALERTS ENDPOINT ---
@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    alerts = []
    
    for log in detection_logs[-20:]:
        attr = log["attributes"]
        triggered_reasons = []

        if active_blacklist.get("mask") and attr.get("mask") == "Yes":
            triggered_reasons.append("Mask Worn")
        if active_blacklist.get("tattoo") and attr.get("tattoo") == "Yes":
            triggered_reasons.append("Tattoo Visible")
        if active_blacklist.get("spectacles") and attr.get("spectacles") == "Yes":
            triggered_reasons.append("Glasses Worn")
        if active_blacklist.get("wristband") and attr.get("wristband") == "Yes":
            triggered_reasons.append("Wristband Worn")
        if active_blacklist.get("bag") and attr.get("accessory") != "None":
            triggered_reasons.append(f"Carrying Bag ({attr.get('accessory')})")

        if triggered_reasons:
            alerts.append({
                "timestamp": f"{log['timestamp']}s",
                "track_id": log["track_id"],
                "reason": ", ".join(triggered_reasons)
            })
            
    unique_alerts = list({(a['track_id'], a['timestamp']): a for a in alerts}.values())
    return jsonify({"alerts": unique_alerts})

def generate_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp_sec = round(frame_count / fps, 2)
        annotated_frame, targets = tracker_system.process_frame(frame)

        for t in targets:
            detection_logs.append({
                "timestamp": timestamp_sec,
                "track_id": t["track_id"],
                "attributes": t
            })

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['video']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    return jsonify({"status": "success", "video_path": file_path})

@app.route('/api/stream')
def video_stream():
    video_path = request.args.get('path', default=0)
    return Response(generate_video_stream(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)