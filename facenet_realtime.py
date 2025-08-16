import os
import cv2
import time
import subprocess
import threading
import re
import uuid
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
from flask import Flask, request, make_response
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import smtplib
from email.message import EmailMessage

# ---------- CONFIGURATION ----------
folder_name = r"G:\My Drive\reference_images"
unknowns_drive_folder = "UnknownVisitors"
threshold = 1.0
from_email = "example@gmail.com"
from_password = "*********"  # Replace with your Gmail app password
to_email = "Example@gmail.com"
video_duration = 10                # Seconds for unknown video
known_alert_interval = 300         # 5 minutes for known
unknown_alert_interval = 60        # 1 minute for unknown
control_port = 5050
stream_timeout = 300               # 5 minutes (300s) auto stream timeout

live_stream_control = {
    "process": None,
    "event_id": None,
    "stream_started": False,
    "cloudflared": None,
    "stream_url": None,
    "force_end_time": None
}

# ---------- EMAIL ALERT ----------
def send_email(subject, body, video_url=None, accept_url=None, deny_url=None, stream_url=None):
    msg_body = body
    if video_url:
        msg_body += f"\n\n📹 Video: {video_url}"
    if stream_url:
        msg_body += f"\n🔴 Live Stream: {stream_url}"
    if accept_url and deny_url:
        msg_body += f"\n\n✅ Accept: {accept_url}\n❌ Deny: {deny_url}"
    msg = EmailMessage()
    msg.set_content(msg_body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, from_password)
        server.send_message(msg)
        server.quit()
        print("[EMAIL SENT ✅]")
    except Exception as e:
        print(f"[EMAIL FAILED ❌] {e}")

# ---------- GOOGLE DRIVE UPLOAD WITH SAVED CREDENTIALS ----------
def upload_video_to_drive(path):
    gauth = GoogleAuth()
    creds_file = "mycreds.txt"
    gauth.LoadCredentialsFile(creds_file)
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile(creds_file)
    drive = GoogleDrive(gauth)
    folder_list = drive.ListFile({
        'q': f"title='{unknowns_drive_folder}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()
    folder_id = folder_list[0]["id"] if folder_list else None
    if folder_id is None:
        folder = drive.CreateFile({'title': unknowns_drive_folder, 'mimeType': 'application/vnd.google-apps.folder'})
        folder.Upload()
        folder_id = folder["id"]
    file = drive.CreateFile({'parents': [{'id': folder_id}], 'title': os.path.basename(path)})
    file.SetContentFile(path)
    file.Upload()
    print(f"[VIDEO UPLOADED ✅] {file['title']}")
    return f"https://drive.google.com/file/d/{file['id']}/view?usp=sharing"

# ---------- VIDEO RECORDING ----------
def record_video(duration, filename):
    rec_cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (int(rec_cap.get(3)), int(rec_cap.get(4))))
    start = time.time()
    while time.time() - start < duration:
        ret, frame = rec_cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow("Recording Unknown", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    out.release()
    rec_cap.release()
    cv2.destroyWindow("Recording Unknown")
    print("[RECORDING COMPLETE ✅]")

# ---------- START LIVE STREAM ----------
def start_live_stream():
    if live_stream_control["process"]:
        return
    print("[INFO] Starting live stream server...")
    live_stream_control["process"] = subprocess.Popen(["python", "live_stream.py"])
    time.sleep(2)

# ---------- CLOUDFLARED ----------
def start_cloudflared():
    cmd = ["cloudflared", "tunnel", "--url", "http://localhost:5000"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print("[cloudflared]", line.strip())
        match = re.search(r"https://[\w\-]+\.trycloudflare\.com", line)
        if match:
            live_stream_control["stream_url"] = match.group(0) + "/video_feed"
            live_stream_control["cloudflared"] = proc
            return

# ---------- FLASK CONTROL SERVER ----------
control_app = Flask(__name__)

@control_app.route('/stream_action')
def stream_action():
    action = request.args.get('action')
    event_id = request.args.get('id')
    if live_stream_control.get('event_id') == event_id:
        if action in ['accept', 'deny']:
            terminate_stream()
            html = f"<html><body><h2>Stream {action}ed successfully.</h2><p>You may close this page.</p></body></html>"
            return make_response(html, 200)
    return "Invalid or expired request.", 400

def terminate_stream():
    print("[INFO] Terminating Live Stream...")
    if live_stream_control["process"]:
        live_stream_control["process"].terminate()
        live_stream_control["process"] = None
    if live_stream_control["cloudflared"]:
        live_stream_control["cloudflared"].terminate()
        live_stream_control["cloudflared"] = None
    live_stream_control["event_id"] = None
    live_stream_control["stream_started"] = False
    live_stream_control["stream_url"] = None
    live_stream_control["force_end_time"] = None

# Start Flask control server in background thread
threading.Thread(target=lambda: control_app.run(host='0.0.0.0', port=control_port), daemon=True).start()

# ---------- MODEL LOADING ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

reference_embeddings = []
reference_names = []
for f in os.listdir(folder_name):
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        path = os.path.join(folder_name, f)
        img = Image.open(path).convert("RGB")
        face = mtcnn(img)
        if face is not None:
            emb = resnet(face.unsqueeze(0).to(device)).detach().cpu()
            reference_embeddings.append(emb)
            reference_names.append(f)

if not reference_embeddings:
    raise Exception(f"No valid faces found in the folder '{folder_name}'.")

cap = cv2.VideoCapture(0)
print("📷 Face recognition running...")

alert_timestamps = {}
last_unknown_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, _ = mtcnn.detect(img)
    if boxes is not None:
        face = mtcnn(img)
        if face is not None and len(boxes) > 0:
            emb = resnet(face.unsqueeze(0).to(device)).detach().cpu()
            dists = [(emb - ref).norm().item() for ref in reference_embeddings]
            mindist = min(dists)
            idx = dists.index(mindist)
            now = time.time()
            x1, y1, x2, y2 = [int(c) for c in boxes[0]]
            if mindist < threshold:
                name = reference_names[idx]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                if now - alert_timestamps.get(name, 0) > known_alert_interval:
                    send_email("✅ Known Person Detected", f"{name} was recognized at the door.")
                    alert_timestamps[name] = now
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, "Unknown", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                if now - last_unknown_time > unknown_alert_interval:
                    cap.release()
                    filename = f"unknown_{int(now)}.avi"
                    record_video(video_duration, filename)
                    video_link = upload_video_to_drive(filename)
                    start_live_stream()
                    cloudflared_thread = threading.Thread(target=start_cloudflared)
                    live_stream_control["stream_url"] = None
                    cloudflared_thread.start()
                    while live_stream_control["stream_url"] is None:
                        time.sleep(0.5)
                    event_id = str(uuid.uuid4())
                    live_stream_control["event_id"] = event_id
                    live_stream_control["stream_started"] = True
                    live_stream_control["force_end_time"] = time.time() + stream_timeout
                    accept_link = f"http://localhost:{control_port}/stream_action?action=accept&id={event_id}"
                    deny_link = f"http://localhost:{control_port}/stream_action?action=deny&id={event_id}"
                    send_email(
                        "🚨 Unknown Visitor Detected",
                        "An unknown person was detected at your door.",
                        video_url=video_link,
                        accept_url=accept_link,
                        deny_url=deny_link,
                        stream_url=live_stream_control["stream_url"])
                    last_unknown_time = now
                    cap = cv2.VideoCapture(0)
    if live_stream_control.get("stream_started") and time.time() > live_stream_control["force_end_time"]:
        terminate_stream()
        print("[STREAM TIMEOUT] Stream automatically terminated after timeout.")
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
