from flask import Flask, Response
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # Use index 0 for the built-in/default webcam

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        # Encode captured frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        # Yield frame to web client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # Returns streamed response mimicking an MJPEG server
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Listen on all interfaces (required for tunnel/remote access)
    app.run(host='0.0.0.0', port=5000)
