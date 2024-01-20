from flask import Flask, render_template, Response
import test
import test
from test import VideoCamera
app = Flask(__name__, template_folder="C:\\stress mini project\\Stress_Detector-master\\Stress_Detector-master\\code\\templates")

@app.route('/')
def index():
    
    # Main page
    return render_template('index.html')
def gen(test):
    """Video streaming generator function."""
    while True:
        frame= test.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
@app.route("/predict")
def predict():  
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
if __name__ == "__main__":
    app.run(debug=False,threaded=False)
