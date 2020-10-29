import imghdr
import os, time
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory, jsonify
from werkzeug.utils import secure_filename
import json
import uuid
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.mp4', '.avi']
app.config['UPLOAD_PATH'] = 'uploads'
app.config['TASKS_PATH'] = 'tasks'
app.config['ERROR_PATH'] = 'errors'
app.config['PROCESSED_PATH'] = 'static/processed'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['SECRET_KEY'] = '__m3t0d1c4__'
socketio = SocketIO(app)


# Create dirs
if not os.path.exists(app.config['UPLOAD_PATH']):
    os.mkdir(app.config['UPLOAD_PATH'])
if not os.path.exists(app.config['PROCESSED_PATH']):
    os.mkdir(app.config['PROCESSED_PATH'])
if not os.path.exists(app.config['TASKS_PATH']):
    os.mkdir(app.config['TASKS_PATH'])
if not os.path.exists(app.config['ERROR_PATH']):
    os.mkdir(app.config['ERROR_PATH'])

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                          'favicon.ico',mimetype='image/vnd.microsoft.icon')
@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    style = request.form['style']
    uid = request.form.get('uid') # could be null
    file_ext = request.form.get('ext')
    print ("UID:", uid)

    if filename != '':
        # CHECK EXTENSION
        file_ext = os.path.splitext(filename)[1]
#        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
#            return redirect(url_for('index', error="Video extension not valid"))

        # SAVE FILE
        if uid == None or uid == "":
            uid = uuid.uuid4()
        saved_filename = str(uid)+file_ext
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], saved_filename))

        # SAVE TASK JSON DATA IF THIS TASK IS NOT ALREADY DONE
        is_file = os.path.isfile(app.config['PROCESSED_PATH']+'/'+str(uid)+'_'+str(style)+'.mp4')
        print("UID", str(uid), "CHECK_FILE", str(is_file))
        if not is_file:
            json_data = { 'uid': str(uid), 'file': saved_filename, 'style': str(style)}
            with open(app.config['TASKS_PATH']+'/'+str(uid)+'_'+str(style)+'.json', 'w') as fp:
                json.dump(json_data, fp)
    else:
        # SAVE TASK JSON DATA IF THIS TASK IS NOT ALREADY DONE
        is_file = os.path.isfile(app.config['PROCESSED_PATH']+'/'+str(uid)+'_'+str(style)+'.mp4')
        print("UID", str(uid), "CHECK FILE", str(is_file))
        if not is_file:
            json_data = { 'uid': str(uid), 'file': str(uid)+file_ext, 'style': str(style)}
            with open(app.config['TASKS_PATH']+'/'+str(uid)+'_'+str(style)+'.json', 'w') as fp:
                json.dump(json_data, fp)

    return redirect(url_for('index', uid=str(uid), ext=file_ext, style=style))

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

@app.route('/check_upload/<uid>/<style>')
def check_upload(uid, style):
    if os.path.isfile(app.config['PROCESSED_PATH']+'/'+uid+'_'+style+'.mp4'):
        size = os.path.getsize(app.config['PROCESSED_PATH']+'/'+uid+'_'+style+'.mp4')
        time.sleep(0.2)
        actual_size = os.path.getsize(app.config['PROCESSED_PATH']+'/'+uid+'_'+style+'.mp4')
        if size == 0 or size != actual_size:
            return jsonify(check=False,error="")
        print ("VIDEO PROCESSED")
        return jsonify(check=True, error="")
    if os.path.isfile(app.config['ERROR_PATH']+'/'+uid+'.json'):
        print ("VIDEO PROCESS ERROR")
        os.rename(app.config['ERROR_PATH']+'/'+uid+'.json', app.config['ERROR_PATH']+'/'+uid+'.processed')
        return jsonify(check=False, error="Video process error, try another style or maybe we cannot detect a face, use other video.")
    return jsonify(check=False, error="")

@socketio.on('check_process', namespace='/check')
def check(uid, style):
    print ("CHECK PROCESS FOR IMAGE:", uid, "AND STYLE:", style)
    t_end = time.time() + 60 * 15
    size = 0
    while time.time() < t_end:
        time.sleep(0.2)
        if os.path.isfile(app.config['PROCESSED_PATH']+'/'+uid+'_'+style+'.mp4'):
            actual_size = os.path.getsize(app.config['PROCESSED_PATH']+'/'+uid+'_'+style+'.mp4')
            if size == 0 or size != actual_size:
               size = actual_size
               continue
            print ("VIDEO PROCESSED")
            emit('check_ready', True)
            return
        if os.path.isfile(app.config['ERROR_PATH']+'/'+uid+'.json'):
            print ("VIDEO PROCESS ERROR")
            emit('check_ready', False)
            os.rename(app.config['ERROR_PATH']+'/'+uid+'.json', app.config['ERROR_PATH']+'/'+uid+'.processed')
            return
    print ("VIDEO PROCESS TIMEOUT")

if __name__ == '__main__':  # pragma: no cover
    app.run(debug=True,host='0.0.0.0',port=80)

