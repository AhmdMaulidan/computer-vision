import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.util import view_as_windows
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- Configuration ---
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SECRET_KEY'] = 'dev'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'tasks.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'static/uploads')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)

# --- Database Model ---
class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    thumbnail = db.Column(db.String(100), nullable=True)
    description = db.Column(db.Text, nullable=False)
    content = db.Column(db.Text, nullable=False)

# --- GLCM Processing Functions ---
def create_feature_maps(image, distances, angles, window_size):
    pad = window_size // 2
    padded_image = np.pad(image, pad, mode='reflect')
    windows = view_as_windows(padded_image, (window_size, window_size))
    properties = ['contrast', 'correlation', 'energy', 'homogeneity']
    feature_maps = {prop: np.zeros(image.shape, dtype=np.float32) for prop in properties}

    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            window = windows[i, j]
            if np.max(window) == np.min(window):
                props = {'contrast': 0, 'correlation': 1, 'energy': 1, 'homogeneity': 1}
            else:
                glcm = graycomatrix(window, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
                props = {prop: graycoprops(glcm, prop)[0, 0] for prop in properties}
            for prop in properties:
                feature_maps[prop][i, j] = props[prop]
    return feature_maps

def save_map_as_image(feature_map, base_filename, prop_name, upload_folder):
    normalized_map = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    color_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_VIRIDIS)
    map_filename = f'{prop_name}_{base_filename}'
    map_filepath = os.path.join(upload_folder, map_filename)
    cv2.imwrite(map_filepath, color_map)
    return url_for('static', filename=f'uploads/{map_filename}')

# --- Routes ---
@app.route('/')
def index():
    tasks = Task.query.all()
    return render_template('index.html', show_sidebar=False, tasks=tasks)

@app.route('/task/<int:task_id>', methods=['GET', 'POST'])
def task_detail(task_id):
    task = Task.query.get_or_404(task_id)
    tasks = Task.query.all()
    
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image selected', 'error')
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            flash('No image selected', 'error')
            return redirect(request.url)
        
        angle = int(request.form.get('angle', '0'))
        distance = int(request.form.get('distance', '1'))
        window_size = int(request.form.get('window_size', '7'))

        if window_size % 2 == 0:
            window_size += 1
            flash(f'Window size must be odd, rounding up to {window_size}', 'info')

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None: raise Exception('Could not read image format.')

            feature_maps_data = create_feature_maps(img, [distance], [np.deg2rad(angle)], window_size)
            
            feature_map_urls = {}
            global_properties = {}
            for prop, data in feature_maps_data.items():
                feature_map_urls[prop] = save_map_as_image(data, filename, prop, app.config['UPLOAD_FOLDER'])
                global_properties[prop.capitalize()] = np.mean(data)

            image_url = url_for('static', filename='uploads/' + filename)
            
            return render_template(
                'task_detail.html', 
                show_sidebar=True, 
                task=task, 
                tasks=tasks, 
                feature_maps=feature_map_urls, 
                global_properties=global_properties,
                image_url=image_url,
                params={'angle': angle, 'distance': distance, 'window': window_size}
            )

        except Exception as e:
            flash(f'An error occurred: {e}', 'error')
            return redirect(request.url)

    return render_template('task_detail.html', show_sidebar=True, task=task, tasks=tasks)


@app.route('/tugas')
def tugas():
    first_task = Task.query.first()
    if first_task: return redirect(url_for('task_detail', task_id=first_task.id))
    else: flash('No project created yet.', 'info'); return redirect(url_for('index'))

def setup_database(app):
    with app.app_context():
        db.drop_all()
        db.create_all()
        if not Task.query.first():
            glcm_task = Task(
                title='Gray-Level Co-occurrence Matrix',
                thumbnail='assets/glcmthumnail.png', 
                description='Analisis tekstur citra menggunakan GLCM untuk ekstraksi fitur.',
                content='''
<p>Alat ini melakukan analisis tekstur lokal pada gambar menggunakan metode <strong>GLCM</strong>. Proses ini memindai gambar dengan 'jendela geser' untuk menghasilkan <strong>peta fitur (feature maps)</strong> yang menunjukkan bagaimana properti tekstur seperti Kontras, Korelasi, Energi, dan Homogenitas didistribusikan di seluruh gambar.</p>
<p>Unggah gambar, pilih parameter analisis, dan lihat bagaimana setiap piksel pada peta hasil mewakili tekstur dari area lokal di sekitarnya pada gambar asli.</p>
'''
            )
            db.session.add(glcm_task)
            db.session.commit()

setup_database(app)

if __name__ == '__main__':
    app.run(debug=True)
