import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- Configuration ---
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SECRET_KEY'] = 'dev' # Needed for flash messages
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'tasks.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'static/uploads')

# Ensure the upload folder exists
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

# --- Routes ---

@app.route('/')
def index():
    tasks = Task.query.all()
    return render_template('index.html', show_sidebar=False, tasks=tasks)

@app.route('/task/<int:task_id>')
def task_detail(task_id):
    task = Task.query.get_or_404(task_id)
    tasks = Task.query.all()
    return render_template('task_detail.html', show_sidebar=True, task=task, tasks=tasks)

@app.route('/tugas')
def tugas():
    tasks = Task.query.all()
    return render_template('tugas.html', show_sidebar=True, tasks=tasks)

@app.route('/new', methods=['GET', 'POST'])
def new_task():
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        content = request.form['content']
        
        thumbnail_file = request.files['thumbnail']
        if thumbnail_file:
            filename = secure_filename(thumbnail_file.filename)
            thumbnail_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            thumbnail_path = 'uploads/' + filename
        else:
            thumbnail_path = None

        new_task = Task(
            title=title,
            description=description,
            content=content,
            thumbnail=thumbnail_path
        )
        
        db.session.add(new_task)
        db.session.commit()
        flash('Materi baru berhasil ditambahkan!', 'success')
        return redirect(url_for('tugas'))
        
    return render_template('new_task.html', show_sidebar=False)

@app.route('/edit/<int:task_id>', methods=['GET', 'POST'])
def edit_task(task_id):
    task = Task.query.get_or_404(task_id)
    if request.method == 'POST':
        task.title = request.form['title']
        task.description = request.form['description']
        task.content = request.form['content']
        
        thumbnail_file = request.files['thumbnail']
        if thumbnail_file:
            filename = secure_filename(thumbnail_file.filename)
            thumbnail_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            task.thumbnail = 'uploads/' + filename

        db.session.commit()
        flash('Materi berhasil diperbarui!', 'success')
        return redirect(url_for('task_detail', task_id=task.id))

    return render_template('edit_task.html', show_sidebar=False, task=task)

@app.route('/delete/<int:task_id>', methods=['POST'])
def delete_task(task_id):
    task = Task.query.get_or_404(task_id)
    db.session.delete(task)
    db.session.commit()
    flash('Materi berhasil dihapus!', 'success')
    return redirect(url_for('tugas'))

# Initialize database within app context
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
