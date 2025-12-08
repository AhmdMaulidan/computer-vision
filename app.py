import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from matplotlib.colors import ListedColormap
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

    if task.title == 'K-Nearest Neighbors':
        if request.method == 'POST':
            flash('Classification process started...', 'info')
            if 'dataset' not in request.files or request.files['dataset'].filename == '':
                flash('No dataset selected. Please upload a CSV file.', 'error')
                return redirect(request.url)
            
            file = request.files['dataset']
            
            try:
                k = int(request.form.get('k_value', '5'))
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                flash(f'File "{filename}" uploaded successfully.', 'info')

                dataset = pd.read_csv(filepath)
                flash('CSV data loaded successfully.', 'info')

                if dataset.shape[1] < 2:
                    raise ValueError("Dataset must have at least two columns (features + target).")

                X = dataset.iloc[:, :-1].values
                y_raw = dataset.iloc[:, -1]

                if pd.api.types.is_string_dtype(y_raw):
                    flash('Text-based target variable detected. Applying LabelEncoder.', 'info')
                    le = LabelEncoder()
                    y = le.fit_transform(y_raw)
                else:
                    y = pd.to_numeric(y_raw, errors='coerce')
                    nan_mask = y.isna()
                    if nan_mask.any():
                        flash(f'Warning: Dropped {nan_mask.sum()} rows with non-numeric target values.', 'warning')
                        X = X[~nan_mask]
                        y = y[~nan_mask].astype(int)

                if X.shape[0] == 0:
                    raise ValueError("Dataset is empty after cleaning. Please check your CSV file's target column.")

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
                flash('Data split into training and testing sets.', 'info')

                sc = StandardScaler()
                X_train_scaled = sc.fit_transform(X_train)
                X_test_scaled = sc.transform(X_test)
                flash('Features scaled successfully.', 'info')

                classifier = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
                classifier.fit(X_train_scaled, y_train)
                flash('KNN model trained successfully.', 'info')

                y_pred = classifier.predict(X_test_scaled)

                cm = confusion_matrix(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)

                plot_url = None
                if X.shape[1] == 2:
                    flash('Generating decision boundary plot for 2 features...', 'info')
                    plt.figure(figsize=(10, 6))
                    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
                    cmap_bold = ['darkred', 'darkgreen', 'darkblue']

                    h = .02
                    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
                    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

                    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)

                    plt.contourf(xx, yy, Z, cmap=cmap_light)

                    import seaborn as sns
                    sns.scatterplot(x=X_train_scaled[:, 0], y=X_train_scaled[:, 1], hue=y_train, palette=cmap_bold, alpha=1.0, edgecolor="black")
                    plt.title(f'KNN Decision Boundary (K={k})')
                    plt.xlabel('Feature 1 (Scaled)')
                    plt.ylabel('Feature 2 (Scaled)')

                    plot_filename = f'knn_plot_{filename}.png'
                    plot_filepath = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)
                    plt.savefig(plot_filepath)
                    plt.close()
                    plot_url = url_for('static', filename=f'uploads/{plot_filename}')
                    flash('Plot generated successfully.', 'info')
                else:
                    flash('Decision boundary plot is only available for datasets with 2 features.', 'info')

                return render_template('knn_detail.html', 
                                    show_sidebar=True, 
                                    task=task, 
                                    tasks=tasks,
                                    results={
                                        'cm': cm.tolist(),
                                        'accuracy': accuracy,
                                        'k': k,
                                        'report': report,
                                        'plot_url': plot_url
                                    },
                                    params={'k': k})

            except Exception as e:
                flash(f'An error occurred: {e}', 'error')
                return redirect(request.url)

        return render_template('knn_detail.html', show_sidebar=True, task=task, tasks=tasks)

    if task.title == 'Naive Bayes':
        # Hardcoded Play Tennis Dataset
        data = {
            'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
            'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
            'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
            'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
            'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
        }
        df = pd.DataFrame(data)

        # Encoders for categorical data
        le_outlook = LabelEncoder()
        le_temp = LabelEncoder()
        le_humidity = LabelEncoder()
        le_wind = LabelEncoder()
        le_play = LabelEncoder()

        # Fit and transform the dataset
        df['Outlook_n'] = le_outlook.fit_transform(df['Outlook'])
        df['Temperature_n'] = le_temp.fit_transform(df['Temperature'])
        df['Humidity_n'] = le_humidity.fit_transform(df['Humidity'])
        df['Wind_n'] = le_wind.fit_transform(df['Wind'])
        df['Play_n'] = le_play.fit_transform(df['Play'])

        # Features and Target
        X = df[['Outlook_n', 'Temperature_n', 'Humidity_n', 'Wind_n']]
        y = df['Play_n']

        # Train Model
        model = GaussianNB()
        model.fit(X, y)

        prediction_result = None
        probs = None

        if request.method == 'POST':
            try:
                # Get form data
                outlook_input = request.form.get('outlook')
                temp_input = request.form.get('temperature')
                humidity_input = request.form.get('humidity')
                wind_input = request.form.get('wind')

                # Encode inputs
                # Note: We need to handle potential unseen labels if user modifies form, 
                # but with fixed select options it should be fine.
                input_data = [[
                    le_outlook.transform([outlook_input])[0],
                    le_temp.transform([temp_input])[0],
                    le_humidity.transform([humidity_input])[0],
                    le_wind.transform([wind_input])[0]
                ]]

                # Predict
                pred_n = model.predict(input_data)[0]
                pred_prob = model.predict_proba(input_data)[0]
                
                prediction_result = le_play.inverse_transform([pred_n])[0]
                probs = {
                    le_play.inverse_transform([0])[0]: f"{pred_prob[0]*100:.2f}%",
                    le_play.inverse_transform([1])[0]: f"{pred_prob[1]*100:.2f}%"
                }
                
                flash(f'Prediction: {prediction_result}', 'success')

            except Exception as e:
                flash(f'Error during prediction: {e}', 'error')

        return render_template('naive_bayes_detail.html', 
                             show_sidebar=True, 
                             task=task, 
                             tasks=tasks,
                             dataset=data,
                             prediction=prediction_result,
                             probs=probs)

    if task.title == 'Decision Tree':
        # Hardcoded Iris Dataset (like Naive Bayes with Play Tennis)
        iris = load_iris()
        
        # Create DataFrame for display
        data = {
            'Sepal Length': iris.data[:, 0].tolist(),
            'Sepal Width': iris.data[:, 1].tolist(),
            'Petal Length': iris.data[:, 2].tolist(),
            'Petal Width': iris.data[:, 3].tolist(),
            'Species': [iris.target_names[t] for t in iris.target]
        }
        df = pd.DataFrame(data)
        
        # Prepare features and target
        X = iris.data
        y = iris.target
        feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        class_names = iris.target_names.tolist()
        
        # Train Decision Tree
        clf = DecisionTreeClassifier(criterion='gini', random_state=42)
        clf.fit(X, y)
        
        # Generate Tree Visualization
        plt.figure(figsize=(16, 10))
        plot_tree(clf, filled=True, feature_names=feature_names, 
                 class_names=class_names, rounded=True, fontsize=9)
        tree_filename = 'decision_tree_iris.png'
        tree_filepath = os.path.join(app.config['UPLOAD_FOLDER'], tree_filename)
        plt.tight_layout()
        plt.savefig(tree_filepath, dpi=150, bbox_inches='tight')
        plt.close()
        tree_url = url_for('static', filename=f'uploads/{tree_filename}')
        
        # Generate Decision Boundary (using first 2 features)
        X_2d = X[:, :2]
        clf_2d = DecisionTreeClassifier(criterion='gini', random_state=42)
        clf_2d.fit(X_2d, y)
        
        plt.figure(figsize=(10, 6))
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                           np.arange(y_min, y_max, 0.02))
        Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        colors_bold = ['#FF0000', '#00FF00', '#0000FF']
        
        plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
        for idx, cls in enumerate(class_names):
            mask = y == idx
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors_bold[idx], 
                       label=cls, edgecolor='black', s=50)
        plt.title('Decision Boundary (Sepal Length vs Sepal Width)')
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.legend()
        
        boundary_filename = 'dt_boundary_iris.png'
        boundary_filepath = os.path.join(app.config['UPLOAD_FOLDER'], boundary_filename)
        plt.savefig(boundary_filepath)
        plt.close()
        boundary_url = url_for('static', filename=f'uploads/{boundary_filename}')
        
        prediction_result = None
        probs = None
        
        if request.method == 'POST':
            try:
                # Get form data
                sepal_length = float(request.form.get('sepal_length'))
                sepal_width = float(request.form.get('sepal_width'))
                petal_length = float(request.form.get('petal_length'))
                petal_width = float(request.form.get('petal_width'))
                
                input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
                pred = clf.predict(input_data)[0]
                pred_prob = clf.predict_proba(input_data)[0]
                
                prediction_result = class_names[pred]
                probs = {class_names[i]: f"{p*100:.2f}%" for i, p in enumerate(pred_prob)}
                
                flash(f'Prediksi: {prediction_result}', 'success')
                
            except Exception as e:
                flash(f'Error: {e}', 'error')
        
        return render_template('decision_tree_detail.html',
                             show_sidebar=True,
                             task=task,
                             tasks=tasks,
                             dataset=data,
                             tree_url=tree_url,
                             boundary_url=boundary_url,
                             prediction=prediction_result,
                             probs=probs)

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
    tasks = Task.query.all()
    return render_template('tugas.html', show_sidebar=True, tasks=tasks)

def setup_database(app):
    with app.app_context():
        db.drop_all()
        db.create_all()
        glcm_task = Task(
            title='Gray-Level Co-occurrence Matrix',
            thumbnail='assets/glcmthumnail.png', 
            description='Analisis tekstur citra menggunakan GLCM untuk ekstraksi fitur.',
            content='''
<p>Alat ini melakukan analisis tekstur lokal pada gambar menggunakan metode <strong>GLCM</strong>. Proses ini memindai gambar dengan \'jendela geser\' untuk menghasilkan <strong>peta fitur (feature maps)</strong> yang menunjukkan bagaimana properti tekstur seperti Kontras, Korelasi, Energi, dan Homogenitas didistribusikan di seluruh gambar.</p>
<p>Unggah gambar, pilih parameter analisis, dan lihat bagaimana setiap piksel pada peta hasil mewakili tekstur dari area lokal di sekitarnya pada gambar asli.</p>
'''
        )
        db.session.add(glcm_task)
        knn_task = Task(
            title='K-Nearest Neighbors',
            thumbnail='assets/knnthumnail.png',
            description='Klasifikasi data menggunakan algoritma KNN.',
            content='''
<p>Alat ini melakukan klasifikasi data menggunakan metode <strong>K-Nearest Neighbors (KNN)</strong>. KNN adalah algoritma supervised learning yang digunakan untuk klasifikasi dan regresi. Dalam kasus ini, kita akan menggunakannya untuk klasifikasi.</p>
<p>Unggah dataset Anda (dengan 2 fitur untuk visualisasi), tentukan jumlah tetangga (K), dan lihat bagaimana data baru diklasifikasikan berdasarkan mayoritas kelas dari K tetangga terdekatnya.</p>
'''
        )
        db.session.add(knn_task)
        nb_task = Task(
            title='Naive Bayes',
            thumbnail='assets/nbthumbnail.png',
            description='Klasifikasi data menggunakan algoritma Naive Bayes.',
            content='''
<p>Alat ini melakukan klasifikasi data menggunakan metode <strong>Naive Bayes</strong>. Naive Bayes adalah algoritma klasifikasi probabilistik berdasarkan Teorema Bayes dengan asumsi independensi yang kuat antar fitur.</p>
<p>Unggah dataset Anda (dengan 2 fitur untuk visualisasi), dan lihat bagaimana data diklasifikasikan menggunakan distribusi Gaussian.</p>
'''
        )
        db.session.add(nb_task)
        dt_task = Task(
            title='Decision Tree',
            thumbnail='assets/dtthumbnail.png',
            description='Klasifikasi data menggunakan algoritma Decision Tree (Iris Dataset).',
            content='''
<p>Alat ini mendemonstrasikan algoritma <strong>Decision Tree</strong> menggunakan dataset standar <strong>Iris</strong>.</p>
<p>Decision Tree membagi data menjadi subset yang lebih kecil berdasarkan aturan keputusan yang diturunkan dari fitur data. Visualisasi di bawah ini menunjukkan struktur pohon keputusan dan batas keputusan untuk dua fitur pertama.</p>
'''
        )
        db.session.add(dt_task)
        db.session.commit()

setup_database(app)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
