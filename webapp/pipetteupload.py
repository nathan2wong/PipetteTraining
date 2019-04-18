from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory, make_response, session
from werkzeug.utils import secure_filename
import os
import datetime
import analysis
from analysis import sampleParse
from analysis import runAnalysis
import urllib

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'data')
MODEL_FOLDER = os.path.join(os.getcwd(), 'model')
ALLOWED_EXTENSIONS = set(['xlsx'])

app = Flask(__name__)
app.secret_key = os.urandom(32)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    experiments = []
    dates = []
    filenames = []
    for folder in os.listdir(app.config['UPLOAD_FOLDER']):
        datafile=""
        if folder == ".DS_Store":
            continue
        for file in os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], folder)):
            if file.endswith(".xlsx"):
                datafile = file
                break
        if datafile is not "":
            headers = {'date': folder, 'raw_excel_name': datafile}
            url = "/analysis?" + urllib.parse.urlencode(headers)
            filenames.append(datafile)
            dates.append(folder.split(".")[0])
            experiments.append(url)
    return render_template("index.html", experiments=experiments, dates=dates, filenames=filenames)
    #return redirect(url_for('upload_file'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST' and 'data' in request.files:
        raw_excel = request.files['data']
        if allowed_file(raw_excel.filename):
            raw_excel_name = secure_filename(raw_excel.filename)
            date = str(datetime.datetime.now())
            save_folder = os.path.join(app.config['UPLOAD_FOLDER'], date)
            os.makedirs(save_folder)
            raw_excel.save(os.path.join(save_folder, raw_excel_name))
            return redirect(url_for('analysis', date=date, raw_excel_name=raw_excel_name))
        else:
            flash('Invalid File')
            return redirect(request.url)
    return render_template("upload.html")

@app.route('/analysis', methods=['GET'])
def analysis():
    date = request.args.get('date')
    save_folder = os.path.join(app.config['UPLOAD_FOLDER'], date)
    name = request.args.get('raw_excel_name')

    #LinReg = sampleParse(name, save_folder)
    #lineplot = url_for('uploaded_file', filename="lineplot.png", date=date)

    LinReg, results, metadata = runAnalysis(name, save_folder, app.config['MODEL_FOLDER'])
    lineplots = []
    for key in LinReg.keys():
        img_name = key.split("_")[0] + "_" + "lineplot.png"
        lineplots.append([url_for('uploaded_file', filename=img_name, date=date), key, LinReg[key]])
    return render_template("upload.html", reg=lineplots, date=date.split(".")[0],
                           ml_results = [list(results["probabilities"])[0], list(results["class_id"])[0]],
                           metadata=metadata)

@app.route("/<path:path>")
def images(path):
    resp = make_response(open(path).read())
    resp.content_type = "image/png"
    return resp

@app.route("/images")
def uploaded_file():
    date = request.args.get('date')
    filename = request.args.get('filename')
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], date), filename)

from random import random

from bokeh.layouts import row
from bokeh.models import CustomJS, ColumnDataSource
from bokeh.plotting import figure, output_file, show
@app.route("/bokeh")
def bokeh():
    output_file("callback.html")
    x = [random() for x in range(500)]
    y = [random() for y in range(500)]
    x2 = [random() for x2 in range(100)]
    y2 = [random() for y2 in range(100)]

    # the two different data sources of different length
    s1 = ColumnDataSource(data=dict(x=x, y=y))
    s1b = ColumnDataSource(data=dict(x2=x2, y2=y2))

    # the figure with all source data where we make selections
    p1 = figure(plot_width=400, plot_height=400, tools="lasso_select", title="Select Here")
    p1.circle('x', 'y', source=s1, alpha=0.6, color='red')
    p1.circle('x2', 'y2', source=s1b, alpha=0.6, color='black')

    # second figure which is empty initially where we show the selected datapoints
    s2 = ColumnDataSource(data=dict(x=[], y=[]))
    s2b = ColumnDataSource(data=dict(x2=[], y2=[]))
    p2 = figure(plot_width=400, plot_height=400, x_range=(0, 1), y_range=(0, 1),
                tools="", title="Watch Here")
    p2.circle('x', 'y', source=s2, alpha=0.6, color='red')
    p2.circle('x2', 'y2', source=s2b, alpha=0.6, color='black')

    def attach_selection_callback(main_ds, selection_ds):
        def cb(attr, old, new):
            new_data = {c: [] for c in main_ds.data}
            for idx in new['1d']['indices']:
                for column, values in main_ds.data.items():
                    new_data[column].append(values[idx])
            # Setting at the very end to make sure that we don't trigger multiple events
            selection_ds.data = new_data

        main_ds.on_change('selected', cb)
    attach_selection_callback(s1, s2)
    attach_selection_callback(s1b, s2b)

    layout = row(p1, p2)

    show(layout)


