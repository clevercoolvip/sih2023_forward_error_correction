from flask import Flask, jsonify, render_template, request, redirect
import keras
import tensorflow as tf
app = Flask(__name__)
import numpy as np
global model
import matplotlib.pyplot as plt
import json

def bpsk_preprocessing(title):
    with open(title, "r+") as f:
        result = f.readlines()

    val = []
    for i in result:
        val.append(i[:-1])
    t = ""
    for i in range(len(val)):
        if len(val[i])<774:
            t = val[i]
            for j in range(abs(774-len(val[i]))):
                t+="0"
            val[i] = t
        if len(val[i])>774:
            val[i] = val[i][:774]
    x_pred = []
    for i in val:
        x_pred.append(list(i))
    final_data=[]
    for i in x_pred:
        final_data.append([[float(x)] for x in i])
    final_data = np.array(final_data)
    return final_data

def qpsk_8psk_preprocessing(title):
    with open(title, "r+") as f:
        result = f.readlines()

    val = []
    for i in result:
        val.append(i[:-1])
    t = ""
    for i in range(len(val)):
        if len(val[i])<200:
            t = val[i]
            for j in range(abs(200-len(val[i]))):
                t+="0"
            val[i] = t
        if len(val[i])>200:
            val[i] = val[i][:200]
    x_pred = []
    for i in val:
        x_pred.append(list(i))
    final_data=[]
    for i in x_pred:
        final_data.append([[float(x)] for x in i])
    final_data = np.array(final_data)
    return final_data

def ping(val):
    data = val.split(",")
    dat = [[x] for x in data]
    x_data = np.array(dat)
    x_data_ = np.expand_dims(x_data, axis=0)
    return x_data_

        

@app.route('/', methods=['GET'])
def landing():
    return render_template("/index.html")


@app.route('/process', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        select_val = request.form.get("modulator_type")
        print(select_val)
        f = request.files['file']
        title = f.filename
        f.save(f.filename)
        print(title)



        if select_val=="BPSK":
            class_names = ["BCH", "Hamming", "Convolutional", "Turbo", "LDPC"]
            model = tf.keras.models.load_model("model_ldpc_within_bpsk.h5")
            final_data = bpsk_preprocessing(title)
        elif select_val=="QPSK":
            class_names = ["BCH", "Convolutional", "TPC"]
            model = tf.keras.models.load_model("model_8psk.h5")
            final_data = qpsk_8psk_preprocessing(title)
        else:
            class_names = ["Convolutional", "TPC", "BCH"]
            model = tf.keras.models.load_model("model_qpsk92acc\content\model_qpsk_92acc")
            final_data = qpsk_8psk_preprocessing(title)



        print(final_data.shape)
    
        pred = model.predict(final_data)
        print("*"*12)
        print(pred)

        probab = {}
        a=0
        for i in class_names:
            probab[i] = round(pred[0][a]*100, 1)
            a+=1
        
        

        # keys = probab.keys()
        # values = probab.values()
        # plt.bar(keys, values)
        # plt.ylabel("Accuracy (%)")
        # plt.xlabel("FEC Schemes")
        # plt.savefig("default.png")

        predictions = [class_names[np.argmax(x)] for x in pred]
        val = "All derived classes: "
        with open("result.txt", "w+") as file:
            file.write(str(predictions))

        f = open("static/history.txt", "a")
        f.writelines("\n")
        f.writelines(str(predictions) + str(probab) + "\n")
        f.close()

        # return redirect("/process?preds=" + str(predictions) + "&probab=" + str(probab))
        return render_template("process.html", value=str(predictions[0]), value2=probab)
    return render_template("process.html")

@app.route('/result', methods=['GET','POST'])
def result():
    return render_template('result.html')

@app.route('/history', methods=['GET'])
def history():
    return render_template('history.html')

if __name__ == "__main__":
    app.run(debug=True)