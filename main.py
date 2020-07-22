from flask import Flask, render_template, request
import pickle
app = Flask(__name__)

file = open('model.pkl','rb')
regressor = pickle.load(file)
file.close()

@app.route('/', methods = ["GET","POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        x = int(myDict["workex"])
        enter = [x]
        y_pred = regressor.predict([enter])[0]
        return render_template("show.html",inf = round(y_pred))
    return render_template("index.html")
if __name__ == '__main__':
    app.run(debug=True)

