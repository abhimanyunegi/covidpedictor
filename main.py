from flask import Flask, render_template, request
app = Flask(__name__)
import pickle


# open a file, where you stored the pickled data
file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['temp'])
        pain = int(myDict['pain'])
        runnyNose = int(myDict['cold'])
        diffBreath = int(myDict['bre'])
        # Code for inference
        inputFeatures = [ pain, runnyNose, fever, diffBreath]
        infProb =clf.predict_proba([inputFeatures])
        r = clf.predict([inputFeatures])
        print(infProb)
        return render_template('show.html', inf=round(infProb[0][1]*100,2))

        
    return render_template('index.html')
    # return 'Hello, World!' + str(infProb)


if __name__ == "__main__":
    app.run(debug=True)