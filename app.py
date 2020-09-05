import flask
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = keras.models.load_model('C:/Users/Amine/Desktop/bidirectionnel lstm.h5')
# Use pickle to load in the pre-trained tokinizer.
# loading
with open('C:/Users/Amine/Desktop/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
app = flask.Flask(__name__, template_folder='newapplication')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        
        max_length = 19
        trunc_type='post'
        padding_type='post'
        oov_tok = "<OOV>"
        new_sample = [flask.request.form['text']]
        seq = tokenizer.texts_to_sequences(new_sample)
        padded = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        prediction=model.predict(padded)
        def f(x):
            if x>0.5:
                return ' Positive! :D'
            else:
                return 'Negative :('
        return flask.render_template('result.html',
                                     
                                            
                                     result=f(prediction),
                                     proba=round(prediction[0][0]*100,2),
                                     
                                     )

if __name__ == '__main__':
    app.run()


