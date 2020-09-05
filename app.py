import flask
import pickle
import emoji
import nltk  
nltk.download('stopwords')
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer                                 # Python library for NLP
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = keras.models.load_model('bidirectionnel lstm.h5')
# Use pickle to load in the pre-trained tokinizer.
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
app = flask.Flask(__name__, template_folder='newapplication')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        stemmer = PorterStemmer()
        max_length = 19
        trunc_type='post'
        padding_type='post'
        oov_tok = "<OOV>"
        new_sample = [flask.request.form['text']]
        def filter_stop_words(train_sentences, stop_words):
            for i, sentence in enumerate(train_sentences):
                new_sent = [stemmer.stem(word) for word in sentence.split() if word not in stop_words]
                train_sentences[i] = ' '.join(new_sent)
            return train_sentences

        stop_words = set(stopwords.words("english"))
        new_sample = filter_stop_words(new_sample, stop_words)
        seq = tokenizer.texts_to_sequences(new_sample)
        padded = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        prediction=model.predict(padded)
        def f(x):
            if x>0.8:
                return emoji.emojize('Definitely Positive!:grinning_face_with_big_eyes:')
            elif 0.6<x<0.8  :
                return 'Positive'
            elif 0.5<x<0.6 :
                return emoji.emojize('Getting some mixed feelings here! :winking_face_with_tongue:')
            elif 0.1<x<0.5:
                return  'Negative'
            else :
                return 'Negativity overload!'
        return flask.render_template('result.html',
                                     
                                            
                                     result=f(prediction),
                                     proba=round(prediction[0][0]*100,2),
                                     
                                     )
        if flask.request.method == 'POST':
            return(flask.render_template('main.html'))

if __name__ == '__main__':
    app.run()


