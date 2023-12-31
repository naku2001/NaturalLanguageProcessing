#Imports
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load the model and tokenizer
model = load_model('best_model2.h5',compile=False)
tokenizer = pickle.load(open('tokenizerr.pkl', 'rb'))




def predict_next_words(model, tokenizer, text, num_words=1):
    """
    Predict the next set of words using the trained model.

    Args:
    - model (keras.Model): The trained model.
    - tokenizer (Tokenizer): The tokenizer object used for preprocessing.
    - text (str): The input text.
    - num_words (int): The number of words to predict.

    Returns:
    - str: The predicted words.
    """
    for _ in range(num_words):
        # Tokenize and pad the text
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = pad_sequences([sequence], maxlen=5, padding='pre')

        # Predict the next word
        predicted_probs = model.predict(sequence, verbose=0)
        predicted = np.argmax(predicted_probs, axis=-1)

        # Convert the predicted word index to a word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        # Append the predicted word to the text
        text += " " + output_word

    return ' '.join(text.split(' ')[-num_words:])



def main():

    
    

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:green ;padding:10px">
    <h2 style="color:white;text-align:center;">Shona Prediction App</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.title("Predict next few words ........")
    user_input = st.text_input("Enter any five words: ")
    lst = list(user_input.split())

            

    if st.button("Generate"):
        
        if (user_input is not None and len(lst)==5):
        
            result =  predict_next_words(model, tokenizer, user_input, num_words=1)
            st.success(result)
        
        else:
            st.write("Please enter five words only")
        
        


if __name__ == '__main__':
    main()
