
import streamlit as st
import pickle
import cv2
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.preprocessing import image

#Loading the Inception model
model_path = r"C:\Users\LENOVO\Downloads\Documents\model.h5"
model= load_model(model_path,compile=(False))
word_to_index = pickle.load(open(r"C:\Users\LENOVO\Downloads\Documents\saved_toix.pkl", 'rb'))
index_to_word = pickle.load(open(r"C:\Users\LENOVO\Downloads\Documents\saved_ixtoword.pkl", 'rb'))
max_length = 29

model_new = load_model(r"C:\Users\LENOVO\Downloads\Documents\inceptionv3.h5")


#Functions
def splitting(name):
    vidcap = cv2.VideoCapture(name)
    success,image = vidcap.read()
    count = 0
    while count:
        cv2.imwrite(r"C:\Users\maban\Documents\Assignment\Frames\frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        #print('Read a new frame: ', success)
        count += 1
        
    #preprocessing()

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
    encode(x)

def encode(image):
    #image = preprocess(image) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec
    
def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [word_to_index[w] for w in in_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_word[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    st.write(final)


def beam_search_predictions(image, beam_index = 3):
    start = [word_to_index["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = model.predict([image,par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [index_to_word[i] for i in start_word]
    final_caption = []

    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    st.write(final_caption)


def main():
    st.title("Computer Vision,Deep Learning Model.")
    
    file = st.file_uploader("Upload Image",type=(['jpeg']))
    if file is not None:
        image = file
        '''path = file.name
        with open(path,mode='wb') as f: 
          f.write(file.read())         
        st.success("Saved File")
        
        video_file = open(path, "rb").read()

        st.video(video_file)'''
        
        if st.button("Detect"):
            st.image(image)
            fea = preprocess(image)
            fea = np.reshape(fea,(1,2048))
            greedySearch(fea)
            beam_search_predictions(fea, beam_index = 2)
            beam_search_predictions(fea, beam_index = 7)
            # output1 = splitting(path)
            # output2 = preprocessing()
            # output = predict(output2)
        
            # #st.success('The Output is {}'.format(output))
            # st.success(output)

        
if __name__=='__main__':
    main()
