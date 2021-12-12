import streamlit as st

from predict import Translate

# Text/Title
st.title("French 2 English Translator")

fr_sent = st.text_area("Sentence")

encoder_path = 'data/best_encoder.pt'
decoder_path = 'data/best_decoder.pt'

if st.button('Translate'):
    if fr_sent is not None:
        try:
            eng_sent = Translate(encoder_path, decoder_path).predict(fr_sent)
        except:
            eng_sent = "Some characters/words are invalid or doesn't exist in the vocabs"
        st.write(eng_sent)
    else:
        st.write("Enter valid sentence")

