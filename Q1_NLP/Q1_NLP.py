from gtts import gTTS
from playsound import playsound
from googletrans import Translator
from langdetect import detect
import os
import PyPDF2
import docx


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
            print(text)
    return text

# Function to extract text from Word
def extract_text_from_word(word_path):
    text = ""
    doc = docx.Document(word_path)
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def lang_detect(text):
    return detect(text)

def translate_text(text, desired_lang):
    translator = Translator()
    translated = translator.translate(text, dest=desired_lang)
    return translated.text

# Function to convert text to speech
def text_to_speech(text,desired_lang):
    tts = gTTS(text=text, lang=desired_lang)
    audio_file = 'project_updates.mp3'
    tts.save(audio_file)
    playsound(audio_file)
    os.remove(audio_file)

# Main function
def main():
    # Specify the path to your PDF or Word document
    file_path = input("Enter the path to your PDF or Word file: ")
    
    if file_path.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        text = extract_text_from_word(file_path)
    else:
        print("Unsupported file format. Please provide a PDF or Word document.")
        return
    processed_text = text.replace('\n', ' ')
    lang_detected = lang_detect(processed_text)
    print(lang_detected)
    desired_language = input("Enter the language you want the text to be read out in: ")
    translated_text_lang = translate_text(processed_text, desired_language)


    # Pass the extracted text to the TTS system
    text_to_speech(translated_text_lang, desired_language)

if __name__ == "__main__":
    main()
