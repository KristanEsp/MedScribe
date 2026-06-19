import numpy as np
import pandas as pd
import librosa
import joblib
import gradio as gr

#For transcription
from faster_whisper import WhisperModel
from keras.models import load_model
import soundfile as sf
import librosa
import re
from statistics import mode

#For Speaker Diarization
from resemblyzer import VoiceEncoder, preprocess_wav
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering

#For Spekear Identification (Classification)
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

#For Text Cleaning
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('stopwords')
# import import_ipynb
# from Classification import text_clean

#For SUmmarization
import os
from huggingface_hub import InferenceClient

import warnings
warnings.filterwarnings('ignore')


# # Input Audio File + Transcription
# 
# Normalization and Audio boosting will be done to help improve the embedding results of resemblyzer
# 
# The waveform of the audio file will be obtained using the Librosa library. Then the faster whisper model will be used to provide the initial transcript with some segmentation

#Convert mp files to wav since resemblyzer embedding works better with wav
def convert_to_wav(input_file, output_file):
    audio, sr = librosa.load(input_file, sr = 16000, mono = True)
    sf.write(output_file, audio, 16000)

def normalize_audio(waveform):
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak
    return waveform

def boost_audio(waveform, db):
    factor = 10 ** (db / 20) 
    boosted = waveform * factor
    boosted = np.clip(boosted, -1.0, 1.0) 
    return boosted

def import_audio_file(file):
    #Extracting audio file's waveform with librosa
    file_path = file
    waveform, sample_rate = librosa.load(file_path, res_type = "kaiser_fast", sr = None)
    return (waveform, sample_rate)

#Transcribe text using faster whisper
if torch.cuda.is_available():
    device = "cuda" 
    compute_type = "float16"
else:
    device = "cpu"
    compute_type = "int8"
whisper_model = WhisperModel("medium.en", device = device, compute_type = compute_type)

def transcribe_text(file, whisper_model):
    #Initiate whisper model
    segments, _ = whisper_model.transcribe(file,
                                           vad_filter = True,
                                           word_timestamps = True, 
                                           beam_size = 8)
    return segments
    

# # Speaker Diarization
# 
# Speaker Diarization/Differentiation will be done by doing the following steps
# 
# 1.) Using resemblyzer to embedd each spoken word from the audio. This embedding is the extracted features, 
#     represetning the voice characteristics of each spoken word and will be used to differentiate between speakers
# 
# 2.) Performing Clustering on the extracted embedded features

# ## Extracting embedding features
def extract_embeddings(segments, waveform, sample_rate):
    #Generate embeddings for each audio word usig resemblyzer
    print("Extracting embeddings, Please Wait...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = VoiceEncoder(device = device)
    embeddings = []
    complete_transcript = []
    start_times = []
    end_times = []
    
    for segment in segments:
        #Extracting embeddings for each individual words
        for word in segment.words:   
            #Get the start and end index of the segment
            start_index = int(word.start * sample_rate)
            end_index = int(word.end * sample_rate)

            start_times.append(word.start)
            end_times.append(word.end)
            
            #Get the complete waveform of the current segment
            segment_waveform = waveform[start_index:end_index]
        
            #Extract embeddings
            segment_embeddings = encoder.embed_utterance(segment_waveform)
            embeddings.append(segment_embeddings)
    
            complete_transcript.append(word.word)

    #Fail check
    if len(embeddings) == 0:
        raise ValueError("Failed to detect any voices. Please upload a new audio file")

    return embeddings, complete_transcript, start_times, end_times


# ## Clustering using Agglomerative Clustering
# 
# Agglomerative clustering was the chosen clustering technique because of its advantage of being able to generate clusters without knowing the K value. 
# This is important as an audio file may contain an unknown number of speakers.

#Setting the distance threshold automatically 
def calculate_threshold(embeddings):  
    Z = linkage(embeddings, method = 'ward', metric = 'euclidean')
    distances = Z[:, 2]
    differences = np.diff(distances)
    
    #getting the index with the largest difference
    optimal_index = np.argmax(differences)
    
    #Selecting the threshold based on the index with the largest difference
    buffer = 0.5 #Add some buffer to help avoid over-splitting (having more clusters)
    threshold = distances[optimal_index]
    threshold = threshold + buffer

    return threshold

def apply_clustering(embeddings, threshold):
    agglo_cluster = AgglomerativeClustering(n_clusters = None, #Setting to None because using the calculated distance threhsold instead
                                            metric = "euclidean",
                                            linkage = "ward",
                                            distance_threshold = threshold)
    
    cluster_pred = agglo_cluster.fit_predict(embeddings)
    return cluster_pred

#Store the transcript and cluster prediction to a df
def create_cluster_df(cluster_pred, complete_transcript, start_times, end_times):
    df_word_transcript = pd.DataFrame(columns = ["Transcripts", "Prediction", "StartTime", "EndTime"])
    df_word_transcript["StartTime"] = start_times
    df_word_transcript["EndTime"] = end_times
    df_word_transcript["Prediction"] = cluster_pred
    df_word_transcript["Transcripts"] = complete_transcript

    return df_word_transcript

# ## Converting the transcript from word per word back to sentence form

def recreate_sentences(df_word_transcript):
    sentences = []
    current_words = []
    current_predictions = []
    sentence_start_time = None
    sentence_end_time = None
    
    #Iterate through each row of the original dataframe
    for indx, row in df_word_transcript.iterrows():
        word = row["Transcripts"]
        pred = row["Prediction"]
        start_time = row["StartTime"]
        end_time = row["EndTime"]
    
        #Append the current word 
        current_words.append(word)
        current_predictions.append(pred)

        #Record start time if the word is the first word of a sentence
        if sentence_start_time is None:
            sentence_start_time = start_time
            
        #Start new sentence if the current word contains a punctuation
        if re.search(r"[.!?]", word):
        
            #Exclude title words such as Mr. Ms. Mrs. Dr.
            if word in [" Mr.", " Ms.", " Mrs.", " Dr."]:
                continue
    
            #Form the current sentence
            sentence_final = " ".join(current_words)
            #Using mode to obtain the final prediction of which cluster spoke the sentence
            prediction_final = mode(current_predictions)

            #Record the end time of the last word of the sentence
            sentence_end_time = end_time
    
            sentences.append({
                "Speaker": prediction_final,
                "Transcripts": sentence_final,
                "StartTime": sentence_start_time,
                "EndTime": sentence_end_time
            })
    
            #Reseting for the next sentence
            current_words = []
            current_predictions = []
            sentence_start_time = None
            sentence_end_time = None
    
    df_sentence_transcript = pd.DataFrame(sentences)
    return(df_sentence_transcript)

# # Speaker Identification
# 
# Labelling which cluster is the doctor and patient using the trained Bidirectional-LSTM model from the Classification notebook

#Combine all transcripts within each clusters into a single string
def string_cluster(df_sentence_transcript):
    cluster_transcripts = {}
    for cluster in df_sentence_transcript["Speaker"].unique():
        cluster_transcripts[cluster] = ""
    
    for idx, row in df_sentence_transcript.iterrows():
        current_label = row["Speaker"]
        cluster_transcripts[current_label] += row["Transcripts"] + ""
    
    #Put the combined trnascripts into a df
    cluster_transcripts = list(cluster_transcripts.items())
    df_prediction = pd.DataFrame(cluster_transcripts, columns = ["cluster", "transcript"])
    return(df_prediction)

### Text Cleaning
def text_clean(text):
    #Lower Casing
    text_clean = text.lower()
    
    #Remove punctuations
    punctuations = r"[^\w\s]"
    text_clean = re.sub(punctuations, " ", text_clean)

    #Tokenization
    text_clean = word_tokenize(text_clean)

    #Remove stop words
    stop_words = set(stopwords.words("english"))
    text_clean = [word for word in text_clean if word not in stop_words]

    #Lemmatization
    wn = nltk.WordNetLemmatizer()
    text_lemmatized = []
    for word in text_clean:
        text_lemmatized.append(wn.lemmatize(word))

    #Remove filler words like "um" "like" and basic greetings like "hi", "hey", "hello"
    filtered_list = []
    for token in text_lemmatized:
        if token in ["um", "like", "uhm", "uh", "hi", "hey", "hello"]:
            continue
        #also remove digits
        if token.isdigit():
            continue
            
        filtered_list.append(token)

    text_clean = filtered_list

    #Concat all tokens into a single string
    text_clean = " ".join(text_clean)

    return text_clean

#Perform classification using the trained LSTM model on each string cluster
def label_speaker(row):
    #Create a new dataframe
    le = joblib.load("/data/SavedModels/LabelEncoder.bin")
    lstm_model = load_model("/data/SavedModels/lstm_model.keras")
    tokenizer = joblib.load("/data/SavedModels/Tokenizer.bin")
    
    #Perform the text processing
    pred_text = text_clean(row["transcript"])
    
    #Preprocess the text
    pred_seq = tokenizer.texts_to_sequences([pred_text])
    max_length = 500
    pred_padded = pad_sequences(pred_seq, maxlen = max_length, padding ='post')
    
    #Predict with the lstm model
    final_prediction = lstm_model.predict(pred_padded)
    #Also get the prediction probability incase that no doctor was predicted
    prediction_prob = float(final_prediction[0][0])
    final_prediction = (final_prediction > 0.5).astype("int")[0]
    
    row["SpeakerLabel"] = le.inverse_transform(final_prediction)[0]
    row["PatientPredictProbability"] = prediction_prob
    
    return row

#At least one doctor and one patient must be identified.
def detect_missing_identities(df):
    #In case no doctor was idenfied, the cluster with lowest patient probability will be labelled as doctor
    if "Doctor" not in df["SpeakerLabel"].values:
        #Pick the row with the lowest patient probability
        chosen_row = 0
        lowest_probability = 1
        for idx, row in df.iterrows():
            if row["PatientPredictProbability"] < lowest_probability:
                lowest_probability = row["PatientPredictProbability"]
                chosen_row = idx
        #Change the chosen row speaker into doctor
        df.loc[chosen_row, "SpeakerLabel"] = "Doctor"
    
    #In case of no patient, identify highest patient probability
    if "Patient" not in df["SpeakerLabel"].values:
        #Pick the row with the lowest patient probability
        chosen_row = 0
        highest_probability = 0
        for idx, row in df.iterrows():
            if row["PatientPredictProbability"] > highest_probability:
                highest_probability = row["PatientPredictProbability"]
                chosen_row = idx
        #Change the chosen row speaker into Patient
        df.loc[chosen_row, "SpeakerLabel"] = "Patient"

    return df
    

#Multiple Patients/Doctors will be numbered e.g. Patient 1, Patient 2
def detect_multiple_identities(df):
    freq = df["SpeakerLabel"].value_counts()
    counts = {}
    counted_labels = []
    
    for label in df["SpeakerLabel"]:
        #Only count multiple identities
        if freq[label] > 1:
            counts[label] = counts.get(label, 0) + 1
            counted_labels.append(f"{label} {counts[label]}")
        else:
            counted_labels.append(label)
    
    df["SpeakerLabel"] = counted_labels
    return df

#Replace the unlabelled clusters in the df_sentence_transcript with the predicted labels
def replace_labels(df_sentence_transcript, df_prediction):
    labelled_map = dict(zip(df_prediction["cluster"], df_prediction["SpeakerLabel"]))
    df_sentence_transcript["Speaker"] = df_sentence_transcript["Speaker"].map(labelled_map)
    return df_sentence_transcript

# # Text Cleaning Function
def text_clean(text):
    #Lower Casing
    text_clean = text.lower()
    
    #Remove punctuations
    punctuations = r"[^\w\s]"
    text_clean = re.sub(punctuations, " ", text_clean)

    #Tokenization
    text_clean = word_tokenize(text_clean)

    #Remove stop words
    stop_words = set(stopwords.words("english"))
    text_clean = [word for word in text_clean if word not in stop_words]

    #Lemmatization
    wn = nltk.WordNetLemmatizer()
    text_lemmatized = []
    for word in text_clean:
        text_lemmatized.append(wn.lemmatize(word))

    #Remove filler words like "um" "like" and basic greetings like "hi", "hey", "hello"
    filtered_list = []
    for token in text_lemmatized:
        if token in ["um", "like", "uhm", "uh", "hi", "hey", "hello"]:
            continue
        #also remove digits
        if token.isdigit():
            continue
            
        filtered_list.append(token)

    text_clean = filtered_list

    #Concat all tokens into a single string
    text_clean = " ".join(text_clean)

    return text_clean


# # Wrapping to one function
def med_scribe(file_path, num_speakers = None): #num_speakers = None if unknown

    #Convert to wav
    convert_to_wav(file_path, "output.wav")

    #Extract waveform and sample rate
    waveform, sample_rate = import_audio_file("output.wav")
    waveform = normalize_audio(waveform)
    waveform = boost_audio(waveform, db = 6)

    #Perform transcription
    print("Performing Transcription...")
    segments = transcribe_text(file_path, whisper_model)

    ###Perform Speaker Diarization
    embeddings, complete_transcript, start_times, end_times = extract_embeddings(segments, waveform, sample_rate)
    print("Done Extracting Embeddings")
    print("Performing Speaker Diarization...")
    threshold = calculate_threshold(embeddings)
    cluster_pred = apply_clustering(embeddings, threshold)
    df_word_transcript = create_cluster_df(cluster_pred, complete_transcript, start_times, end_times)

    ###Perform Speaker Identification
    print("Performing Speaker Identification...")
    df_sentence_transcript = recreate_sentences(df_word_transcript)
    df_prediction = string_cluster(df_sentence_transcript)
    df_prediction = df_prediction.apply(label_speaker, axis = 1)

    ###Check for any inconsistencies such as missing doctor/patient identities and multiple identities
    df_prediction = detect_missing_identities(df_prediction)
    df_prediction = detect_multiple_identities(df_prediction)

    #Wrap to one final df
    df_sentence_transcript = replace_labels(df_sentence_transcript, df_prediction)
    
    return df_sentence_transcript


# # GUI w/ Gradio

#css styling
css = """
/* Gr.Audio component styling */
.hf-audio {
    background: rgba(0, 123, 255, 0.15) !important;
    border-radius: 14px !important;
    padding: 20px !important;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25) !important;
    border: 1px solid #333 !important;
}
.hf-audio label {
    color: rgba(0, 123, 255, 1.0) !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

/* Increasing the size of the timestamps */
#time.svelte-1ffmt2w,
#duration.svelte-1ffmt2w {
    font-size: 22px !important;
    font-weight: 600 !important;
    padding-top: 12px !important;
    border-radius: 8px !important;
}


.hf-audio button {
    background-color: #1e1e1e !important;   
    border-color: #1e1e1e !important;       
    color: white !important;               
}



/* Transcription box */
.transcript-box * {
    font-size: 22px !important;
    line-height: 1.65 !important;
    padding: 20px !important;
    background: rgba(0, 123, 255, 0.15) !important;
    border-radius: 10px !important;
}

/* Summary box */
.summary-box * {
    font-size: 1.25rem;
    line-height: 1.5 !important;
    padding: 1px !important;
    background: rgba(0, 123, 255, 0.15) !important;
    border-radius: 10px !important;
    width: 100% !important;
    max-width: 100% !important;
    flex-grow: 2.5 !important;
}



/* Instruction box */
.instruction-box * {
    font-size: 15px !important;
    line-height: 1.5 !important;
    padding: 1px !important;
    background: rgba(0, 0, 0, 0.05) !important;
    border-radius: 10px !important;
}



"""


# ## Helper Functions

#Using Meta's Llama LLM model to summarize text
def summarize_text(full_transcript):
    #Get inference client using my HF token
    client = InferenceClient(api_key = os.environ.get("HF_TOKEN"))

    #Use model to generate summary based on the prompt
    response = client.chat.completions.create(
    model = "meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a clinical summarization assistant."},
            {"role": "user", "content": f"""
    Generate a SOAP-style clinical summary based on the given Conversation between Doctor and Patient.
    
    S (Subjective): Summarize the patient's subjective concerns, symptoms, context, and relevant history.
    O (Objective): Extract objective findings mentioned in the transcript (vitals, observations, exam findings, measurable data).
    A (Assessment): Provide a concise assessment of the patient's condition based only on the transcript.
    P (Plan): Suggest an appropriate plan or next steps based strictly on what was discussed.
    
    If a section has no information, write "None reported." DO not include the instructions in the output.
    Format the final answer exactly as:
    # S (Subjective):
    [subjective content]
    
    # O (Objective):
    [objective content]
    
    # A (Assessment):
    [assessment content]
    
    # P (Plan):
    [plan content]
    
    Conversation:
    {full_transcript}
    
    Clinical Summary:
    """}
        ],
        max_tokens = 300
    )

    return (response.choices[0].message.content)

#Make the re-upload button apear when an audio file is currently uploaded
def show_reupload(audio):
    return gr.update(visible = audio)

def print_transcripts(audio):
    if audio is None:
        return "", "", gr.update(visible = False)

    #Process the audio to get the dataframe transcript
    df_transcript = med_scribe(audio)

    #Get each line one by one
    lines = []
    for idx, row in df_transcript.iterrows():
        #Get the start time of the sentence
        sentence_start_time = round(row["StartTime"])
        minutes = sentence_start_time // 60
        seconds = sentence_start_time % 60
        
        current_line = f"[{minutes}:{seconds:02d}] {row['Speaker']}: {row['Transcripts']}"
        lines.append(current_line)

    lines = "\n\n".join(lines)

    #Get the summary
    summary = summarize_text(lines)
    summary = f"**Summary:** \n\n {summary}"
    
    return lines, summary, gr.update(visible = audio)


# ## Gradio

# Main GUI
with gr.Blocks(title = "MedScribe", css = css) as demo:
    ### A heading for the title of the application
    gr.HTML("""
    <div style = "
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
        padding: 0 4px 20px 4px;
        line-height: 1.45;
    ">
        <h1 style = "
            font-size: 35px;
            font-weight: 700;
            margin: 0 0 6px 0;
            display: inline-flex;
            align-items: center;
            gap: 10px;
        ">
            <span>🏥 Medscribe</span>
        </h1>
        <div style = 
            "font-size: 16px; 
             opacity: 0.7; 
             margin-bottom: 12px;
        ">
            Transcribes doctor-patient audio consultations with speaker diarization and speaker identification. <br>
            Generates a summary of the transcript using Llama-3-8B-Instruct
        </div>
    """)

    #Audio Section
    with gr.Row():
        with gr.Column(scale = 1):
            audio = gr.Audio(type = "filepath",
                    label = "Audio File",
                    sources = ["upload"], #Only audio import no mic
                    interactive = True,
                    editable = False,
                    scale = 1,
                    elem_classes = ["hf-audio "]
                    )

            #Button to re-upload
            reupload_button = gr.ClearButton(components = [audio], 
                                             value = "Re-upload Audio",
                                             visible = False,
                                             elem_classes = ["hf-audio "])
            
            #An example audio files that users can click
            gr.Examples(
                examples = [  
                    ["/data/SampleAudio/Derma_10min.mp3"],
                    ["/data/SampleAudio/Cardio_10min.mp3"],
                    ["/data/SampleAudio/Urinary_8min.mp3"],
                    ["/data/SampleAudio/MusculoSkeletal_10min.mp3"],
                    ["/data/SampleAudio/Resp_13min.mp3"],
                    ["/data/SampleAudio/YT_GPBehindDoors_8min.mp3"],
                ],
                inputs = audio,
                label = "Quick Audio Examples - Click to load"
            )

            #Description about the application
            gr.Markdown(
                """
                <div>
                    <b>Note:</b>
                    <ul>
                        <li>Uploaded audio file must be at least 3 minutes long to provide enough data for speaker diarization and identification</li>
                        <li>Works best with 2 speakers (1 doctor & 1 patient)</li>
                        <li>Voices must be audible with good volume</li>
                        <li>A 10 minute audio file takes ~1-2 minutes to process</li>
                        <em>* You can find other doctor-patient audio samples from: 
                            <a href=https://www.kaggle.com/datasets/najamahmed97/audio-recording-whisper?resource=download>https://www.kaggle.com/datasets/najamahmed97/audio-recording-whisper?resource=download</a>
                        </em></br>
                        <b>Disclamer: Transcription and summarization may be inaccurate. Proper validation and expert review are necessary for medical applications </b>
                    </ul>
                </div>
                """,
                elem_classes = ["instruction-box"]
            )
        #Transcript Section
        with gr.Column(scale = 2):
            transcript_markdown = gr.Markdown(label = "Transcript", 
                                              height = 1000,
                                              elem_classes = ["transcript-box"])


        #Summary Section
        with gr.Column(scale = 1):
            summary_markdown = gr.Markdown(label = "Summary", 
                                           height = 750,
                                           elem_classes = ["summary-box"])
            
        audio.change(print_transcripts, 
                     inputs = audio, 
                     outputs = [transcript_markdown, summary_markdown, reupload_button])

#Launch the GUI
demo.launch(share = True, debug = True, inbrowser = True, allowed_paths = ["/data/SampleAudio"])