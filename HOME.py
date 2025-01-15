import streamlit as st
import pandas as pd
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandasai"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain_groq"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit_extras"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "whisper"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import warnings
import pydeck as pdk
from functools import partial
import time
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.dataframe_explorer import dataframe_explorer
import whisper
from transformers import pipeline
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="AI Customer Analyzer", initial_sidebar_state="collapsed")

# Loading environment variables from .env file
load_dotenv()

@st.cache_resource(show_spinner=False)
def classeimp():
    classifier = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion',return_all_scores=False)
    return classifier

@st.cache_resource(show_spinner=False)
def pipeimp():
    pipe = pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    return pipe

classifier = classeimp()
pipe = pipeimp()

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def app(data):
    if 'df' not in st.session_state:
        st.session_state.df = data

def dtf(data):
    filtered_df = dataframe_explorer(data, case=False)
    st.dataframe(filtered_df, use_container_width=True)

@st.cache_data(show_spinner=False)
def sentfunc(data):
    data["Label"] = data["Comment"].apply(lambda x: pipe(x)[0]["label"].capitalize())
    data["Emotion"] = data["Comment"].apply(lambda x: classifier(x)[0]["label"].capitalize())
    labelavg = int(round(data["Label"].value_counts()[0] / len(data), 2) * 100)
    topemo = pd.DataFrame(data["Emotion"].value_counts()).reset_index()["index"][0]
    topemo = emotion_emoji[topemo]
    emoavg = int(((pd.DataFrame(data["Emotion"].value_counts()).reset_index()["Emotion"][0]) / len(
        data)) * 100)


    st.write("")
    st.markdown(""" 
                                            <style> 
                                                @import url("https://fonts.googleapis.com/css2?family=Besley:ital,wght@0,400..900;1,400..900&family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..900;1,6..96,400..900&family=Cinzel:wght@400..900&family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" );
    
                                                .sent0-font {
                                                    font-size: 20px;
                                                    font-family:  "Poppins", sans-serif;
                                                    color: white;
                                                    font-weight: 300;
                                                    padding-top: -2.5rem;
                                                    margin-top: -2.5rem;
                                                } 
                                            </style> 
                                        """, unsafe_allow_html=True)

    st.markdown(
    '<p class="sent0-font" style="text-align:center;">Average Sentiment</p>',
    unsafe_allow_html=True)
    st.markdown(""" 
                                                                <style> 
                                                                    @import url("https://fonts.googleapis.com/css2?family=Besley:ital,wght@0,400..900;1,400..900&family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..900;1,6..96,400..900&family=Cinzel:wght@400..900&family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" );
    
                                                                    .numb-font {
                                                                        font-size: 35px;
                                                                        font-family:  "Poppins", sans-serif;
                                                                        color: white;
                                                                        font-weight: 500;
                                                                        padding-top: -3rem;
                                                                        margin-top: -3rem;
                                                                    } 
                                                                </style> 
                                                            """, unsafe_allow_html=True)
    topemo_font_style = f'<span class="numb-font">{topemo}</span>'
    emoavg_font_style = f'<span class="numb-font">{emoavg}%</span>'
    lab_font_style = f'<span class="numb-font">{labelavg}%</span>'

    st.markdown(""" 
                                            <style> 
                                                @import url("https://fonts.googleapis.com/css2?family=Besley:ital,wght@0,400..900;1,400..900&family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..900;1,6..96,400..900&family=Cinzel:wght@400..900&family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" );
    
                                                .sent-font {
                                                    font-size: 20px;
                                                    font-family:  "Poppins", sans-serif;
                                                    color: white;
                                                    font-weight: 300;
                                                    padding-top: -3.5rem;
                                                    margin-top: -3.5rem;
                                                } 
                                            </style> 
                                        """, unsafe_allow_html=True)

    st.markdown(
    f'<p class="sent-font" style="text-align:center;">{lab_font_style} Positive </p>',
    unsafe_allow_html=True
    )

    st.markdown(""" 
                                                                <style> 
                                                                    @import url("https://fonts.googleapis.com/css2?family=Besley:ital,wght@0,400..900;1,400..900&family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..900;1,6..96,400..900&family=Cinzel:wght@400..900&family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" );
    
                                                                    .sent1-font {
                                                                        font-size: 20px;
                                                                        font-family:  "Poppins", sans-serif;
                                                                        color: white;
                                                                        font-weight: 300;
                                                                        padding-top: -2rem;
                                                                        margin-top: -2rem;
                                                                    } 
                                                                </style> 
                                                            """, unsafe_allow_html=True)

    st.markdown(
    '<p class="sent1-font" style="text-align:center;">Average Emotion</p>',
    unsafe_allow_html=True)
    st.markdown(""" 
                                                                                    <style> 
                                                                                        @import url("https://fonts.googleapis.com/css2?family=Besley:ital,wght@0,400..900;1,400..900&family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..900;1,6..96,400..900&family=Cinzel:wght@400..900&family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" );
    
                                                                                        .sent2-font {
                                                                                            font-size: 20px;
                                                                                            font-family:  "Poppins", sans-serif;
                                                                                            color: white;
                                                                                            font-weight: 300;
                                                                                            padding-top: -2rem;
                                                                                            margin-top: -2rem;
                                                                                        } 
                                                                                    </style> 
                                                                                """, unsafe_allow_html=True)
    st.markdown(
    f'<p class="sent2-font" style="text-align:center;">{topemo_font_style} for {emoavg_font_style} clients</p>',
    unsafe_allow_html=True
    )
    return data
# Function to chat with CSV data
@st.cache_data(show_spinner=False)
def chat_with_csv(df, query):
    # Loading environment variables from .env file
    load_dotenv()

    # Function to initialize conversation chain with GROQ language model
    groq_api_key = os.environ['GROQ_API_KEY']

    # Initializing GROQ chat with provided API key, model name, and settings
    llm = ChatGroq(
        groq_api_key=groq_api_key, model_name="llama3-70b-8192",
        temperature=0.4)
    # Initialize SmartDataframe with DataFrame and LLM configuration
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    # Chat with the DataFrame using the provided query
    result = pandas_ai.chat(query)
    return result
@st.cache_data
def mapdash1(df):
    layer = pdk.Layer(
        'ScatterplotLayer',
        df,
        get_position=['Longitude', 'Latitude'],
        auto_highlight=True,
        get_radius=10000,
        get_fill_color=[102, 144, 183],  # Adjust color as needed
        pickable=True
    )

    # Set the initial view state
    view_state = pdk.ViewState(
        longitude=12.4964,
        latitude=41.9028,
        zoom=3,
        pitch=20,
    )

    # Create the DeckGL chart
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
    )

    # Display the DeckGL chart
    st.pydeck_chart(deck)


def click_button():
    st.session_state.clicked = True



@st.cache_data
def engrating(df):
    df["ExperienceRating"] = df["ExperienceRating"].astype(int)
    experience_mean = df["ExperienceRating"].mean()
    st.markdown(
        """
        <style> 
            @import url("https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,400;0,700;0,900&display=swap");
            
            .tt-rating {
                font-size: 20px;
                font-family: "Poppins", sans-serif;
                color: white;
                font-weight: 300;
                text-align: center;
            }

        </style> 
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f'<p class="tt-rating">Average Experience Rating</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style> 
            @import url("https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,400;0,700;0,900&display=swap");

            .mean-rating {
                font-size: 50px;
                font-family: "Poppins", sans-serif;
                color: white;
                font-weight: 600;
                text-align: center;
                padding-top: -1rem;
                margin-top: -1rem;
            }

        </style> 
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f'<p class="mean-rating">{experience_mean:.2f}/5</p>',
        unsafe_allow_html=True
    )

st.markdown(""" 
            <style> 
                @import url("https://fonts.googleapis.com/css2?family=Besley:ital,wght@0,400..900;1,400..900&family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..900;1,6..96,400..900&family=Cinzel:wght@400..900&family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" );

                .title-font {
                    font-size: 70px;
                    font-family:  "Poppins", sans-serif;
                    color: white;
                    font-weight: 800;
                    padding-top: -5rem;
                    margin-top: -5rem;
                } 
            </style> 
        """, unsafe_allow_html=True)

st.markdown(
    '<p class="title-font" style="text-align:center;">AI Customer Feedback Analyzer</p>',unsafe_allow_html=True)

#Reduce whitespace on top
st.markdown("""
<style>

.block-container
{
    padding-top: -3rem;
    padding-bottom: 1rem;
    margin-top: 1.2rem;
    padding-left: 2rem;
    margin-left: 2rem;
    padding-right: 2rem;
    margin-right: 2rem;
}

</style>
""", unsafe_allow_html=True)

box0,box1,box2 = st.columns((3,5,3))
dash0, dash1, dash2, dash3, dash4, dash5, dash6, dash7, dash8 = st.columns((2,30,1,30,1,30,1,30,2))
bot0,bot1,bot2,dat0, dat1 = st.columns((0.6,30,0.6,17,0.6))

# Upload multiple CSV files
with box1:
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    with st.container(border = True, height = 220):
        st.markdown(""" 
                    <style> 
                        @import url("https://fonts.googleapis.com/css2?family=Besley:ital,wght@0,400..900;1,400..900&family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..900;1,6..96,400..900&family=Cinzel:wght@400..900&family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" );

                        .upload-font {
                            font-size: 30px;
                            font-family:  "Poppins", sans-serif;
                            color: white;
                            font-weight: 400;
                            padding-top: 0rem;
                            margin-top: 0rem;
                        } 
                    </style> 
                """, unsafe_allow_html=True)

        st.markdown(
            '<p class="upload-font" style="text-align:center;">Upload your CSV files</p>',
            unsafe_allow_html=True)
        input_csvs = st.file_uploader("",type=['csv'], accept_multiple_files=True)

    if input_csvs:
        st.markdown(""" 
                    <style> 
                        @import url("https://fonts.googleapis.com/css2?family=Besley:ital,wght@0,400..900;1,400..900&family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..900;1,6..96,400..900&family=Cinzel:wght@400..900&family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" );
        
                        .upload-font {
                            font-size: 30px;
                            font-family:  "Poppins", sans-serif;
                            color: white;
                            font-weight: 400;
                            padding-top: 0rem;
                            margin-top: 0rem;
                        } 
                    </style> 
                """, unsafe_allow_html=True)

        st.markdown(
            '<p class="upload-font" style="text-align:center;">Upload successful.</p>',
            unsafe_allow_html=True)
        selected_file = st.selectbox("", [file.name for file in input_csvs])
        selected_index = [file.name for file in input_csvs].index(selected_file)
        data = pd.read_csv(input_csvs[selected_index])
        st.dataframe(data.head(10), use_container_width=True)
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")

        with dash1:
            st.markdown(""" 
                                        <style> 
                                            @import url("https://fonts.googleapis.com/css2?family=Besley:ital,wght@0,400..900;1,400..900&family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..900;1,6..96,400..900&family=Cinzel:wght@400..900&family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" );

                                            .ttdash-font {
                                                font-size: 25px;
                                                font-family:  "Poppins", sans-serif;
                                                color: white;
                                                font-weight: 600;
                                                padding-top: -2rem;
                                                margin-top: -2rem;
                                                padding-bottom: -1rem;
                                                margin-bottom: -1rem;
                                            } 
                                        </style> 
                                    """, unsafe_allow_html=True)

            st.markdown(
                '<p class="ttdash-font" style="text-align:center;">Regional Data</p>',
                unsafe_allow_html=True)

            with st.container(border = True, height = 200):
                mapdash1(data)



        with dash3:
            st.markdown(""" 
                        <style> 
                            @import url("https://fonts.googleapis.com/css2?family=Besley:ital,wght@0,400..900;1,400..900&family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..900;1,6..96,400..900&family=Cinzel:wght@400..900&family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" );

                            .ttdash-font {
                                font-size: 25px;
                                font-family:  "Poppins", sans-serif;
                                color: white;
                                font-weight: 600;
                                padding-top: 0rem;
                                margin-top: 0rem;
                            } 
                        </style> 
                    """, unsafe_allow_html=True)

            st.markdown(
                '<p class="ttdash-font" style="text-align:center;">Engagement Rating</p>',
                unsafe_allow_html=True)

            with st.container(border=True, height=200):
                engrating(data)



        with dash5:
            st.markdown(""" 
                        <style> 
                            @import url("https://fonts.googleapis.com/css2?family=Besley:ital,wght@0,400..900;1,400..900&family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..900;1,6..96,400..900&family=Cinzel:wght@400..900&family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" );

                            .ttdash-font {
                                font-size: 25px;
                                font-family:  "Poppins", sans-serif;
                                color: white;
                                font-weight: 600;
                                padding-top: 0rem;
                                margin-top: 0rem;
                            } 
                        </style> 
                    """, unsafe_allow_html=True)

            st.markdown(
                '<p class="ttdash-font" style="text-align:center;">Sentiment Overtime</p>',
                unsafe_allow_html=True)
            with st.container(border=True, height=200):
                if not st.session_state.clicked:
                    with st.empty():
                        button = st.button('Try out our Sentiment AI', on_click=partial(click_button))
                if st.session_state.clicked:
                    with st.spinner('Wait for it...'):
                        emotion_emoji = {
                            "Joy": "😊",
                            "Anger": "🤬",
                            "Sadness": "😭",
                            "Fear": "😨",
                            "Surprise": "😯"
                        }
                        sentfunc(data)

        with dash7:
            st.markdown(""" 
                        <style> 
                            @import url("https://fonts.googleapis.com/css2?family=Besley:ital,wght@0,400..900;1,400..900&family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..900;1,6..96,400..900&family=Cinzel:wght@400..900&family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" );

                            .ttdash-font {
                                font-size: 25px;
                                font-family:  "Poppins", sans-serif;
                                color: white;
                                font-weight: 600;
                                padding-top: 0rem;
                                margin-top: 0rem;
                            } 
                        </style> 
                    """, unsafe_allow_html=True)

            st.markdown(
                '<p class="ttdash-font" style="text-align:center;">Urgent Matters</p>',
                unsafe_allow_html=True)
            with st.container(border=True, height=200):
                if st.session_state.clicked:
                    anger = "🤬"
                    angeravg = int((len(data[data["Emotion"] == "Anger"]) / len(data)) * 100)
                    anger_font_style = f'<span class="bold-font">{anger}</span>'
                    angeravg_font_style = f'<span class="bold-font">{angeravg}%</span>'
                    sad = "😭"
                    sadavg = int((len(data[data["Emotion"] == "Sadness"]) / len(data)) * 100)
                    sad_font_style = f'<span class="bold-font">{sad}</span>'
                    sadavg_font_style = f'<span class="bold-font">{sadavg}%</span>'

                    st.markdown(""" 
                        <style> 
                            @import url("https://fonts.googleapis.com/css2?family=Besley:ital,wght@0,400..900;1,400..900&family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..900;1,6..96,400..900&family=Cinzel:wght@400..900&family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap");

                            .angerem-font {
                                font-size: 20px;
                                font-family: "Poppins", sans-serif;
                                color: white;
                                font-weight: 300;
                                padding-top: 0rem;
                                margin-top: 0rem;
                                padding-left: 0rem;
                                margin-left: 0rem;
                            } 

                            .bold-font {
                                font-size: 35px;
                                font-family: "Poppins", sans-serif;
                                color: white;
                                font-weight: 500;
                            }
                        </style> 
                    """, unsafe_allow_html=True)

                    # Combining styles in a single sentence
                    st.markdown(
                        f'<p class="angerem-font" style="text-align:center;">{anger_font_style} in {angeravg_font_style} reviews</p>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f'<p class="angerem-font" style="text-align:center;">{sad_font_style} in {sadavg_font_style} reviews</p>',
                        unsafe_allow_html=True
                    )

        with bot1:
            st.text("")
            st.text("")
            st.markdown(""" 
                        <style> 
                            @import url("https://fonts.googleapis.com/css2?family=Besley:ital,wght@0,400..900;1,400..900&family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..900;1,6..96,400..900&family=Cinzel:wght@400..900&family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" );

                            .ttdash-font {
                                font-size: 25px;
                                font-family:  "Poppins", sans-serif;
                                color: white;
                                font-weight: 600;
                                padding-top: 0rem;
                                margin-top: 0rem;
                            } 
                        </style> 
                    """, unsafe_allow_html=True)

            st.markdown(
                '<p class="ttdash-font" style="text-align:center;">Ask our AI Assistant</p>',
                unsafe_allow_html=True)
            with st.container(border = True, height = 500):
                st.markdown(
                    """
                    <style>
                    .custom-textarea textarea {
                        font-size: 16px !important;  /* Text size */
                        height: 150px !important;   /* Height of the textarea */
                        width: 100% !important;     /* Width of the textarea */
                        border: 2px solid #4CAF50;  /* Green border */
                        border-radius: 8px;         /* Rounded corners */
                        padding: 10px;              /* Inner padding */
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                # Add the text area with a custom class
                input_text = st.text_area("Enter your query", key="custom", placeholder="Type your query here...",
                                          label_visibility="collapsed")

                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []

            # Perform analysis
                if input_text:
                    # Add the query to the chat history
                    st.session_state.chat_history.append({"query": input_text})

                    # Simulate response from the `chat_with_csv` function
                    result = chat_with_csv(data, input_text)

                    # Add the result to the chat history
                    st.session_state.chat_history[-1]["response"] = result

                    # Display the result of the current query with custom styles
                    st.markdown(
                        f"""
                        <div style='background-color: #63747A; 
                                    color: white; 
                                    border-radius: 10px; 
                                    padding: 10px; 
                                    margin-bottom: 10px;'>
                            <strong>Bot:</strong> {result}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.text("")
                    st.text("")
                    st.text("")
                # Display the chat history
                if st.session_state.chat_history:
                    for entry in st.session_state.chat_history:
                        st.markdown(
                            f"""
                            <div style='background-color: #E0E0E0; border-radius: 10px; padding: 10px; margin-bottom: 10px; color: #1E232D;'>
                                <strong>You:</strong> {entry['query']}
                            </div>
                            """, unsafe_allow_html=True)

                        # Bot response with green background
                        st.markdown(
                            f"""
                            <div style='background-color: #A0A0A0; border-radius: 10px; padding: 10px; margin-bottom: 10px; color: #1E232D;'>
                                <strong>Bot:</strong> {entry['response']}
                            </div>
                            """, unsafe_allow_html=True)

        with dat0:
            st.text("")
            st.text("")
            st.markdown(""" 
                        <style> 
                            @import url("https://fonts.googleapis.com/css2?family=Besley:ital,wght@0,400..900;1,400..900&family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..900;1,6..96,400..900&family=Cinzel:wght@400..900&family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" );

                            .ttdash-font {
                                font-size: 25px;
                                font-family:  "Poppins", sans-serif;
                                color: white;
                                font-weight: 600;
                                padding-top: 0rem;
                                margin-top: 0rem;
                            } 
                        </style> 
                    """, unsafe_allow_html=True)

            st.markdown(
                '<p class="ttdash-font" style="text-align:center;">Navigate the Data</p>',
                unsafe_allow_html=True)
            with st.container(border = True, height = 500):
                dtf(data)



        if __name__ == '__main__':
            app(data)






