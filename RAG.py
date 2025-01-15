from langchain_groq import ChatGroq
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_csv_agent
import pandas as pd
import random
import faker
from datetime import datetime, timedelta
import streamlit as st
import whisper
from transformers import pipeline
import time

#DF CREATION
# Initialize Faker and set seed for reproducibility
fake = faker.Faker()
random.seed(42)

# Generate random data for 50 rows
items = ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard', 'Mouse']
countries = [
    'Germany', 'France', 'Italy', 'Spain', 'UK', 'Netherlands',
    'Sweden', 'Norway', 'Denmark', 'Switzerland', 'Belgium',
    'Austria', 'Ireland', 'Portugal', 'Poland', 'Czech Republic',
    'Hungary', 'Slovakia', 'Finland', 'Iceland', 'Greece',
    'Estonia', 'Latvia', 'Lithuania', 'Luxembourg', 'Croatia',
    'Romania', 'Bulgaria', 'Serbia', 'Slovenia'
]
coordinates = {
    'Germany': [(52.5200, 13.4050), (48.1351, 11.5820), (50.1109, 8.6821)],  # Berlin, Munich, Frankfurt
    'France': [(48.8566, 2.3522), (43.7102, 7.2620), (45.7640, 4.8357)],    # Paris, Nice, Lyon
    'Italy': [(41.9028, 12.4964), (45.4642, 9.1900), (42.3520, 14.4030)],   # Rome, Milan, Naples
    'Spain': [(40.4168, -3.7038), (41.3874, 2.1686), (37.3891, -5.9845)],   # Madrid, Barcelona, Seville
    'UK': [(51.5074, -0.1278), (55.9533, -3.1883), (53.4839, -2.2446)],     # London, Edinburgh, Manchester
    'Netherlands': [(52.3676, 4.9041), (51.9244, 4.4777), (53.2194, 6.5665)], # Amsterdam, Rotterdam, Groningen
    'Sweden': [(59.3293, 18.0686), (57.7089, 11.9746), (63.8258, 20.2630)], # Stockholm, Gothenburg, Umeå
    'Norway': [(59.9139, 10.7522), (63.4305, 10.3951), (69.6496, 18.9560)], # Oslo, Trondheim, Tromsø
    'Denmark': [(55.6761, 12.5683), (56.1629, 10.2039), (57.0488, 9.9217)], # Copenhagen, Aarhus, Aalborg
    'Switzerland': [(46.9481, 7.4474), (47.3769, 8.5417), (46.2044, 6.1432)], # Bern, Zurich, Geneva
    'Belgium': [(50.8503, 4.3517), (51.2194, 4.4025), (50.6386, 5.5721)],   # Brussels, Antwerp, Liège
    'Austria': [(48.2082, 16.3738), (47.0707, 15.4395), (47.8095, 13.0550)], # Vienna, Graz, Salzburg
    'Ireland': [(53.3498, -6.2603), (51.8985, -8.4756), (54.5973, -5.9301)], # Dublin, Cork, Belfast
    'Portugal': [(38.7169, -9.1399), (41.1579, -8.6291), (37.0179, -7.9308)], # Lisbon, Porto, Faro
    'Poland': [(52.2297, 21.0122), (50.0647, 19.9450), (53.1325, 23.1688)], # Warsaw, Krakow, Bialystok
    'Czech Republic': [(50.0755, 14.4378), (49.1951, 16.6068), (50.2104, 12.8655)], # Prague, Brno, Karlovy Vary
    'Hungary': [(47.4979, 19.0402), (46.2530, 20.1414), (48.1000, 19.7908)], # Budapest, Szeged, Miskolc
    'Slovakia': [(48.1486, 17.1077), (49.2239, 18.7390), (48.7394, 19.1534)], # Bratislava, Žilina, Banská Bystrica
    'Finland': [(60.1695, 24.9354), (61.4978, 23.7610), (68.9674, 33.6751)], # Helsinki, Tampere, Rovaniemi
    'Iceland': [(64.1355, -21.8954), (65.6833, -18.1059), (63.9839, -22.5714)], # Reykjavik, Akureyri, Keflavik
    'Greece': [(37.9838, 23.7275), (40.6401, 22.9444), (35.5122, 24.0180)], # Athens, Thessaloniki, Chania
    'Estonia': [(59.4370, 24.7536), (58.3776, 26.7290), (58.3460, 24.4725)], # Tallinn, Tartu, Pärnu
    'Latvia': [(56.9496, 24.1052), (56.5093, 21.0108), (57.3139, 25.2689)], # Riga, Liepaja, Valmiera
    'Lithuania': [(54.6872, 25.2797), (55.7033, 21.1443), (55.9276, 23.3154)], # Vilnius, Klaipėda, Šiauliai
    'Luxembourg': [(49.8153, 6.1296), (49.6106, 6.1319), (49.4464, 5.9814)], # Luxembourg City, Esch-sur-Alzette, Differdange
    'Croatia': [(45.8150, 15.9819), (44.1194, 15.2314), (42.6507, 18.0944)], # Zagreb, Zadar, Dubrovnik
    'Romania': [(44.4268, 26.1025), (46.7704, 23.5914), (47.6617, 26.2611)], # Bucharest, Cluj-Napoca, Suceava
    'Bulgaria': [(42.6977, 23.3219), (43.2141, 27.9147), (42.1354, 24.7453)], # Sofia, Varna, Plovdiv
    'Serbia': [(44.8176, 20.4569), (45.2510, 19.8369), (43.3223, 21.8956)], # Belgrade, Novi Sad, Niš
    'Slovenia': [(46.0569, 14.5058), (46.2383, 14.3557), (46.4830, 15.0086)]  # Ljubljana, Bled, Maribor
}
comments = [
    "I luv it! Totally worth it!",
    "Would NOT recomend. very poor qualty.",
    "absolUtely fanTastic!!",
    "not wroth the money tbh.",
    "Its ok, but not great.",
    "OMG BEST PURCHASE EVER!!!",
    "bad experience. broke within a week.",
    "Decent product, but not for everyone.",
    "what a waste. Really disapointing.",
    "LOVE the features. So useful!!!",
    "meh, it's fine i guess.",
    "thE build quality is awesome!",
    "Had high expectations... but its just ok.",
    "Totally exceeded my expctations! great stuff.",
    "the prce was too high for the quality.",
    "Amazing product but delivery was super sloooow.",
    "WOULD buy again in a heartbeat!",
    "the screen scratches waaaaay too easily.",
    "Ugh, it BROKE after just a month!?",
    "works like a charm, but kinda noisy.",
    "The packaging SUCKED but the product is awesome.",
    "Not gr8, but I’ve seen worse.",
    "Can't blieve how good this turned out!!!",
    "Worst purchase evr. Pls avoid.",
    "the sound quality is SICK. love it!",
    "Honestly? dont waste ur money on this.",
    "IT’S PERFECT!!! Cant reccomend enuf.",
    "Took me FOREVER to figure out how to set it up :(",
    "It’s fine, but def not worth the hype.",
    "BEST deal ever! So so happy rn.",
    "ugh, rlly regret buying this. not worth it.",
    "Looks fab, works meh.",
    "NO regrets. best purchase of 2024!!!",
    "I thought it wld b better... kinda disappointed.",
    "Build qualty is gud, but software is trash.",
    "Totally underwhelmed. Expected more 4 the price.",
    "It's amazng! My fam loves it too!",
    "Horrible! I WANT A REFUND NOW.",
    "Not bad, but there’s better stuff out there.",
    "Such a steal!!! Worth every penny.",
    "Battery life sux, but everything else is nice.",
    "OMG the BEST THING EVER!",
    "Cant stop using it. rlly happy wth it.",
    "looks cheap. feels cheap. IS cheap.",
    "ehhhh, its okay. wouldn’t recomend tho.",
    "WOAH!!! Did not expect it to b this gud!",
    "Total waste of money. Not evn close to the reviews.",
    "its gud but overpriced.",
    "love it, but it’s kinda fragile :(.",
    "Arrived damaged. so upset rn.",
    "Cant live without it now. lifesaver!!!",
    "Meh. Just meh.",
    "GREAT BUY, but shipping was slooooooow.",
    "Rlly pricy, but so worth it!",
    "I’m SO MAD. It stopped working on day 2!",
    "what a joke. do NOT buy this garbage.",
    "Decent enough, but lacks key features.",
    "Th build quality is STUNNING!!!",
    "ugh, rlly dissapointed. not as promised.",
    "its fine, but nothing amazing.",
    "OMG LOVEEEE IT! best decision EVER!",
    "what a disaster. save ur $$$.",
    "Wow. Just wow. exceeded ALL my expectations!",
    "Kinda clunky but works great.",
    "looks great, works better!",
    "This is my 3rd purchase n im STILL in love!!!",
    "STOPPED working aftr a few days. NOT happy.",
    "Battery life is AMAZING but takes forever to charge.",
    "it’s okay. not bad for the price.",
    "So far so gud! Hope it lasts long.",
    "Looks cool but feels rlly cheap.",
    "Cant b happier! Totally worth it.",
    "The buttons feel rlly flimsy tho.",
    "It’s aight. Def not what I expected.",
    "Came in late, but rlly luv it now.",
    "Worst EVER!!! dont buy this pls.",
    "The color is sooooo pretty! LOVE IT!!!"
]


# Generate data
data = []
for _ in range(500):
    name = fake.name()
    sales_amount = round(random.uniform(100, 5000), 2)
    item_bought = random.choice(items)
    purchase_date = fake.date_this_year(before_today=True, after_today=False)
    country = random.choice(countries)
    review = random.randint(1, 5)

    data.append([name, sales_amount, item_bought, purchase_date, country, review])

# Create a DataFrame
df = pd.DataFrame(data, columns=['Name', 'SalesAmount', 'ItemBought', 'PurchaseDate', 'Country', 'Review'])
df['ExperienceRating'] = [random.randint(1, 5) for _ in range(len(df))]
df['Coordinates'] = df['Country'].apply(lambda x: random.choice(coordinates[x]))
df['Comment'] = [random.choice(comments) for _ in range(len(df))]
df['Latitude'] = df['Coordinates'].apply(lambda x: x[0])
df['Longitude'] = df['Coordinates'].apply(lambda x: x[1])

classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=False)
prediction = classifier("OMG BEST PURcASE EVER", )
print(prediction[0]["label"])
pipe = pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
sent = pipe("luv this imma use it 4eva", )
print(sent)
mod = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
grade = mod("luv this imma use it 4eva", )
print(grade)

start = time.time()
df["Label"] = df["Comment"].apply(lambda x: pipe(x)[0]["label"].capitalize())
end = time.time()
print(end - start)
start = time.time()
df["Review"] = df["Comment"].apply(lambda x: mod(x)[0]["label"][0])
end = time.time()
print(end - start)
start = time.time()
df["Emotion"] = df["Comment"].apply(lambda x: classifier(x)[0]["label"].capitalize())
end = time.time()
print(end - start)

round(df["Label"].value_counts()[0] / len(df), 2)
print(pd.DataFrame(df["Emotion"].value_counts()).reset_index()["Emotion"][0] / len(df))


df.to_csv('RAG.csv', index=False)
df_path = "/Users/luigicolonna/Desktop/PycharmProjects/GenAI/RAG.csv"

#groq_api = 'your-groq-api-key'
llm = ChatGroq(temperature=0.2, model="mixtral-8x7b-32768", api_key="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")


# Create the CSV agent
agent = create_csv_agent(llm, df_path, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True, agent_executor_kwargs={"output_parser": None})

conversation_history = []

def query_data(query):
    try:
        # Here, interact with your LangChain agent
        response = agent.invoke(query)
        if isinstance(response, dict) and 'output' in response:
            response = response['output']
        return response
    except Exception as e:
        return f"Sorry, I couldn't process your request. Error: {e}"

def chatbot():
    st.title("ChatBot")

    # Store conversation history in session state for persistence
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # Display conversation so far
    for message in st.session_state.conversation:
        st.write(message)

    # User input
    query = st.text_input("You:", key="query")

    if query:
        # Add user message to conversation history
        st.session_state.conversation.append(f"User: {query}")

        # Get bot response
        response = query_data(query)

        # Add bot message to conversation history
        st.session_state.conversation.append(f"Bot: {response}")

        # Display response
        st.write(f"Bot: {response}")

# Run the Streamlit app
if __name__ == "__main__":
    chatbot()

import whisper
from transformers import pipeline

classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=False)
prediction = classifier("OMG BEST PURcASE EVER", )
print(prediction[0]["label"])
pipe = pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
sent = pipe("luv this imma use it 4eva", )
print(sent)
mod = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
grade = mod("luv this imma use it 4eva", )
print(grade)

model = whisper.load_model("turbo")
result = model.transcribe("audio.mp3")
print(result["text"])

df = pd.read_csv("/Users/luigicolonna/Desktop/PycharmProjects/GenAI/RAG.csv")
df.Emotion.value_counts()

int((len(df[df["Emotion"] == "Anger"]) / len(df))*100)

len(df[df["Emotion"] == "Sadness"])

