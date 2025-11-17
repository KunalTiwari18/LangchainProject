from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline

# ------------------------------------------
# 50+ Question–Answer Data About Mumbai City
# ------------------------------------------
qa_pairs = [

    {"question": "What is Mumbai famous for?",
     "answer": "Mumbai is famous for Bollywood, the Gateway of India, Marine Drive, local trains, street food, and beaches."},

    {"question": "Which is the biggest film industry in Mumbai?",
     "answer": "Bollywood is the biggest film industry in Mumbai."},

    {"question": "Which place is known as the Queen's Necklace in Mumbai?",
     "answer": "Marine Drive is called the Queen’s Necklace because of its night lighting."},

    {"question": "What is the major airport in Mumbai?",
     "answer": "Chhatrapati Shivaji Maharaj International Airport is the main airport in Mumbai."},

    {"question": "Which beach in Mumbai is most famous?",
     "answer": "Juhu Beach is the most famous beach in Mumbai."},

    {"question": "What is the iconic monument of Mumbai?",
     "answer": "The Gateway of India is the most iconic monument of Mumbai."},

    {"question": "Which food is Mumbai most famous for?",
     "answer": "Mumbai is famous for Vada Pav, Pav Bhaji, Bhel Puri, and Misal."},

    {"question": "Which place in Mumbai is best for nightlife?",
     "answer": "Bandra, Lower Parel, and Colaba are best known for nightlife in Mumbai."},

    {"question": "Where is the Elephanta Caves located?",
     "answer": "Elephanta Caves are located on Elephanta Island, accessible by ferry from Gateway of India."},

    {"question": "Which sea surrounds Mumbai?",
     "answer": "Mumbai is surrounded by the Arabian Sea."},

    {"question": "Which Mumbai train line is the busiest?",
     "answer": "The Western Line is considered the busiest local train route in Mumbai."},

    {"question": "Where is Siddhivinayak Temple located?",
     "answer": "The Siddhivinayak Temple is located in Prabhadevi, Mumbai."},

    {"question": "Which is the biggest park in Mumbai?",
     "answer": "Sanjay Gandhi National Park in Borivali is the largest park in Mumbai."},

    {"question": "Which picnic spot is best for nature lovers in Mumbai?",
     "answer": "Sanjay Gandhi National Park and Kanheri Caves are ideal for nature lovers."},

    {"question": "Which place in Mumbai is best for couples?",
     "answer": "Marine Drive, Bandra Bandstand, and Hanging Gardens are popular for couples."},

    {"question": "What is the best time to visit Mumbai?",
     "answer": "October to February is the best time, due to pleasant weather."},

    {"question": "Which street food of Mumbai is most popular?",
     "answer": "Vada Pav is considered Mumbai’s most iconic street food."},

    {"question": "Which Mumbai market is famous for shopping?",
     "answer": "Colaba Causeway, Linking Road, Chor Bazaar, and Crawford Market are famous shopping spots."},

    {"question": "Which museum in Mumbai is very popular?",
     "answer": "Chhatrapati Shivaji Maharaj Vastu Sangrahalaya is a major museum."},

    {"question": "Where can you see sunset in Mumbai?",
     "answer": "Juhu Beach, Versova Beach, and Marine Drive are best for sunset views."},

    {"question": "What is the nickname of Mumbai?",
     "answer": "Mumbai is also called the City of Dreams and Bollywood City."},

    {"question": "Which area in Mumbai is famous for celebrities?",
     "answer": "Bandra and Juhu are areas where many Bollywood celebrities live."},

    {"question": "Which is Mumbai’s largest mall?",
     "answer": "Phoenix Marketcity in Kurla is one of Mumbai’s largest malls."},

    {"question": "Which sea link is famous in Mumbai?",
     "answer": "The Bandra-Worli Sea Link is a major landmark."},

    {"question": "Where is Haji Ali Dargah located?",
     "answer": "Haji Ali Dargah is located in Worli, on a small island linked by a walkway."},

    {"question": "What is Mumbai local train called?",
     "answer": "It is called the Mumbai Suburban Railway, the lifeline of the city."},

    {"question": "Which is the oldest station in Mumbai?",
     "answer": "Chhatrapati Shivaji Maharaj Terminus (CSMT) is one of the oldest stations."},

    {"question": "Which park in Mumbai is good for picnics?",
     "answer": "Hanging Garden and Priyadarshini Park are popular picnic spots."},

    {"question": "What is the main language spoken in Mumbai?",
     "answer": "Marathi is the official language, but Hindi and English are widely spoken."},

    {"question": "What is Mumbai’s famous dessert?",
     "answer": "Kulfi, Falooda, and Mango Mastani are popular Mumbai desserts."},

    {"question": "Where can I try the best Pav Bhaji in Mumbai?",
     "answer": "Sardar Pav Bhaji in Tardeo is very famous."},

    {"question": "Which fort is famous near Mumbai?",
     "answer": "Bandra Fort is a popular fort in Mumbai."},

    {"question": "Which area is famous for pubs and cafés?",
     "answer": "Bandra and Lower Parel are hotspots for cafés and pubs."},

    {"question": "Where can you take a ferry ride in Mumbai?",
     "answer": "Ferry rides are available from the Gateway of India."},

    {"question": "What is the famous art district in Mumbai?",
     "answer": "Kala Ghoda is known for its art and cultural festivals."},

    {"question": "Which library is famous in Mumbai?",
     "answer": "Asiatic Society Library at Town Hall is very famous."},

    {"question": "Where is the best street food in Mumbai?",
     "answer": "Juhu Chowpatty, Girgaum Chowpatty, and Ghatkopar Khau Galli."},

    {"question": "Which is the cleanest beach in Mumbai?",
     "answer": "Girgaum Chowpatty and Versova Beach have improved cleanliness efforts."},

    {"question": "Which island near Mumbai has caves?",
     "answer": "Elephanta Island has ancient caves."},

    {"question": "Where is the biggest amusement park?",
     "answer": "EsselWorld and Water Kingdom are large amusement parks in Mumbai."},

    {"question": "Which place is best for photography in Mumbai?",
     "answer": "Worli Sea Face, Marine Drive, Velankanni Church, and Bandra Fort."},

    {"question": "Which is Mumbai’s most crowded area?",
     "answer": "Dadar and Andheri are among the most crowded locations."},

    {"question": "Which zoo is in Mumbai?",
     "answer": "Veermata Jijabai Bhosale Udyan (Byculla Zoo) is Mumbai’s zoo."},

    {"question": "Which market is famous for antiques?",
     "answer": "Chor Bazaar in Mumbai is famous for antiques."},

    {"question": "Which area is famous for budget food?",
     "answer": "Mohammad Ali Road, Ghatkopar Khau Galli, and Fort area are affordable food hubs."},

    {"question": "Which is the largest lake in Mumbai?",
     "answer": "Vihar Lake is the largest lake in Mumbai."},
]

# ---------------------------------------------------
# Embeddings & Vector Store
# ---------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [f"Q: {q['question']} A: {q['answer']}" for q in qa_pairs]

vector_store = FAISS.from_texts(documents, embeddings)

# ---------------------------------------------------
# Small, CPU-Friendly HuggingFace LLM
# ---------------------------------------------------
hf_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=128,
)

llm = HuggingFacePipeline(pipeline=hf_model)

# ---------------------------------------------------
# Conversation Chain with Memory
# ---------------------------------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chatbot = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory
)

# ---------------------------------------------------
# Chat Loop
# ---------------------------------------------------
print("\n🤖 Mumbai AI Chatbot Ready! Ask anything about Mumbai. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    response = chatbot({"question": user_input})
    print("Bot:", response["answer"], "\n")
