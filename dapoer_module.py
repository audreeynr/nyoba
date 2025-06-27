# dapoer_module.py
import pandas as pd
import re
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
import random

# Load dan bersihkan data
CSV_FILE_PATH = 'https://raw.githubusercontent.com/audreeynr/dapoer-ai/refs/heads/main/data/Indonesian_Food_Recipes.csv'
df = pd.read_csv(CSV_FILE_PATH)
df_cleaned = df.dropna(subset=['Title', 'Ingredients', 'Steps']).drop_duplicates()

# Normalisasi
def normalize_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return text

df_cleaned['Title_Normalized'] = df_cleaned['Title'].apply(normalize_text)
df_cleaned['Ingredients_Normalized'] = df_cleaned['Ingredients'].apply(normalize_text)
df_cleaned['Steps_Normalized'] = df_cleaned['Steps'].apply(normalize_text)

# Format hasil masakan
def format_recipe(row):
    bahan_raw = re.split(r'\n|--|,', row['Ingredients'])
    bahan_list = [b.strip().capitalize() for b in bahan_raw if b.strip()]
    bahan_md = "\n".join([f"- {b}" for b in bahan_list])

    langkah_md = row['Steps'].strip()

    return f"""🍽 {row['Title']}\n\nBahan-bahan:  \n{bahan_md}\n\nLangkah Memasak:  \n{langkah_md}"""

# Tool 1: Cari berdasarkan nama masakan
def search_by_title(query):
    query_normalized = normalize_text(query)
    match_title = df_cleaned[df_cleaned['Title_Normalized'].str.contains(query_normalized)]
    if not match_title.empty:
        return format_recipe(match_title.iloc[0])
    return "Resep tidak ditemukan berdasarkan judul."

# Tool 2: Cari berdasarkan bahan
def search_by_ingredients(query):
    stopwords = {"masakan", "apa", "saja", "yang", "bisa", "dibuat", "dari", "menggunakan", "bahan", "resep"}
    prompt_lower = normalize_text(query)
    bahan_keywords = [w for w in prompt_lower.split() if w not in stopwords and len(w) > 2]

    if bahan_keywords:
        mask = df_cleaned['Ingredients_Normalized'].apply(lambda x: all(k in x for k in bahan_keywords))
        match_bahan = df_cleaned[mask]
        if not match_bahan.empty:
            hasil = match_bahan.head(5)['Title'].tolist()
            return "Masakan yang menggunakan bahan tersebut:\n- " + "\n- ".join(hasil)
    return "Tidak ditemukan masakan dengan bahan tersebut."

# Tool 3: Cari berdasarkan metode masak
def search_by_method(query):
    prompt_lower = normalize_text(query)
    for metode in ['goreng', 'panggang', 'rebus', 'kukus']:
        if metode in prompt_lower:
            cocok = df_cleaned[df_cleaned['Steps_Normalized'].str.contains(metode)]
            if not cocok.empty:
                hasil = cocok.head(5)['Title'].tolist()
                return f"Masakan yang dimasak dengan cara {metode}:\n- " + "\n- ".join(hasil)
    return "Tidak ditemukan metode memasak yang cocok."

# Tool 4: Rekomendasi masakan mudah
def recommend_easy_recipes(query):
    prompt_lower = normalize_text(query)
    if "mudah" in prompt_lower or "pemula" in prompt_lower:
        hasil = df_cleaned[df_cleaned['Steps'].str.len() < 300].head(5)['Title'].tolist()
        return "Rekomendasi masakan mudah:\n- " + "\n- ".join(hasil)
    return "Tidak ditemukan masakan mudah yang relevan."

# Tool 5: RAG dengan FAISS dan fallback RAG-Like

def build_vectorstore(api_key):
    docs = []
    for _, row in df_cleaned.iterrows():
        content = f"Title: {row['Title']}\nIngredients: {row['Ingredients']}\nSteps: {row['Steps']}"
        docs.append(Document(page_content=content))

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    texts = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key)
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore


def rag_search(api_key, query):
    vectorstore = build_vectorstore(api_key)
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)

    if not docs:
        fallback_samples = df_cleaned.sample(5)
        fallback_response = "\n\n".join([
            f"{row['Title']}:\nBahan: {row['Ingredients']}\nLangkah: {row['Steps']}"
            for _, row in fallback_samples.iterrows()
        ])
        return f"Tidak ditemukan informasi yang relevan. Berikut beberapa rekomendasi masakan acak:\n\n{fallback_response}"

    return "\n\n".join([doc.page_content for doc in docs[:5]])

# Membuat Agent

def create_agent(api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.7
    )

    def rag_tool_func(query):
        return rag_search(api_key, query)

    tools = [
        Tool(name="SearchByTitle", func=search_by_title, description="Cari resep berdasarkan judul masakan."),
        Tool(name="SearchByIngredients", func=search_by_ingredients, description="Cari masakan berdasarkan bahan."),
        Tool(name="SearchByMethod", func=search_by_method, description="Cari masakan berdasarkan metode memasak."),
        Tool(name="RecommendEasyRecipes", func=recommend_easy_recipes, description="Rekomendasi masakan yang mudah dibuat."),
        Tool(name="RAGSearch", func=rag_tool_func, description="Cari informasi masakan menggunakan FAISS dan RAG dengan fallback rekomendasi acak.")
    ]

    memory = ConversationBufferMemory(memory_key="chat_history")

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        memory=memory,
        verbose=False
    )

    return agent
