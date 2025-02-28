from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import re
import psycopg2
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
import os
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)


segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)


model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')


POSTERS_FOLDER = os.path.join('static', 'Tosters')

app = Flask(__name__)


def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="films",
        user="postgres",
        password="tatavk14"
    )


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def preprocess_text(text):

    text = re.sub('<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()


    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)


    tokens = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        if token.lemma:
            tokens.append(token.lemma)
    return " ".join(tokens)


def search_films(search_type, search_text):
    conn = get_db_connection()
    search_text = search_text.lower()

    if search_type == 'по тегу':
        query = f"""
            SELECT name, rating, description, embedding 
            FROM films 
            WHERE tag ILIKE %s 
            ORDER BY 
                CASE 
                    WHEN rating = 'Неизвестно' THEN 1 
                    ELSE 0 
                END, 
                rating DESC
        """
        params = (f'%{search_text}%',)
    elif search_type == 'по названию':
        query = f"""
            SELECT name, rating, description, embedding 
            FROM films 
            WHERE name ILIKE %s 
            ORDER BY 
                CASE 
                    WHEN rating = 'Неизвестно' THEN 1 
                    ELSE 0 
                END, 
                rating DESC
        """
        params = (f'%{search_text}%',)
    elif search_type == 'по описанию':

        processed_text = preprocess_text(search_text)


        print(f"Поисковый запрос после Natasha: '{processed_text}'")


        query_vector = model.encode([processed_text])[0]
        query_vector = normalize([query_vector])[0]


        print(f"Вектор поискового запроса: {query_vector}")


        query = """
            SELECT name, rating, description, embedding 
            FROM films
        """
        df = pd.read_sql_query(query, conn)


        similarities = []
        for index, row in df.iterrows():

            if isinstance(row['embedding'], str):
                embedding = np.array(json.loads(row['embedding']))
            else:
                embedding = np.array(row['embedding'])
            embedding = normalize([embedding])[0]
            similarity = cosine_similarity([query_vector], [embedding])[0][0]
            similarities.append(similarity)


        df['similarity'] = similarities


        df = df.sort_values(by='similarity', ascending=False)


        if not df.empty:
            best_film = df.iloc[0]
            print(f"Лучший фильм: '{best_film['name']}'")
            print(f"Описание лучшего фильма: '{best_film['description']}'")
            print(f"Косинусное сходство: {best_film['similarity']:.4f}")


            if isinstance(best_film['embedding'], str):
                best_film_embedding = np.array(json.loads(best_film['embedding']))
            else:
                best_film_embedding = np.array(best_film['embedding'])
            best_film_embedding = normalize([best_film_embedding])[0]
            print(f"Вектор лучшего фильма: {best_film_embedding}")


        return df.head(100)
    elif search_type == 'по рейтингу':
        query = f"""
            SELECT name, rating, description, embedding 
            FROM films 
            ORDER BY 
                CASE 
                    WHEN rating = 'Неизвестно' THEN 1 
                    ELSE 0 
                END, 
                rating DESC
            LIMIT 100
        """
        params = ()
    else:
        return pd.DataFrame()

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        search_type = request.form.get('search_type')
        search_text = request.form.get('search_text')
        results = search_films(search_type, search_text)
    else:
        search_type = 'по рейтингу'
        search_text = ''
        results = search_films(search_type, search_text)


    films_with_posters = []
    for film in results.to_dict('records'):
        poster_filename = f"{film['name']}.jpg"
        poster_path = os.path.join(POSTERS_FOLDER, poster_filename)
        absolute_poster_path = f"/static/Posters/{poster_filename}".replace('\\', '/')
        print(f"Постер для фильма '{film['name']}': {absolute_poster_path}")
        films_with_posters.append({
            'name': film['name'],
            'rating': film['rating'],
            'description': film['description'],
            'poster_path': absolute_poster_path if os.path.exists(poster_path) else None
        })

    return render_template('index.html', films=films_with_posters)


if __name__ == '__main__':
    app.run(debug=True)