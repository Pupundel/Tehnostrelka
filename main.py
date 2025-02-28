import requests
from bs4 import BeautifulSoup
import re
import psycopg2
import multiprocessing
from sentence_transformers import SentenceTransformer, util
import json
import os
import wget
import logging
import time
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


logging.basicConfig(filename='errors.log', level=logging.ERROR)


model = SentenceTransformer('all-mpnet-base-v2')


def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="films",
        user="postgres",
        password="tatavk14"
    )

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


def add_film_to_db(name, description, tag, rating, embedding):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO films (name, description, tag, rating, embedding)
            VALUES (%s, %s, %s, %s, %s);
        """

        embedding_json = json.dumps(embedding.tolist())
        cursor.execute(query, (name, description, tag, rating, embedding_json))
        conn.commit()
    except Exception as e:
        logging.error(f"Ошибка при добавлении фильма '{name}': {e}")
    finally:
        if conn:
            conn.close()


def requests_to_page(page_ranges):
    start_page, end_page = page_ranges

    for page in range(start_page, end_page + 1):
        try:
            link = f"https://baskino.fm/filmy/page/{page}"
            response = requests.get(link)
            if response.status_code != 200:
                logging.error(f"Ошибка при запросе страницы: {link}")
                continue

            soup = BeautifulSoup(response.text, 'lxml')
            block = soup.find('div', class_="content_block")
            if not block:
                logging.error(f"Блок с фильмами не найден на странице: {link}")
                continue

            films = block.find_all('div', class_="shortpost")
            for film in films:
                try:

                    try:
                        link_to_film = film.find('a').get('href')
                    except Exception as e:
                        logging.error(f"Ошибка при получении ссылки на фильм: {e}")
                        continue


                    try:
                        name = film.find('div', class_='posttitle').text
                    except Exception as e:
                        logging.error(f"Ошибка при получении названия фильма: {e}")
                        continue


                    try:
                        film_page_response = requests.get(link_to_film)
                        if film_page_response.status_code != 200:
                            logging.error(f"Ошибка при запросе страницы фильма: {link_to_film}")
                            continue
                        film_page_soup = BeautifulSoup(film_page_response.text, 'lxml')
                    except Exception as e:
                        logging.error(f"Ошибка при запросе страницы фильма: {e}")
                        continue


                    try:
                        description_element = film_page_soup.find('div', class_='full_movie_desc')
                        if description_element:
                            description = description_element.text


                            description = preprocess_text(description)

                            try:

                                embedding = model.encode([description])[0]
                            except Exception as e:
                                embedding = None
                                logging.error(f"Ошибка при векторизации описания фильма '{name}': {e}")
                        else:
                            description = "Описание отсутствует"
                            logging.error(f"Описание не найдено для фильма: {name}")
                    except Exception as e:
                        logging.error(f"Ошибка при получении описания фильма: {e}")
                        description = "Описание отсутствует"


                    try:
                        rating_element = film_page_soup.find('div', class_='full_movie_rating_bob')
                        if rating_element and len(rating_element.text) > 6:
                            rating = rating_element.text[6:]
                        else:
                            rating = "Рейтинг отсутствует"
                            logging.error(f"Рейтинг не найден для фильма: {name}")
                    except Exception as e:
                        logging.error(f"Ошибка при получении рейтинга фильма: {e}")
                        rating = "Рейтинг отсутствует"


                    try:
                        full_info = film_page_soup.find('div', class_='right_block')
                        if full_info:
                            li_elements = full_info.find_all('li')
                            if len(li_elements) > 2 and len(li_elements[2].find_all('span')) > 1:
                                tag = li_elements[2].find_all('span')[1].text
                            else:
                                tag = "Тег отсутствует"
                        else:
                            tag = "Тег отсутствует"
                            logging.error(f"Тег не найден для фильма: {name}")
                    except Exception as e:
                        logging.error(f"Ошибка при получении тега фильма: {e}")
                        tag = "Тег отсутствует"


                    try:
                        image_element = film_page_soup.find('div', class_='poster_full_movie')
                        if image_element:
                            image = str(image_element)
                            match = re.search(r'data-src="([^"]+)"', image)
                            if match:
                                data_src_value = match.group(1)
                                if data_src_value[0] == 'h':
                                    url = data_src_value
                                else:
                                    url = f"https://baskino.fm/{data_src_value}"
                            else:
                                url = None
                        else:
                            url = None
                    except Exception as e:
                        logging.error(f"Ошибка при получении URL постера: {e}")
                        url = None


                    if url:
                        try:
                            if not os.path.exists("static/Posters"):
                                os.makedirs("static/Posters")
                            safe_name = re.sub(r'[\\/*?:"<>|]', "_", name)
                            poster_path = os.path.join("static/Posters", f"{safe_name}.jpg")
                            wget.download(url, out=poster_path)
                        except Exception as e:
                            logging.error(f"Ошибка при скачивании постера для фильма '{name}': {e}")


                    try:
                        add_film_to_db(name, description, tag, rating, embedding)
                    except Exception as e:
                        logging.error(f"Ошибка при добавлении фильма '{name}' в базу данных: {e}")

                except Exception as e:
                    logging.error(f"Ошибка при обработке фильма: {e}")
                    continue

        except Exception as e:
            logging.error(f"Ошибка при обработке страницы {page}: {e}")
            continue


        time.sleep(1)

if __name__ == "__main__":
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS films (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                tag TEXT,
                rating TEXT,
                embedding JSONB 
            );
        """)
        conn.commit()


    num_processes = multiprocessing.cpu_count() - 4
    pages_per_process = 1186 // num_processes

    page_ranges = []
    for i in range(num_processes):
        start_page = i * pages_per_process + 1
        if i == num_processes - 1:
            end_page = 1186
        else:
            end_page = (i + 1) * pages_per_process
        page_ranges.append((start_page, end_page))

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(requests_to_page, page_ranges)

    print("Все процессы завершены.")
