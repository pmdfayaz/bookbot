import spacy
from fuzzywuzzy import fuzz
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import scrolledtext

nlp = spacy.load("en_core_web_sm")
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

books_df = pd.read_csv("books.csv")
books_df = books_df[books_df['Description'].notna() & (books_df['Description'].str.strip() != "")].copy()
books_df['Description_filled'] = books_df['Description'].fillna('')

conversation_state = "initial"

def classify_intent(user_input):
    summary_keywords = ["summary", "summarize"]
    recommend_keywords = ["recommend", "suggest"]

    user_input_lower = user_input.lower()
    summary_score = max([fuzz.partial_ratio(user_input_lower, keyword) for keyword in summary_keywords])
    recommend_score = max([fuzz.partial_ratio(user_input_lower, keyword) for keyword in recommend_keywords])

    if summary_score > recommend_score and summary_score > 60:
        return "summary"
    elif recommend_score > summary_score and recommend_score > 60:
        return "recommend"
    return "general"

def extract_best_matching_book_title(user_input):
    user_input_lower = user_input.lower()
    best_match_title = None
    highest_score = 0
    for book_name in books_df['Book'].unique():
        score = fuzz.token_set_ratio(user_input_lower, book_name.lower())
        if score > highest_score:
            highest_score = score
            best_match_title = book_name
    if highest_score > 80:
        return best_match_title
    else:
        print(f"[DEBUG] No good title match. Best was: '{best_match_title}' with score {highest_score}")
        return None

def extract_preferences(user_input):
    user_input_lower = user_input.lower()
    found_genres = []
    found_author = None

    available_genres = []
    for col in ['Genre1', 'Genre2', 'Genre3']:
        if col in books_df.columns:
            available_genres.extend(books_df[col].dropna().unique().tolist())
    available_genres = list(set([g.lower() for g in available_genres if isinstance(g, str)]))

    for genre in available_genres:
        if genre in user_input_lower:
            found_genres.append(genre)

    available_authors = books_df['Author'].dropna().unique().tolist()
    best_score = 0
    for author in available_authors:
        score = fuzz.token_set_ratio(user_input_lower, author.lower())
        if score > best_score and score > 80:
            best_score = score
            found_author = author

    return found_genres, found_author

def get_book_summary(book_title):
    try:
        book_row = books_df[books_df['Book'] == book_title].iloc[0]
        return f"Book Title: {book_row['Book']}\nSummary: {book_row['Description']}"
    except IndexError:
        return "Sorry, I could not find a summary for this book."

def recommend_books(book_title=None, genres=None, author=None, max_recs=5):
    if book_title:
        target_book_row = books_df[books_df['Book'] == book_title]
        if target_book_row.empty:
            return ["Book not found to base recommendations on."]

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(books_df['Description_filled'])

        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        idx = target_book_row.index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        target_genres = [target_book_row.iloc[0][col] for col in ['Genre1', 'Genre2', 'Genre3'] if pd.notna(target_book_row.iloc[0][col])]
        rec_books = []
        for i, score in sim_scores:
            if i == idx or score < 0.3:
                continue
            rec_row = books_df.iloc[i]
            rec_genres = [rec_row[col] for col in ['Genre1', 'Genre2', 'Genre3'] if pd.notna(rec_row[col])]
            if not target_genres or any(g in target_genres for g in rec_genres):
                rec_books.append(rec_row['Book'])
            if len(rec_books) >= max_recs:
                break
        return rec_books or ["Sorry, couldn't find similar books with matching genres."]

    elif genres or author:
        filtered_books = books_df.copy()
        if genres:
            genre_mask = pd.Series(False, index=filtered_books.index)
            for g in genres:
                for col in ['Genre1', 'Genre2', 'Genre3']:
                    genre_mask |= filtered_books[col].astype(str).str.contains(g, case=False, na=False)
            filtered_books = filtered_books[genre_mask]
        if author:
            filtered_books = filtered_books[filtered_books['Author'].astype(str).str.contains(author, case=False, na=False)]
        if filtered_books.empty:
            return ["Sorry, I couldn't find books matching those criteria."]
        return filtered_books['Book'].head(max_recs).tolist()
    else:
        return books_df['Book'].sample(min(max_recs, len(books_df))).tolist()

def chatbot_response(user_input):
    global conversation_state

    if user_input.lower() in ['exit', 'quit', 'çık']:
        return "Goodbye! Hope to see you again soon."

    if conversation_state == "awaiting_book_title_for_summary":
        book_title = extract_best_matching_book_title(user_input)
        if book_title:
            conversation_state = "initial"
            return get_book_summary(book_title)
        return "I couldn't find that book. Try the full title or another one?"

    elif conversation_state == "awaiting_preference_for_recommendation":
        book_title = extract_best_matching_book_title(user_input)
        genres, author = extract_preferences(user_input)
        if book_title:
            conversation_state = "initial"
            recs = recommend_books(book_title=book_title)
            return f"Based on '{book_title}', I recommend: " + ", ".join(recs)
        elif genres or author:
            conversation_state = "initial"
            recs = recommend_books(genres=genres, author=author)
            info = []
            if genres: info.append("genre(s): " + ", ".join(genres))
            if author: info.append("author: " + author)
            return f"I recommend: " + ", ".join(recs)
        return "Can you tell me a book, a genre or author you enjoy?"

    intent = classify_intent(user_input)
    if intent == "summary":
        book_title = extract_best_matching_book_title(user_input)
        if book_title:
            return get_book_summary(book_title)
        conversation_state = "awaiting_book_title_for_summary"
        return "Which book's summary are you looking for?"

    elif intent == "recommend":
        book_title = extract_best_matching_book_title(user_input)
        genres, author = extract_preferences(user_input)
        if book_title:
            recs = recommend_books(book_title=book_title)
            return f"Based on '{book_title}', I recommend: " + ", ".join(recs)
        elif genres or author:
            recs = recommend_books(genres=genres, author=author)
            info = []
            if genres: info.append("genre(s): " + ", ".join(genres))
            if author: info.append("author: " + author)
            return f"Based on your interest in {' and '.join(info)}, I recommend: " + ", ".join(recs)
        conversation_state = "awaiting_preference_for_recommendation"
        return "What genre or author do you like?"

    return "Hello! I'm BookieBot. I can help you with book summaries or recommend new books. What would you like to do?"


class ChatBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Book Chatbot")

        self.bg = '#2e2e2e'; self.fg = '#ffffff'
        self.text_bg = '#3c3c3c'; self.input_bg = '#4a4a4a'
        self.chat_color = '#28a745'; self.user_color = '#00aaff'

        self.root.configure(bg=self.bg)
        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled',
                                                   bg=self.text_bg, fg=self.fg, font=('Arial', 12))
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.user_input = tk.Entry(root, width=100, bg=self.input_bg, fg=self.fg, font=('Arial', 12),
                                   insertbackground=self.fg)
        self.user_input.pack(padx=10, pady=10, fill=tk.X)
        self.user_input.bind("<Return>", self.send_message)

        self.send_button = tk.Button(root, text="Enter", command=self.send_message,
                                     bg="#007bff", fg=self.fg, font=('Arial', 12, 'bold'))
        self.send_button.pack(padx=10, pady=10)

        self.chat_area.tag_config('chatbot', foreground=self.chat_color, font=('Arial', 12, 'bold'))
        self.chat_area.tag_config('user', foreground=self.user_color, font=('Arial', 12, 'italic'))

        self.display_message("Chatbot", "Hello! How can I help you?")

    def send_message(self, event=None):
        msg = self.user_input.get()
        if msg.strip() == "":
            return
        if msg.lower() == "exit":
            self.root.quit()
        self.display_message("You", msg)
        self.user_input.delete(0, tk.END)
        response = chatbot_response(msg)
        self.display_message("Chatbot", response)

    def display_message(self, sender, msg):
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, f"{sender}: {msg}\n")
        self.chat_area.yview(tk.END)
        self.chat_area.config(state='disabled')
        tag = 'chatbot' if sender == "Chatbot" else 'user'
        self.chat_area.tag_add(tag, f"{self.chat_area.index('end')}-2l", f"{self.chat_area.index('end')}-1l")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatBotGUI(root)
    root.mainloop()
