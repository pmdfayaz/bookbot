BookieBot

BookieBot is an AI-based book chatbot that provides quick summaries and smart book recommendations using natural language input. BookieBot understands user queries and responds with relevant book insights using fuzzy matching, intent classification, and NLP techniques.

# System Requirements

- Python 3.8 or higher
- A CSV dataset of books (`books.csv`) with the following fields:
  - `Book`, `Author`, `Genre1`, `Genre2`, `Genre3`, `Description`


# Libraries Used

Install the required Python libraries using pip:

```bash
pip install pandas numpy scikit-learn fuzzywuzzy nltk spacy
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
# bookbot
