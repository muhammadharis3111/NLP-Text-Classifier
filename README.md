# NLP Sentiment Classifier

### Technical Implementation
The pipeline focuses on three different vectorization strategies to determine which holds up best against unseen data:
* [cite_start]**TF-IDF:** Configured with a maximum of 10,000 features to capture discriminative unigram patterns[cite: 20].
* [cite_start]**Word2Vec:** Utilized Google News 300d pre-trained embeddings to leverage dense semantic relationships[cite: 21].
* [cite_start]**GloVe:** Utilized Wikipedia 100d pre-trained word vectors for mean-pooling document representation[cite: 22].

### Engineering Challenges
* [cite_start]**HTML Markup:** Because the IMDB dataset contains raw HTML, a specific preprocessing step was implemented to strip tags before normalization[cite: 13].
* [cite_start]**Semantic Normalization:** We opted for WordNetLemmatizer rather than simple stemming to ensure words were reduced to accurate base forms, preserving the semantic integrity needed for Word2Vec and GloVe.
* [cite_start]**Real-time Evaluation:** The Streamlit interface was built to handle CSV uploads of up to 200MB, allowing for side-by-side comparison of how different vectorizers classify the same input[cite: 10, 35].

### Final Performance Metrics
[cite_start]The following results were recorded using a Logistic Regression classifier across the IMDB test set[cite: 9, 23]:

| Method | Accuracy | Correct | Incorrect |
| :--- | :--- | :--- | :--- |
| TF-IDF | 88.09% | 22,023 | 2,977 |
| Word2Vec | 82.54% | 20,635 | 4,365 |
| GloVe | 79.82% | 19,955 | 5,045 |

### Setup and Usage
1. Clone the repository:
   git clone https://github.com/yourusername/nlp-robust-classification.git

2. Install the requirements:
   pip install -r requirements.txt

3. Download required NLTK resources:
   import nltk
   nltk.download('wordnet')
   nltk.download('stopwords')
   nltk.download('punkt')

4. Run the Streamlit application:
   streamlit run app.py

---
**Owner:** Muhammad Haris.
