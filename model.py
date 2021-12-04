import numpy as np

from gensim import corpora, models
from gensim.models import CoherenceModel

from preprocessing import text_preprocessing


class LDA:
    def __init__(self, version: str):
        assert version in ["2020.2", "2020.3"], "Available options are '2020.2', '2020.3'."
        df = text_preprocessing(version)
        self.issues = df.summary
        self.model = None
        self.best_topic_number = None
        self.dictionary = None

    def __get_corpus(self):
        dictionary = corpora.Dictionary(self.issues)
        corpus = [dictionary.doc2bow(text) for text in self.issues]
        tfidf_model = models.TfidfModel(corpus)
        tfidf = tfidf_model[corpus]
        return dictionary, tfidf

    def __evaluate(self, model, coherence) -> float:
        coherence_model = CoherenceModel(
            model=model, texts=self.issues, dictionary=self.dictionary, coherence=coherence)
        return coherence_model.get_coherence()

    def fit(self, min_topics: int = 5, max_topics: int = 12, coherence: str = "c_npmi", n_topics: int = None):
        self.dictionary, tfidf = self.__get_corpus()
        coherence_scores = np.array([[i, 0] for i in range(min_topics, max_topics + 1)], dtype=np.float32)

        if not n_topics:
            for idx, n_topics in enumerate(range(min_topics, max_topics + 1)):
                model = models.LdaModel(tfidf, id2word=self.dictionary, num_topics=n_topics)
                coherence_scores[idx, 1] = self.__evaluate(model, coherence)
            max_coh_score = np.max(coherence_scores[:, 1])
            best_topic_number = coherence_scores[:, 0][np.argmax(coherence_scores[:, 1])]
            print(f"Maximum coherence score: {max_coh_score} with {best_topic_number} topics.")
            self.best_topic_number = best_topic_number
        else:
            self.best_topic_number = n_topics
        self.model = models.LdaModel(tfidf, id2word=self.dictionary, num_topics=self.best_topic_number)
        return

    def get_top_words(self):
        all_keywords = []
        for i in range(int(self.best_topic_number)):
            topics = self.model.get_topic_terms(i)
            words_idx = np.array(topics)[:, 0]
            words = [self.dictionary.get(idx) for idx in words_idx]
            all_keywords.append(words)
        return all_keywords

