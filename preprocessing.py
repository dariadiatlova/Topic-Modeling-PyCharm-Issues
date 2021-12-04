import pandas as pd
import numpy as np
import nltk
import itertools

from string import punctuation
punctuation += "``"
punctuation += "''"

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

from data import DATA_ROOT_DIRECTORY_PATH
from os.path import join
from typing import List, Optional


def create_datasets_for_two_version() -> None:
    """
    Call this function to create 2 csv files with columns: "summary" and "version" for 2020.2 and 2020.3 versions.
    Function read initial dataset and save created to the data directory in the repository root.
    :return: None
    """
    file_path = join(DATA_ROOT_DIRECTORY_PATH, "pycharm_issues.json")
    lines_df = pd.read_json(file_path, lines=True).rename(columns={"Affected versions": "version"})

    summaries = lines_df.summary
    versions_list = lines_df.version

    df_second_version = pd.DataFrame(columns=["summary", "version"])
    df_third_version = pd.DataFrame(columns=["summary", "version"])

    n2 = 0
    n3 = 0

    for i, summary, versions in zip(range(lines_df.shape[0]), summaries, versions_list):

        if "2020.2" in versions:
            df_second_version = df_second_version.append(
                pd.DataFrame([{"summary": summary, "version": "2020.2"}]), ignore_index=True)
            n2 += 1

        if "2020.3" in versions:
            df_third_version = df_third_version.append(
                pd.DataFrame([{"summary": summary, "version": "2020.3"}]), ignore_index=True)
            n3 += 1

    print(f"{n2} issued for version 2020.2 found. \n{n3} issued for version 2020.3 found.")

    df_second_version.to_csv(join(DATA_ROOT_DIRECTORY_PATH, "second_version_issue_summaries.csv"), index=False)
    df_third_version.to_csv(join(DATA_ROOT_DIRECTORY_PATH, "third_version_issue_summaries.csv"), index=False)


def text_preprocessing(version: str):
    """
    Function reads dataset (version 2020.2 or 2020.3) and returns it's df-representation but after
    lemmatization and lower representstion of text without stopwords.
    :param version: str, 2020.2 or 2020.3 – version that has a relation to the issue.
    :return: pd.DataFrame,
    """

    assert version in ["2020.2", "2020.3"], "Available options are '2020.2' and '2020.3'"

    if version == "2020.2":
        dataset_path = join(DATA_ROOT_DIRECTORY_PATH, "second_version_issue_summaries.csv")
    else:
        dataset_path = join(DATA_ROOT_DIRECTORY_PATH, "third_version_issue_summaries.csv")

    df = pd.read_csv(dataset_path)
    summaries = df.summary.to_list()

    lematizer = nltk.stem.WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    stopwords_set.update(["python", "pycharm", "code", "n't", "doe"])

    text = [word_tokenize(issue) for issue in summaries]
    text = [[token.strip().lower() for token in issue] for issue in text]
    lem_text = [[lematizer.lemmatize(token) for token in issue] for issue in text]
    no_stopwords_text = [[token for token in issue if token not in stopwords_set] for issue in lem_text]
    no_punctuation_text = [[token for token in issue if token not in punctuation] for issue in no_stopwords_text]

    df["summary"] = pd.Series(no_punctuation_text)
    return df


def get_n_most_popular_words(version: str, n_most_popular: int = 10, return_counts: bool = True) -> Optional[List[str]]:

    assert version in ["2020.2", "2020.3"], "Available options are '2020.2' and '2020.3'"

    df = text_preprocessing(version)

    flat_list = list(itertools.chain(*df.summary.to_list()))
    counter_of_flat_list = Counter(flat_list)

    most_popular_pairs = np.array(counter_of_flat_list.most_common(n_most_popular))
    most_popular_words = np.array(most_popular_pairs[:, 0], dtype=str)
    most_popular_counts = np.array(most_popular_pairs[:, 1], dtype=int)

    print(f"Total word count in dataset: {sum(counter_of_flat_list.values())}.")

    if return_counts:
        return most_popular_words, most_popular_counts
    else:
        return most_popular_words
