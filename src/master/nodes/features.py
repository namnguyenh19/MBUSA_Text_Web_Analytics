import pandas as pd
import numpy as np
from nltk.corpus import stopwords

stops = stopwords.words("english")

from string import punctuation


def _word_count(text):
    return len(text.split()) if text != np.nan else np.nan


def _char_count(text):
    return len(text)


def _avg_word(text):
    words = text.split()
    return sum(len(word) for word in words) / len(words)


def _stop_count(text):
    return len([word for word in text.split() if word in stops])


def _num_count(text):
    return len([word for word in text.split() if word.isdigit()])


def _upper_count(text):
    return len([word for word in text.split() if word.isupper()])


def _punc_count(text):
    return len([word for word in text.split() if word in punctuation])


def summarise_text(reviews: pd.DataFrame) -> pd.DataFrame:
    """Extracts summary features from the review text.

    These features are on simple descriptive statistics of the review text
    or aspects of the documents that will be removed during tokenisation, such as:
        - Raw character count
        - Raw word count
        - Number of punctuation marks
        - Number of capital letters
        - Number of numeric characters

        Args:
            reviews: Preprocessed data.
        Returns:
            Pandas data frame with summary features added as new columns.

    """

    # Generate summary statistics
    reviews["text_word_count"] = reviews["review_text"].apply(_word_count)
    reviews["text_char_count"] = reviews["review_text"].apply(_char_count)
    reviews["text_avg_word"] = reviews["review_text"].apply(_avg_word)
    reviews["text_stop_count"] = reviews["review_text"].apply(_stop_count)
    reviews["text_stop_freq"] = reviews["text_stop_count"] / reviews["text_word_count"]
    reviews["text_num_count"] = reviews["review_text"].apply(_num_count)
    reviews["text_upper_count"] = reviews["review_text"].apply(_upper_count)
    reviews["text_punc_count"] = reviews["review_text"].apply(_punc_count)

    # Filter summary statistics
    reviews = reviews.filter(like="text")

    return reviews
