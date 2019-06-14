import pandas as pd


def preprocess_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the review data.

        Args:
            reviews: Source data.
        Returns:
            preproc_reviews: Preprocessed data.

    """

    # Rename columns of data frame
    preproc_reviews = reviews.rename(
        columns={
            "Unnamed: 0": "id",
            "Clothing ID": "product_id",
            "Age": "author_age",
            "Title": "review_title",
            "Review Text": "review_text",
            "Rating": "star_rating",
            "Recommended IND": "recommend_flag",
            "Positive Feedback Count": "upvotes",
            "Division Name": "product_category_division",
            "Department Name": "product_category_department",
            "Class Name": "product_category_class",
        }
    )

    # Update review index
    assert preproc_reviews["id"].is_unique, "Review identifier must be unique."
    preproc_reviews = preproc_reviews.set_index("id")

    # Lower case of category hierarchy
    category_hierarchy = [
        "product_category_division",
        "product_category_department",
        "product_category_class",
    ]
    preproc_reviews[category_hierarchy] = preproc_reviews[category_hierarchy].apply(
        lambda x: x.str.lower(), axis=0
    )

    # Replace incorrect spelling of 'intimates'
    preproc_reviews["product_category_division"] = preproc_reviews[
        "product_category_division"
    ].replace("initmates", "intimates")

    # Remove reviews without review text
    preproc_reviews = preproc_reviews.dropna(subset=["review_text"])

    return preproc_reviews
