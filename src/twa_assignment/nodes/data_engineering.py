import pandas as pd


def preprocess_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the review data.

        Args:
            reviews: source data.
        Returns:
            Preprocessed data.

    """

    # Rename columns of data frame
    reviews = reviews.rename(
        columns={
            "Unnamed: 0": "review_id",
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

    # Lower case of category hierarchy
    category_hierarchy = [
        "product_category_division",
        "product_category_department",
        "product_category_class",
    ]
    reviews[category_hierarchy] = reviews[category_hierarchy].apply(
        lambda x: x.str.lower(), axis=0
    )

    # Replace incorrect spelling of 'intimates'
    reviews["product_category_division"] = reviews["product_category_division"].replace(
        "initmates", "intimates"
    )

    return reviews
