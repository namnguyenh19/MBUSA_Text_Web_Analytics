import pandas as pd


def preprocess_columns(reviews: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the review data.

        Args:
            reviews: Source data.
        Returns:
            Preprocessed data.

    """

    # Rename columns of data frame
    reviews = reviews.rename(
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
    assert reviews["id"].is_unique, "Review identifier must be unique."
    reviews = reviews.set_index("id")

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


def clean_missing_values(reviews: pd.DataFrame) -> pd.DataFrame:
    """Fixes missing values in the data by omitting or imputing where required.

        Args:
            reviews: Preprocessed data.
        Returns:
            Data frame with missing values omitted or imputed.

    """

    # Remove reviews without review text
    reviews = reviews.dropna(subset=["review_text"])

    return reviews
