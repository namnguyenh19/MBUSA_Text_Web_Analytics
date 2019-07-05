import pandas as pd
import numpy as np
import typing


def wc(x):
    """Count words in string."""
    if x == np.nan:
        return np.nan
    else:
        return len(x.split())


def summarise(df):
    """Generate custom summary statistics for a DataFrame."""
    DIGITS = 3  # used for rounding
    dtypes = df.dtypes  # data types
    nulls = df.isnull().sum()  # number of null values
    not_nulls = df.notnull().sum()  # number of not-null values
    avg_wcs = df.apply(
        lambda col: np.mean(col.apply(lambda x: wc(x) if type(x) == str else np.nan))
    )
    wcs = df.apply(
        lambda col: np.mean(col.apply(lambda x: wc(x) if type(x) == str else np.nan))
    )
    uniques = df.apply(lambda col: len(col.unique()))
    summary1 = pd.DataFrame(
        {"dtype": dtypes, "n_null": nulls, "n_valid": not_nulls, "unique": uniques}
    )
    summary2 = df.describe().T.drop("count", axis=1)
    skews = pd.DataFrame({"skew": df.skew()})
    return round(pd.concat([summary1, summary2, skews], axis=1, sort=False), DIGITS)


def preprocess_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the review data.

        Args:
            reviews: source data.
        Returns:
            Preprocessed data.

    """

    # Rename columns of data frame
    df = df.rename(
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
    assert df["id"].is_unique, "Review identifier must be unique."
    df = df.set_index("id")

    # Lower case of category hierarchy
    CATEGORIES = [
        "product_category_division",
        "product_category_department",
        "product_category_class",
    ]
    df[CATEGORIES] = df[CATEGORIES].apply(lambda x: x.str.lower(), axis=0)

    # Replace incorrect spelling of 'intimates'
    df["product_category_division"] = df["product_category_division"].replace(
        "initmates", "intimates"
    )

    # Change category variables to category type
    df[CATEGORIES] = df[CATEGORIES].astype("category")

    return df


# Find missing text fields
def text_fields(df: pd.DataFrame) -> list:
    """Returns the list of text fields in the DataFrame."""
    return df.select_dtypes("object")


def drop_missing_text(df):
    """Drop any reviews that are missing entries for all text fields."""
    return df.dropna(subset=text_fields(df).columns, how="all")


# Find missing categories
def category_list(df: pd.DataFrame) -> list:
    """Returns the list of category fields in the DataFrame."""
    return list(df.filter(like="category").columns)


def uncategorised(df):
    """Returns the reviews that do not have product categories."""
    return df[df.filter(category_list(df)).isnull().any(1)]


def search(df: pd.DataFrame, term: str) -> pd.DataFrame:
    """Searches the review text and title fields for a token string.

    Args:
        df: the DataFrame in which to run the seaarch.
        term: the keyword used to filter the reviews.

    Returns:
        A DataFrame with the subset of reviews that match the search.
        If no rows match the search, returns None.
    """
    matches = text_fields(df).apply(lambda x: x.str.contains(term)).any(1)
    if any(matches):
        return df.loc[matches]
    else:
        return None


def top_category(df: pd.DataFrame, ID: int) -> list:
    """Get the most frequent category hierarchy (division, department and class) for a
    given product ID.

    Args:
        df: A Pandas dataframe with the preprocessed reviews data.
        ID: The product ID

    Returns:
        A dictionary with two key-value pairs:
            'category': The category hierarchy as a list [division, department, class].
            'sample_size': The number of similar reviews.
            'keyword': The keyword used to search the table for similar reviews.
    """

    # Search terms
    KEYWORDS = {
        72: "socks",
        492: "hoodie",
        152: "leg warmer",
        184: "tights",
        772: "sweatshirt",
        665: "underwear",
        136: "socks",
    }

    # Product category list
    CATEGORIES = category_list(df)

    # Get similar reviews based on keywords in review title and body text
    keyword = KEYWORDS[ID]
    similar_reviews = search(df, keyword)
    n_similar = len(similar_reviews)

    # Get the most frequent hierarchy using a pivot table
    pivot = (
        similar_reviews.filter(CATEGORIES)
        .pivot_table(index=CATEGORIES, aggfunc="size")
        .sort_values(ascending=False)
    )
    levels = pivot.index.levels  # category levels
    codes = pivot.index.codes  # integer codes mapping to category levels
    top_code = [x[0] for x in codes]
    top_level = [levels[i][top_code[i]] for i in range(len(top_code))]
    return {"category": top_level, "sample_size": n_similar, "keyword": keyword}


def impute_categories(df: pd.DataFrame) -> pd.DataFrame:

    # Get products without category
    missing = list(uncategorised(df).index)

    if not missing:
        print("No uncategorised reviews.")
        return df
    else:
        # Get category hierarchy and IDs of missing products
        CATEGORIES = category_list(df)
        missing_products = uncategorised(df).product_id.unique()

        # Impute categories for missing products
        imputed = list(
            zip(*[top_category(df=df, ID=i)["category"] for i in missing_products])
        )

        # Create lookup table for imputed values
        imputed_lookup = pd.DataFrame(
            {CATEGORIES[i]: imputed[i] for i in range(len(imputed))},
            index=missing_products,
        )

        # Copy the table and add imputed categories
        df_i = df.copy(deep=True)
        for cat in CATEGORIES:
            df_i.loc[missing, cat] = df.product_id.map(imputed_lookup[cat])

        return df_i


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fixes missing values in the data by omitting or imputing where required.

        Args:
            df: Preprocessed data.
        Returns:
            Data frame with missing values omitted or imputed.

    """

    # Drop reviews with missing text fields
    df_clean = drop_missing_text(df)

    # Impute missing categories
    df_clean = impute_categories(df)

    return df_clean
