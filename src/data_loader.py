import os
import pandas as pd

# Base paths 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath("C:/Users/srush/Desktop/00_Labmentix/14_Voyage_ML/Voyage_Analytics/src/data_loader.py")))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

# Load raw data from the path in directory
def load_raw_data(
    flights_filename: str = "flights.csv",
    hotels_filename: str = "hotels.csv",
    users_filename: str = "users.csv",
):
    """Load raw CSVs from data/raw/ as DataFrames."""

    flights_path = os.path.join(DATA_DIR, flights_filename)
    hotels_path = os.path.join(DATA_DIR, hotels_filename)
    users_path = os.path.join(DATA_DIR, users_filename)

    flights = pd.read_csv(flights_path)
    hotels = pd.read_csv(hotels_path)
    users = pd.read_csv(users_path)

    return flights, hotels, users

# Left join on table flights with table users
def join_flights_users(flights: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Join flights with users on userCode/code."""

    return flights.merge(
        users,
        left_on="userCode",
        right_on="code",
        how="left",
        suffixes=("", "_user"),
    )

# Left join on table flights with table users
def join_hotels_users(hotels: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Join hotels with users on userCode/code."""

    return hotels.merge(
        users,
        left_on="userCode",
        right_on="code",
        how="left",
        suffixes=("", "_user"),
    )


print(DATA_DIR)