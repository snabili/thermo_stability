import os

API_KEY = os.environ.get("MPR_APIKEY")

if API_KEY is None:
    raise ValueError("API key not found. Please set the 'MPR_APIKEY' environment variable.")
