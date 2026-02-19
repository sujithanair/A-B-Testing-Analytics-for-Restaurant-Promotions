from kaggle.api.kaggle_api_extended import KaggleApi
import os

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

dataset = "chebotinaa/fast-food-marketing-campaign-ab-test"

# Download & unzip
api.dataset_download_files(dataset, path="dataset", unzip=True)

print("Files downloaded:", os.listdir("dataset"))
