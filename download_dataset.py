import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

ds = load_dataset("yahma/alpaca-cleaned", cache_dir="./datasets/alpaca-cleaned")