#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/TinyLlama-1.1B-Chat-v1.0', local_dir="./models/TinyLlama-1.1B-Chat-v1.0")