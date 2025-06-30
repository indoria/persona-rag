### azureuser@flaskAI:~/persona-rag$ python3 -m venv venv
### azureuser@flaskAI:~/persona-rag$ source venv/bin/activate
### (venv) azureuser@flaskAI:~/persona-rag$ pip install -r requirements.txt
```
Collecting openai
  Downloading openai-1.91.0-py3-none-any.whl (735 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 735.8/735.8 KB 5.6 MB/s eta 0:00:00
Requirement already satisfied: pysqlite3-binary in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 16)) (0.5.4)
Requirement already satisfied: gunicorn in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 19)) (23.0.0)
Collecting tiktoken
  Downloading tiktoken-0.9.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 10.5 MB/s eta 0:00:00
Collecting jiter<1,>=0.4.0
  Downloading jiter-0.10.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (352 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 352.5/352.5 KB 9.6 MB/s eta 0:00:00
Collecting flask
  Downloading flask-3.1.1-py3-none-any.whl (103 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 103.3/103.3 KB 2.3 MB/s eta 0:00:00
Collecting spacy
  Downloading spacy-3.8.7-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (31.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 31.5/31.5 MB 35.3 MB/s eta 0:00:00
Collecting nltk
  Downloading nltk-3.9.1-py3-none-any.whl (1.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5/1.5 MB 50.1 MB/s eta 0:00:00
Collecting scikit-learn
  Downloading scikit_learn-1.7.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.9/12.9 MB 66.0 MB/s eta 0:00:00
Collecting textblob
  Downloading textblob-0.19.0-py3-none-any.whl (624 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 624.3/624.3 KB 28.8 MB/s eta 0:00:00
Collecting sentence-transformers
  Downloading sentence_transformers-4.1.0-py3-none-any.whl (345 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 345.7/345.7 KB 24.5 MB/s eta 0:00:00
Collecting chromadb
  Downloading chromadb-1.0.13-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (19.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 19.3/19.3 MB 53.7 MB/s eta 0:00:00
Collecting numpy
  Downloading numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.8/16.8 MB 60.5 MB/s eta 0:00:00
Collecting transformers[torch]>=4.28.0
  Downloading transformers-4.52.4-py3-none-any.whl (10.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.5/10.5 MB 62.0 MB/s eta 0:00:00
Collecting torch>=2.0.0
  Downloading torch-2.7.1-cp310-cp310-manylinux_2_28_x86_64.whl (821.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 821.2/821.2 MB 1.1 MB/s eta 0:00:00
Collecting accelerate>=0.26.0
  Downloading accelerate-1.8.1-py3-none-any.whl (365 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 365.3/365.3 KB 266.5 kB/s eta 0:00:00
Collecting python-dotenv
  Downloading python_dotenv-1.1.0-py3-none-any.whl (20 kB)
Collecting pysqlite3-binary
  Downloading pysqlite3_binary-0.5.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.2/5.2 MB 63.2 MB/s eta 0:00:00
Collecting gunicorn
  Downloading gunicorn-23.0.0-py3-none-any.whl (85 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85.0/85.0 KB 6.6 MB/s eta 0:00:00
Collecting click>=8.1.3
  Downloading click-8.2.1-py3-none-any.whl (102 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 102.2/102.2 KB 8.5 MB/s eta 0:00:00
Collecting markupsafe>=2.1.1
  Downloading MarkupSafe-3.0.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (20 kB)
Collecting jinja2>=3.1.2
  Downloading jinja2-3.1.6-py3-none-any.whl (134 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 134.9/134.9 KB 10.3 MB/s eta 0:00:00
Collecting blinker>=1.9.0
  Downloading blinker-1.9.0-py3-none-any.whl (8.5 kB)
Collecting werkzeug>=3.1.0
  Downloading werkzeug-3.1.3-py3-none-any.whl (224 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 224.5/224.5 KB 19.6 MB/s eta 0:00:00
Collecting itsdangerous>=2.2.0
  Downloading itsdangerous-2.2.0-py3-none-any.whl (16 kB)
Collecting tqdm<5.0.0,>=4.38.0
  Downloading tqdm-4.67.1-py3-none-any.whl (78 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.5/78.5 KB 6.1 MB/s eta 0:00:00
Collecting packaging>=20.0
  Downloading packaging-25.0-py3-none-any.whl (66 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 66.5/66.5 KB 4.3 MB/s eta 0:00:00
Collecting catalogue<2.1.0,>=2.0.6
  Downloading catalogue-2.0.10-py3-none-any.whl (17 kB)
Collecting spacy-loggers<2.0.0,>=1.0.0
  Downloading spacy_loggers-1.0.5-py3-none-any.whl (22 kB)
Collecting langcodes<4.0.0,>=3.2.0
  Downloading langcodes-3.5.0-py3-none-any.whl (182 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 183.0/183.0 KB 13.7 MB/s eta 0:00:00
Collecting typer<1.0.0,>=0.3.0
  Downloading typer-0.16.0-py3-none-any.whl (46 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46.3/46.3 KB 3.4 MB/s eta 0:00:00
Collecting preshed<3.1.0,>=3.0.2
  Downloading preshed-3.0.10-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (795 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 795.1/795.1 KB 35.2 MB/s eta 0:00:00
Collecting spacy-legacy<3.1.0,>=3.0.11
  Downloading spacy_legacy-3.0.12-py2.py3-none-any.whl (29 kB)
Collecting cymem<2.1.0,>=2.0.2
  Downloading cymem-2.0.11-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (204 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 204.8/204.8 KB 14.1 MB/s eta 0:00:00
Requirement already satisfied: setuptools in ./venv/lib/python3.10/site-packages (from spacy->-r requirements.txt (line 2)) (59.6.0)
Collecting srsly<3.0.0,>=2.4.3
  Downloading srsly-2.5.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 42.1 MB/s eta 0:00:00
Collecting murmurhash<1.1.0,>=0.28.0
  Downloading murmurhash-1.0.13-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (117 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 117.1/117.1 KB 9.6 MB/s eta 0:00:00
Collecting weasel<0.5.0,>=0.1.0
  Downloading weasel-0.4.1-py3-none-any.whl (50 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.3/50.3 KB 4.2 MB/s eta 0:00:00
Collecting requests<3.0.0,>=2.13.0
  Downloading requests-2.32.4-py3-none-any.whl (64 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 64.8/64.8 KB 5.5 MB/s eta 0:00:00
Collecting pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4
  Downloading pydantic-2.11.7-py3-none-any.whl (444 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 444.8/444.8 KB 25.8 MB/s eta 0:00:00
Collecting thinc<8.4.0,>=8.3.4
  Downloading thinc-8.3.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.1/4.1 MB 71.6 MB/s eta 0:00:00
Collecting wasabi<1.2.0,>=0.9.1
  Downloading wasabi-1.1.3-py3-none-any.whl (27 kB)
Collecting regex>=2021.8.3
  Downloading regex-2024.11.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (781 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 781.7/781.7 KB 37.3 MB/s eta 0:00:00
Collecting joblib
  Downloading joblib-1.5.1-py3-none-any.whl (307 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 307.7/307.7 KB 18.8 MB/s eta 0:00:00
Collecting threadpoolctl>=3.1.0
  Downloading threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Collecting scipy>=1.8.0
  Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (37.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.7/37.7 MB 28.7 MB/s eta 0:00:00
Collecting Pillow
  Downloading pillow-11.2.1-cp310-cp310-manylinux_2_28_x86_64.whl (4.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.6/4.6 MB 69.0 MB/s eta 0:00:00
Collecting typing_extensions>=4.5.0
  Downloading typing_extensions-4.14.0-py3-none-any.whl (43 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 43.8/43.8 KB 3.6 MB/s eta 0:00:00
Collecting huggingface-hub>=0.20.0
  Downloading huggingface_hub-0.33.0-py3-none-any.whl (514 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 514.8/514.8 KB 28.2 MB/s eta 0:00:00
Collecting build>=1.0.3
  Downloading build-1.2.2.post1-py3-none-any.whl (22 kB)
Collecting uvicorn[standard]>=0.18.3
  Downloading uvicorn-0.34.3-py3-none-any.whl (62 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.4/62.4 KB 5.8 MB/s eta 0:00:00
Collecting importlib-resources
  Downloading importlib_resources-6.5.2-py3-none-any.whl (37 kB)
Collecting tokenizers>=0.13.2
  Downloading tokenizers-0.21.1-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.0/3.0 MB 60.8 MB/s eta 0:00:00
Collecting mmh3>=4.0.1
  Downloading mmh3-5.1.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (99 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 99.2/99.2 KB 7.6 MB/s eta 0:00:00
Collecting overrides>=7.3.1
  Downloading overrides-7.7.0-py3-none-any.whl (17 kB)
Collecting pyyaml>=6.0.0
  Downloading PyYAML-6.0.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (751 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 751.2/751.2 KB 34.0 MB/s eta 0:00:00
Collecting orjson>=3.9.12
  Downloading orjson-3.10.18-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (132 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 132.8/132.8 KB 4.2 MB/s eta 0:00:00
Collecting opentelemetry-sdk>=1.2.0
  Downloading opentelemetry_sdk-1.34.1-py3-none-any.whl (118 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 118.5/118.5 KB 8.1 MB/s eta 0:00:00
Collecting jsonschema>=4.19.0
  Downloading jsonschema-4.24.0-py3-none-any.whl (88 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 88.7/88.7 KB 6.6 MB/s eta 0:00:00
Collecting tenacity>=8.2.3
  Downloading tenacity-9.1.2-py3-none-any.whl (28 kB)
Collecting pybase64>=1.4.1
  Downloading pybase64-1.4.1-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (68 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 68.7/68.7 KB 5.4 MB/s eta 0:00:00
Collecting posthog>=2.4.0
  Downloading posthog-5.4.0-py3-none-any.whl (105 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 105.4/105.4 KB 9.9 MB/s eta 0:00:00
Collecting httpx>=0.27.0
  Downloading httpx-0.28.1-py3-none-any.whl (73 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 73.5/73.5 KB 7.1 MB/s eta 0:00:00
Collecting grpcio>=1.58.0
  Downloading grpcio-1.73.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (6.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.0/6.0 MB 75.9 MB/s eta 0:00:00
Collecting onnxruntime>=1.14.1
  Downloading onnxruntime-1.22.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (16.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.4/16.4 MB 64.5 MB/s eta 0:00:00
Collecting bcrypt>=4.0.1
  Downloading bcrypt-4.3.0-cp39-abi3-manylinux_2_34_x86_64.whl (284 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 284.2/284.2 KB 18.0 MB/s eta 0:00:00
Collecting pypika>=0.48.9
  Downloading PyPika-0.48.9.tar.gz (67 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 67.3/67.3 KB 4.9 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Collecting opentelemetry-api>=1.2.0
  Downloading opentelemetry_api-1.34.1-py3-none-any.whl (65 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 65.8/65.8 KB 5.3 MB/s eta 0:00:00
Collecting rich>=10.11.0
  Downloading rich-14.0.0-py3-none-any.whl (243 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 243.2/243.2 KB 16.7 MB/s eta 0:00:00
Collecting opentelemetry-exporter-otlp-proto-grpc>=1.2.0
  Downloading opentelemetry_exporter_otlp_proto_grpc-1.34.1-py3-none-any.whl (18 kB)
Collecting kubernetes>=28.1.0
  Downloading kubernetes-33.1.0-py2.py3-none-any.whl (1.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.9/1.9 MB 46.3 MB/s eta 0:00:00
Collecting safetensors>=0.4.3
  Downloading safetensors-0.5.3-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (471 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 471.6/471.6 KB 28.4 MB/s eta 0:00:00
Collecting filelock
  Downloading filelock-3.18.0-py3-none-any.whl (16 kB)
Collecting torch>=2.0.0
  Downloading torch-2.6.0-cp310-cp310-manylinux1_x86_64.whl (766.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 766.7/766.7 MB 1.1 MB/s eta 0:00:00
Collecting nvidia-cusparse-cu12==12.3.1.170
  Downloading nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl (207.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 207.5/207.5 MB 3.2 MB/s eta 0:00:00
Collecting nvidia-nvjitlink-cu12==12.4.127
  Downloading nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (21.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 21.1/21.1 MB 46.3 MB/s eta 0:00:00
Collecting triton==3.2.0
  Downloading triton-3.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (253.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 253.1/253.1 MB 4.2 MB/s eta 0:00:00
Collecting networkx
  Downloading networkx-3.4.2-py3-none-any.whl (1.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 54.1 MB/s eta 0:00:00
Collecting nvidia-cudnn-cu12==9.1.0.70
  Downloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl (664.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 664.8/664.8 MB 1.0 MB/s eta 0:00:00
Collecting nvidia-cuda-runtime-cu12==12.4.127
  Downloading nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (883 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 883.7/883.7 KB 4.1 MB/s eta 0:00:00
Collecting nvidia-cufft-cu12==11.2.1.3
  Downloading nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl (211.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 211.5/211.5 MB 4.7 MB/s eta 0:00:00
Collecting fsspec
  Downloading fsspec-2025.5.1-py3-none-any.whl (199 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 199.1/199.1 KB 15.5 MB/s eta 0:00:00
Collecting nvidia-cuda-nvrtc-cu12==12.4.127
  Downloading nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (24.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 24.6/24.6 MB 46.1 MB/s eta 0:00:00
Collecting nvidia-nccl-cu12==2.21.5
  Downloading nvidia_nccl_cu12-2.21.5-py3-none-manylinux2014_x86_64.whl (188.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 188.7/188.7 MB 5.3 MB/s eta 0:00:00
Collecting nvidia-cusparselt-cu12==0.6.2
  Downloading nvidia_cusparselt_cu12-0.6.2-py3-none-manylinux2014_x86_64.whl (150.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 150.1/150.1 MB 6.0 MB/s eta 0:00:00
Collecting nvidia-cuda-cupti-cu12==12.4.127
  Downloading nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (13.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.8/13.8 MB 57.9 MB/s eta 0:00:00
Collecting nvidia-cusolver-cu12==11.6.1.9
  Downloading nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl (127.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 127.9/127.9 MB 7.2 MB/s eta 0:00:00
Collecting sympy==1.13.1
  Downloading sympy-1.13.1-py3-none-any.whl (6.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.2/6.2 MB 65.3 MB/s eta 0:00:00
Collecting nvidia-cublas-cu12==12.4.5.8
  Downloading nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl (363.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 363.4/363.4 MB 1.9 MB/s eta 0:00:00
Collecting nvidia-nvtx-cu12==12.4.127
  Downloading nvidia_nvtx_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (99 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 99.1/99.1 KB 7.9 MB/s eta 0:00:00
Collecting nvidia-curand-cu12==10.3.5.147
  Downloading nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl (56.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 56.3/56.3 MB 18.6 MB/s eta 0:00:00
Collecting mpmath<1.4,>=1.1.0
  Downloading mpmath-1.3.0-py3-none-any.whl (536 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 536.2/536.2 KB 28.1 MB/s eta 0:00:00
Collecting psutil
  Downloading psutil-7.0.0-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (277 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 278.0/278.0 KB 19.7 MB/s eta 0:00:00
Collecting pyproject_hooks
  Downloading pyproject_hooks-1.2.0-py3-none-any.whl (10 kB)
Collecting tomli>=1.1.0
  Downloading tomli-2.2.1-py3-none-any.whl (14 kB)
Collecting idna
  Downloading idna-3.10-py3-none-any.whl (70 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 70.4/70.4 KB 5.0 MB/s eta 0:00:00
Collecting certifi
  Downloading certifi-2025.6.15-py3-none-any.whl (157 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 157.7/157.7 KB 11.8 MB/s eta 0:00:00
Collecting anyio
  Downloading anyio-4.9.0-py3-none-any.whl (100 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.9/100.9 KB 8.0 MB/s eta 0:00:00
Collecting httpcore==1.*
  Downloading httpcore-1.0.9-py3-none-any.whl (78 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.8/78.8 KB 7.2 MB/s eta 0:00:00
Collecting h11>=0.16
  Downloading h11-0.16.0-py3-none-any.whl (37 kB)
Collecting hf-xet<2.0.0,>=1.1.2
  Downloading hf_xet-1.1.5-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1/3.1 MB 58.4 MB/s eta 0:00:00
Collecting jsonschema-specifications>=2023.03.6
  Downloading jsonschema_specifications-2025.4.1-py3-none-any.whl (18 kB)
Collecting attrs>=22.2.0
  Downloading attrs-25.3.0-py3-none-any.whl (63 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.8/63.8 KB 5.8 MB/s eta 0:00:00
Collecting rpds-py>=0.7.1
  Downloading rpds_py-0.25.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (386 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 387.0/387.0 KB 21.5 MB/s eta 0:00:00
Collecting referencing>=0.28.4
  Downloading referencing-0.36.2-py3-none-any.whl (26 kB)
Collecting python-dateutil>=2.5.3
  Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 229.9/229.9 KB 14.7 MB/s eta 0:00:00
Collecting oauthlib>=3.2.2
  Downloading oauthlib-3.3.1-py3-none-any.whl (160 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 160.1/160.1 KB 13.6 MB/s eta 0:00:00
Collecting durationpy>=0.7
  Downloading durationpy-0.10-py3-none-any.whl (3.9 kB)
Collecting requests-oauthlib
  Downloading requests_oauthlib-2.0.0-py2.py3-none-any.whl (24 kB)
Collecting websocket-client!=0.40.0,!=0.41.*,!=0.42.*,>=0.32.0
  Downloading websocket_client-1.8.0-py3-none-any.whl (58 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.8/58.8 KB 3.9 MB/s eta 0:00:00
Collecting google-auth>=1.0.1
  Downloading google_auth-2.40.3-py2.py3-none-any.whl (216 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 216.1/216.1 KB 14.2 MB/s eta 0:00:00
Collecting urllib3>=1.24.2
  Downloading urllib3-2.5.0-py3-none-any.whl (129 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 129.8/129.8 KB 8.3 MB/s eta 0:00:00
Collecting six>=1.9.0
  Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
Collecting language-data>=1.2
  Downloading language_data-1.3.0-py3-none-any.whl (5.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.4/5.4 MB 64.7 MB/s eta 0:00:00
Collecting flatbuffers
  Downloading flatbuffers-25.2.10-py2.py3-none-any.whl (30 kB)
Collecting protobuf
  Downloading protobuf-6.31.1-cp39-abi3-manylinux2014_x86_64.whl (321 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 321.1/321.1 KB 21.8 MB/s eta 0:00:00
Collecting coloredlogs
  Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46.0/46.0 KB 3.5 MB/s eta 0:00:00
Collecting importlib-metadata<8.8.0,>=6.0
  Downloading importlib_metadata-8.7.0-py3-none-any.whl (27 kB)
Collecting opentelemetry-exporter-otlp-proto-common==1.34.1
  Downloading opentelemetry_exporter_otlp_proto_common-1.34.1-py3-none-any.whl (18 kB)
Collecting opentelemetry-proto==1.34.1
  Downloading opentelemetry_proto-1.34.1-py3-none-any.whl (55 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 55.7/55.7 KB 5.3 MB/s eta 0:00:00
Collecting googleapis-common-protos~=1.52
  Downloading googleapis_common_protos-1.70.0-py3-none-any.whl (294 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 294.5/294.5 KB 21.7 MB/s eta 0:00:00
Collecting protobuf
  Downloading protobuf-5.29.5-cp38-abi3-manylinux2014_x86_64.whl (319 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 319.9/319.9 KB 16.0 MB/s eta 0:00:00
Collecting opentelemetry-semantic-conventions==0.55b1
  Downloading opentelemetry_semantic_conventions-0.55b1-py3-none-any.whl (196 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 196.2/196.2 KB 17.1 MB/s eta 0:00:00
Collecting backoff>=1.10.0
  Downloading backoff-2.2.1-py3-none-any.whl (15 kB)
Collecting distro>=1.5.0
  Downloading distro-1.9.0-py3-none-any.whl (20 kB)
Collecting typing-inspection>=0.4.0
  Downloading typing_inspection-0.4.1-py3-none-any.whl (14 kB)
Collecting pydantic-core==2.33.2
  Downloading pydantic_core-2.33.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 MB 50.5 MB/s eta 0:00:00
Collecting annotated-types>=0.6.0
  Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
Collecting charset_normalizer<4,>=2
  Downloading charset_normalizer-3.4.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (149 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 149.5/149.5 KB 10.5 MB/s eta 0:00:00
Collecting markdown-it-py>=2.2.0
  Downloading markdown_it_py-3.0.0-py3-none-any.whl (87 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 87.5/87.5 KB 8.3 MB/s eta 0:00:00
Collecting pygments<3.0.0,>=2.13.0
  Downloading pygments-2.19.2-py3-none-any.whl (1.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 49.6 MB/s eta 0:00:00
Collecting confection<1.0.0,>=0.0.1
  Downloading confection-0.1.5-py3-none-any.whl (35 kB)
Collecting blis<1.4.0,>=1.3.0
  Downloading blis-1.3.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (11.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.5/11.5 MB 66.7 MB/s eta 0:00:00
Collecting shellingham>=1.3.0
  Downloading shellingham-1.5.4-py2.py3-none-any.whl (9.8 kB)
Collecting watchfiles>=0.13
  Downloading watchfiles-1.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (453 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 453.1/453.1 KB 26.7 MB/s eta 0:00:00
Collecting uvloop>=0.15.1
  Downloading uvloop-0.21.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.8/3.8 MB 60.4 MB/s eta 0:00:00
Collecting httptools>=0.6.3
  Downloading httptools-0.6.4-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (442 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 442.1/442.1 KB 28.0 MB/s eta 0:00:00
Collecting websockets>=10.4
  Downloading websockets-15.0.1-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (181 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 181.6/181.6 KB 16.0 MB/s eta 0:00:00
Collecting smart-open<8.0.0,>=5.2.1
  Downloading smart_open-7.1.0-py3-none-any.whl (61 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.7/61.7 KB 5.0 MB/s eta 0:00:00
Collecting cloudpathlib<1.0.0,>=0.7.0
  Downloading cloudpathlib-0.21.1-py3-none-any.whl (52 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 52.8/52.8 KB 4.8 MB/s eta 0:00:00
Collecting rsa<5,>=3.1.4
  Downloading rsa-4.9.1-py3-none-any.whl (34 kB)
Collecting pyasn1-modules>=0.2.1
  Downloading pyasn1_modules-0.4.2-py3-none-any.whl (181 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 181.3/181.3 KB 16.3 MB/s eta 0:00:00
Collecting cachetools<6.0,>=2.0.0
  Downloading cachetools-5.5.2-py3-none-any.whl (10 kB)
Collecting zipp>=3.20
  Downloading zipp-3.23.0-py3-none-any.whl (10 kB)
Collecting marisa-trie>=1.1.0
  Downloading marisa_trie-1.2.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 48.9 MB/s eta 0:00:00
Collecting mdurl~=0.1
  Downloading mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Collecting wrapt
  Downloading wrapt-1.17.2-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (82 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 82.8/82.8 KB 6.5 MB/s eta 0:00:00
Collecting sniffio>=1.1
  Downloading sniffio-1.3.1-py3-none-any.whl (10 kB)
Collecting exceptiongroup>=1.0.2
  Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)
Collecting humanfriendly>=9.1
  Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 86.8/86.8 KB 8.1 MB/s eta 0:00:00
Collecting pyasn1<0.7.0,>=0.6.1
  Downloading pyasn1-0.6.1-py3-none-any.whl (83 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 83.1/83.1 KB 7.9 MB/s eta 0:00:00
Collecting azure-core>=1.30.0
  Downloading azure_core-1.34.0-py3-none-any.whl (207 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 207.4/207.4 KB 6.0 MB/s eta 0:00:00
Collecting azure-ai-inference
  Downloading azure_ai_inference-1.0.0b9-py3-none-any.whl (124 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 124.9/124.9 KB 2.4 MB/s eta 0:00:00
Collecting isodate>=0.6.1
  Downloading isodate-0.7.2-py3-none-any.whl (22 kB)
Building wheels for collected packages: pypika
  Building wheel for pypika (pyproject.toml) ... done
  Created wheel for pypika: filename=pypika-0.48.9-py2.py3-none-any.whl size=53803 sha256=7104ff2a477ef783a1e9f63a4883d35de34bf7bb9131122bb5a83fa279dbd64e
  Stored in directory: /home/azureuser/.cache/pip/wheels/e1/26/51/d0bffb3d2fd82256676d7ad3003faea3bd6dddc9577af665f4
Successfully built pypika
Installing collected packages: triton, pysqlite3-binary, pypika, nvidia-cusparselt-cu12, mpmath, flatbuffers, durationpy, cymem, zipp, wrapt, websockets, websocket-client, wasabi, uvloop, urllib3, typing_extensions, tqdm, tomli, threadpoolctl, tenacity, sympy, spacy-loggers, spacy-legacy, sniffio, six, shellingham, safetensors, rpds-py, regex, pyyaml, python-dotenv, pyproject_hooks, pygments, pybase64, pyasn1, psutil, protobuf, Pillow, packaging, overrides, orjson, oauthlib, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, networkx, murmurhash, mmh3, mdurl, markupsafe, marisa-trie, joblib, itsdangerous, importlib-resources, idna, humanfriendly, httptools, hf-xet, h11, grpcio, fsspec, filelock, distro, click, charset_normalizer, certifi, catalogue, cachetools, blinker, bcrypt, backoff, attrs, annotated-types, werkzeug, uvicorn, typing-inspection, srsly, smart-open, scipy, rsa, requests, referencing, python-dateutil, pydantic-core, pyasn1-modules, preshed, opentelemetry-proto, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nltk, markdown-it-py, language-data, jinja2, importlib-metadata, httpcore, gunicorn, googleapis-common-protos, exceptiongroup, coloredlogs, cloudpathlib, build, blis, textblob, scikit-learn, rich, requests-oauthlib, pydantic, posthog, opentelemetry-exporter-otlp-proto-common, opentelemetry-api, onnxruntime, nvidia-cusolver-cu12, langcodes, jsonschema-specifications, huggingface-hub, google-auth, flask, anyio, watchfiles, typer, torch, tokenizers, opentelemetry-semantic-conventions, kubernetes, jsonschema, httpx, confection, weasel, transformers, thinc, opentelemetry-sdk, accelerate, spacy, sentence-transformers, opentelemetry-exporter-otlp-proto-grpc, chromadb
Successfully installed Pillow-11.2.1 accelerate-1.8.1 annotated-types-0.7.0 anyio-4.9.0 attrs-25.3.0 backoff-2.2.1 bcrypt-4.3.0 blinker-1.9.0 blis-1.3.0 build-1.2.2.post1 cachetools-5.5.2 catalogue-2.0.10 certifi-2025.6.15 charset_normalizer-3.4.2 chromadb-1.0.13 click-8.2.1 cloudpathlib-0.21.1 coloredlogs-15.0.1 confection-0.1.5 cymem-2.0.11 distro-1.9.0 durationpy-0.10 exceptiongroup-1.3.0 filelock-3.18.0 flask-3.1.1 flatbuffers-25.2.10 fsspec-2025.5.1 google-auth-2.40.3 googleapis-common-protos-1.70.0 grpcio-1.73.0 gunicorn-23.0.0 h11-0.16.0 hf-xet-1.1.5 httpcore-1.0.9 httptools-0.6.4 httpx-0.28.1 huggingface-hub-0.33.0 humanfriendly-10.0 idna-3.10 importlib-metadata-8.7.0 importlib-resources-6.5.2 itsdangerous-2.2.0 jinja2-3.1.6 joblib-1.5.1 jsonschema-4.24.0 jsonschema-specifications-2025.4.1 kubernetes-33.1.0 langcodes-3.5.0 language-data-1.3.0 marisa-trie-1.2.1 markdown-it-py-3.0.0 markupsafe-3.0.2 mdurl-0.1.2 mmh3-5.1.0 mpmath-1.3.0 murmurhash-1.0.13 networkx-3.4.2 nltk-3.9.1 numpy-2.2.6 nvidia-cublas-cu12-12.4.5.8 nvidia-cuda-cupti-cu12-12.4.127 nvidia-cuda-nvrtc-cu12-12.4.127 nvidia-cuda-runtime-cu12-12.4.127 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.2.1.3 nvidia-curand-cu12-10.3.5.147 nvidia-cusolver-cu12-11.6.1.9 nvidia-cusparse-cu12-12.3.1.170 nvidia-cusparselt-cu12-0.6.2 nvidia-nccl-cu12-2.21.5 nvidia-nvjitlink-cu12-12.4.127 nvidia-nvtx-cu12-12.4.127 oauthlib-3.3.1 onnxruntime-1.22.0 opentelemetry-api-1.34.1 opentelemetry-exporter-otlp-proto-common-1.34.1 opentelemetry-exporter-otlp-proto-grpc-1.34.1 opentelemetry-proto-1.34.1 opentelemetry-sdk-1.34.1 opentelemetry-semantic-conventions-0.55b1 orjson-3.10.18 overrides-7.7.0 packaging-25.0 posthog-5.4.0 preshed-3.0.10 protobuf-5.29.5 psutil-7.0.0 pyasn1-0.6.1 pyasn1-modules-0.4.2 pybase64-1.4.1 pydantic-2.11.7 pydantic-core-2.33.2 pygments-2.19.2 pypika-0.48.9 pyproject_hooks-1.2.0 pysqlite3-binary-0.5.4 python-dateutil-2.9.0.post0 python-dotenv-1.1.0 pyyaml-6.0.2 referencing-0.36.2 regex-2024.11.6 requests-2.32.4 requests-oauthlib-2.0.0 rich-14.0.0 rpds-py-0.25.1 rsa-4.9.1 safetensors-0.5.3 scikit-learn-1.7.0 scipy-1.15.3 sentence-transformers-4.1.0 shellingham-1.5.4 six-1.17.0 smart-open-7.1.0 sniffio-1.3.1 spacy-3.8.7 spacy-legacy-3.0.12 spacy-loggers-1.0.5 srsly-2.5.1 sympy-1.13.1 tenacity-9.1.2 textblob-0.19.0 thinc-8.3.6 threadpoolctl-3.6.0 tokenizers-0.21.1 tomli-2.2.1 torch-2.6.0 tqdm-4.67.1 transformers-4.52.4 triton-3.2.0 typer-0.16.0 typing-inspection-0.4.1 typing_extensions-4.14.0 urllib3-2.5.0 uvicorn-0.34.3 uvloop-0.21.0 wasabi-1.1.3 watchfiles-1.1.0 weasel-0.4.1 websocket-client-1.8.0 websockets-15.0.1 werkzeug-3.1.3 wrapt-1.17.2 zipp-3.23.0 jiter-0.10.0 openai-1.91.0 tiktoken-0.9.0
```
### (venv) azureuser@flaskAI:~/persona-rag$ python -m spacy download en_core_web_sm
```
Collecting en-core-web-sm==3.8.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl (12.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.8/12.8 MB 62.2 MB/s eta 0:00:00
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-3.8.0
✔ Download and installation successful
You can now load the package via spacy.load('en_core_web_sm')
```
### (venv) azureuser@flaskAI:~/persona-rag$ python -m data_ingestion.ingest
```
System SQLite version detected: 3.37.2
System SQLite version (3.37.2) is sufficient. No pysqlite3 patch needed based on version check.
PYSQLITE3_PATCH_ENABLED is false and FORCE_PYSQLITE3_PATCH not set. Using default sqlite3 module.
Performing HARD rollback: Deleting database file at database/persona.db...
No database file found at database/persona.db.
Attempting to delete existing database file at database/persona.db for recreation.
No existing database file found at database/persona.db for recreation.
Database initialized or verified at database/persona.db

--- Starting Persona Data Ingestion from data_ingestion/persona ---
Skipping persona file: barkha.json (not in filter list)
Skipping persona file: hitchens.json (not in filter list)
Ingesting persona data for journalist 'Casey Newton' (DB ID: 1)...
Successfully ingested persona data for 'Casey Newton'.
Ingesting persona data for journalist 'Morgan Housel' (DB ID: 2)...
Successfully ingested persona data for 'Morgan Housel'.
Ingesting persona data for journalist 'Dave Gruber' (DB ID: 3)...
Successfully ingested persona data for 'Dave Gruber'.

--- Starting Corpus Data Ingestion from data_ingestion/corpus ---
Corpus path : data_ingestion/corpus/casey_newton
Ingesting corpus for journalist: Casey Newton (ID: 1) from data_ingestion/corpus/casey_newton
Corpus path : data_ingestion/corpus/morgan_housel
Ingesting corpus for journalist: Morgan Housel (ID: 2) from data_ingestion/corpus/morgan_housel
Corpus path : data_ingestion/corpus/dave_gruber
Ingesting corpus for journalist: Dave Gruber (ID: 3) from data_ingestion/corpus/dave_gruber

--- Ingestion complete. ---
```
### (venv) azureuser@flaskAI:~/persona-rag$ python -m data_ingestion.embed_corpus
```
System SQLite version detected: 3.37.2
System SQLite version (3.37.2) is sufficient. No pysqlite3 patch needed based on version check.
PYSQLITE3_PATCH_ENABLED is false and FORCE_PYSQLITE3_PATCH not set. Using default sqlite3 module.
modules.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 349/349 [00:00<00:00, 1.54MB/s]
config_sentence_transformers.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 116/116 [00:00<00:00, 755kB/s]
README.md: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10.5k/10.5k [00:00<00:00, 45.7MB/s]
sentence_bert_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 53.0/53.0 [00:00<00:00, 354kB/s]
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 612/612 [00:00<00:00, 4.86MB/s]
model.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 90.9M/90.9M [00:00<00:00, 127MB/s]
tokenizer_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 350/350 [00:00<00:00, 1.80MB/s]
vocab.txt: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 39.5MB/s]
tokenizer.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 466k/466k [00:00<00:00, 31.5MB/s]
special_tokens_map.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 112/112 [00:00<00:00, 583kB/s]
config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 190/190 [00:00<00:00, 994kB/s]
Embedded 3 documents into ChromaDB.
```
### (venv) azureuser@flaskAI:~/persona-rag$ python -m scripts.fine_tune_all
```
System SQLite version detected: 3.37.2
System SQLite version (3.37.2) is sufficient. No pysqlite3 patch needed based on version check.
PYSQLITE3_PATCH_ENABLED is false and FORCE_PYSQLITE3_PATCH not set. Using default sqlite3 module.
Fine-tuning model for journalist: Casey Newton
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 762/762 [00:00<00:00, 4.11MB/s]
model.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 353M/353M [00:01<00:00, 205MB/s]
generation_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 124/124 [00:00<00:00, 756kB/s]
tokenizer_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 26.0/26.0 [00:00<00:00, 142kB/s]
vocab.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.04M/1.04M [00:00<00:00, 21.9MB/s]
merges.txt: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 456k/456k [00:00<00:00, 42.4MB/s]
tokenizer.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1.36M/1.36M [00:00<00:00, 4.70MB/s]
/home/azureuser/persona-rag/venv/lib/python3.10/site-packages/transformers/data/datasets/language_modeling.py:53: FutureWarning: This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets library. You can have a look at this example script for pointers: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
  warnings.warn(
  0%|                                                                                                                                    | 0/214 [00:00<?, ?it/s]`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
{'loss': 4.2292, 'grad_norm': 9.895827293395996, 'learning_rate': 4.7897196261682245e-05, 'epoch': 0.09}                                                         
{'loss': 4.1703, 'grad_norm': 9.290998458862305, 'learning_rate': 4.556074766355141e-05, 'epoch': 0.19}                                                          
{'loss': 3.9432, 'grad_norm': 9.615720748901367, 'learning_rate': 4.3224299065420565e-05, 'epoch': 0.28}                                                         
{'loss': 3.9674, 'grad_norm': 8.59230899810791, 'learning_rate': 4.088785046728972e-05, 'epoch': 0.37}                                                           
{'loss': 3.8275, 'grad_norm': 10.277146339416504, 'learning_rate': 3.855140186915888e-05, 'epoch': 0.47}                                                         
{'loss': 3.9345, 'grad_norm': 9.236841201782227, 'learning_rate': 3.621495327102804e-05, 'epoch': 0.56}                                                          
{'loss': 3.7179, 'grad_norm': 9.363876342773438, 'learning_rate': 3.38785046728972e-05, 'epoch': 0.65}                                                           
{'loss': 3.9459, 'grad_norm': 9.951996803283691, 'learning_rate': 3.1542056074766355e-05, 'epoch': 0.75}                                                         
{'loss': 3.7073, 'grad_norm': 9.802257537841797, 'learning_rate': 2.9205607476635515e-05, 'epoch': 0.84}                                                         
{'loss': 3.8204, 'grad_norm': 9.671603202819824, 'learning_rate': 2.6869158878504675e-05, 'epoch': 0.93}                                                         
{'loss': 3.9197, 'grad_norm': 10.507174491882324, 'learning_rate': 2.4532710280373832e-05, 'epoch': 1.03}                                                        
{'loss': 3.5243, 'grad_norm': 8.888338088989258, 'learning_rate': 2.2196261682242992e-05, 'epoch': 1.12}                                                         
{'loss': 3.4834, 'grad_norm': 10.185053825378418, 'learning_rate': 1.985981308411215e-05, 'epoch': 1.21}                                                         
{'loss': 3.4975, 'grad_norm': 10.255139350891113, 'learning_rate': 1.752336448598131e-05, 'epoch': 1.31}                                                         
{'loss': 3.2326, 'grad_norm': 10.247980117797852, 'learning_rate': 1.5186915887850467e-05, 'epoch': 1.4}                                                         
{'loss': 3.4721, 'grad_norm': 10.168941497802734, 'learning_rate': 1.2850467289719625e-05, 'epoch': 1.5}                                                         
{'loss': 3.4051, 'grad_norm': 11.437166213989258, 'learning_rate': 1.0514018691588785e-05, 'epoch': 1.59}                                                        
{'loss': 3.4432, 'grad_norm': 10.301460266113281, 'learning_rate': 8.177570093457943e-06, 'epoch': 1.68}                                                         
{'loss': 3.3538, 'grad_norm': 10.628300666809082, 'learning_rate': 5.841121495327103e-06, 'epoch': 1.78}                                                         
{'loss': 3.3826, 'grad_norm': 10.425619125366211, 'learning_rate': 3.5046728971962617e-06, 'epoch': 1.87}                                                        
{'loss': 3.3085, 'grad_norm': 11.748620986938477, 'learning_rate': 1.1682242990654206e-06, 'epoch': 1.96}                                                        
{'train_runtime': 415.4623, 'train_samples_per_second': 1.025, 'train_steps_per_second': 0.515, 'train_loss': 3.6717984475822094, 'epoch': 2.0}                  
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 214/214 [06:55<00:00,  1.94s/it]
Fine-tuned model saved at database/models/casey newton
Fine-tuning model for journalist: Dave Gruber
/home/azureuser/persona-rag/venv/lib/python3.10/site-packages/transformers/data/datasets/language_modeling.py:53: FutureWarning: This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets library. You can have a look at this example script for pointers: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
  warnings.warn(
{'loss': 4.0619, 'grad_norm': 10.219564437866211, 'learning_rate': 4.423076923076923e-05, 'epoch': 0.26}                                                         
{'loss': 4.0037, 'grad_norm': 9.320098876953125, 'learning_rate': 3.782051282051282e-05, 'epoch': 0.51}                                                          
{'loss': 3.9022, 'grad_norm': 9.896652221679688, 'learning_rate': 3.141025641025641e-05, 'epoch': 0.77}                                                          
{'loss': 3.8501, 'grad_norm': 9.39674186706543, 'learning_rate': 2.5e-05, 'epoch': 1.03}                                                                         
{'loss': 3.6616, 'grad_norm': 9.255657196044922, 'learning_rate': 1.858974358974359e-05, 'epoch': 1.28}                                                          
{'loss': 3.4739, 'grad_norm': 10.479140281677246, 'learning_rate': 1.217948717948718e-05, 'epoch': 1.54}                                                         
{'loss': 3.4689, 'grad_norm': 8.651473999023438, 'learning_rate': 5.76923076923077e-06, 'epoch': 1.79}                                                           
{'train_runtime': 152.3673, 'train_samples_per_second': 1.011, 'train_steps_per_second': 0.512, 'train_loss': 3.740824479323167, 'epoch': 2.0}                   
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 78/78 [02:32<00:00,  1.95s/it]
Fine-tuned model saved at database/models/dave gruber
Fine-tuning model for journalist: Morgan Housel
/home/azureuser/persona-rag/venv/lib/python3.10/site-packages/transformers/data/datasets/language_modeling.py:53: FutureWarning: This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets library. You can have a look at this example script for pointers: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
  warnings.warn(
{'loss': 3.4809, 'grad_norm': 10.26502513885498, 'learning_rate': 4.795454545454546e-05, 'epoch': 0.09}                                                          
{'loss': 3.4451, 'grad_norm': 10.28603458404541, 'learning_rate': 4.5681818181818186e-05, 'epoch': 0.18}                                                         
{'loss': 3.4269, 'grad_norm': 9.045283317565918, 'learning_rate': 4.340909090909091e-05, 'epoch': 0.27}                                                          
{'loss': 3.4615, 'grad_norm': 9.326863288879395, 'learning_rate': 4.113636363636364e-05, 'epoch': 0.36}                                                          
{'loss': 3.5186, 'grad_norm': 9.48390007019043, 'learning_rate': 3.8863636363636364e-05, 'epoch': 0.45}                                                          
{'loss': 3.3624, 'grad_norm': 9.929384231567383, 'learning_rate': 3.659090909090909e-05, 'epoch': 0.55}                                                          
{'loss': 3.189, 'grad_norm': 9.495574951171875, 'learning_rate': 3.431818181818182e-05, 'epoch': 0.64}                                                           
{'loss': 3.3102, 'grad_norm': 9.354437828063965, 'learning_rate': 3.204545454545455e-05, 'epoch': 0.73}                                                          
{'loss': 3.3477, 'grad_norm': 9.271230697631836, 'learning_rate': 2.9772727272727273e-05, 'epoch': 0.82}                                                         
{'loss': 3.2519, 'grad_norm': 8.940250396728516, 'learning_rate': 2.7500000000000004e-05, 'epoch': 0.91}                                                         
{'loss': 3.0535, 'grad_norm': 11.969822883605957, 'learning_rate': 2.5227272727272726e-05, 'epoch': 1.0}                                                         
{'loss': 2.842, 'grad_norm': 8.811469078063965, 'learning_rate': 2.2954545454545457e-05, 'epoch': 1.09}                                                          
{'loss': 3.0257, 'grad_norm': 9.933993339538574, 'learning_rate': 2.0681818181818182e-05, 'epoch': 1.18}                                                         
{'loss': 3.043, 'grad_norm': 9.227004051208496, 'learning_rate': 1.840909090909091e-05, 'epoch': 1.27}                                                           
{'loss': 2.9458, 'grad_norm': 10.86384105682373, 'learning_rate': 1.6136363636363638e-05, 'epoch': 1.36}                                                         
{'loss': 3.0089, 'grad_norm': 9.673541069030762, 'learning_rate': 1.3863636363636364e-05, 'epoch': 1.45}                                                         
{'loss': 2.9099, 'grad_norm': 9.724315643310547, 'learning_rate': 1.159090909090909e-05, 'epoch': 1.55}                                                          
{'loss': 2.8821, 'grad_norm': 10.338022232055664, 'learning_rate': 9.318181818181819e-06, 'epoch': 1.64}                                                         
{'loss': 2.8234, 'grad_norm': 10.062150955200195, 'learning_rate': 7.045454545454545e-06, 'epoch': 1.73}                                                         
{'loss': 3.0564, 'grad_norm': 9.879551887512207, 'learning_rate': 4.772727272727273e-06, 'epoch': 1.82}                                                          
{'loss': 2.9774, 'grad_norm': 9.989140510559082, 'learning_rate': 2.5e-06, 'epoch': 1.91}                                                                        
{'loss': 3.0651, 'grad_norm': 14.6470365524292, 'learning_rate': 2.2727272727272726e-07, 'epoch': 2.0}                                                           
{'train_runtime': 429.0669, 'train_samples_per_second': 1.021, 'train_steps_per_second': 0.513, 'train_loss': 3.155794230374423, 'epoch': 2.0}                   
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 220/220 [07:09<00:00,  1.95s/it]
```