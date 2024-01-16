from setuptools import setup, find_packages

setup(
    name="openvoice",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "librosa",
        "faster-whisper",
        "pydub",
        "wavmark",
        "numpy",
        "eng_to_ipa",
        "inflect",
        "unidecode",
        "whisper-timestamped",
        "openai",
        "python-dotenv",
        "pypinyin",
        "cn2an",
        "jieba",
        "gradio",
        "langid",
    ],
)
