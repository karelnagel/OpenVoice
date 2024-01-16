from setuptools import setup, find_packages

setup(
    name="openvoice",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "librosa==0.9.1",
        "faster-whisper==0.9.0",
        "pydub==0.25.1",
        "wavmark",
        "numpy",
        "eng_to_ipa==0.0.2",
        "inflect==7.0.0",
        "unidecode==1.3.7",
        "whisper-timestamped==1.14.2",
        "openai",
        "python-dotenv",
        "pypinyin==0.50.0",
        "cn2an==0.5.22",
        "jieba==0.42.1",
        "gradio==3.48.0",
        "langid==1.1.6",
    ],
)
