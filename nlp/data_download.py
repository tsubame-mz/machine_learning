import os
import urllib.request
import tarfile
import zipfile

data_dir = "data"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

# Word2Vecの日本語学習済みモデル
url = "http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/data/20170201.tar.bz2"
save_path = "./data/20170201.tar.bz2"
if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)
if not os.path.exists("./data/entity_vector"):
    tar = tarfile.open("./data/20170201.tar.bz2", "r|bz2")
    tar.extractall("./data/")
    tar.close()

# fastTextの英語学習済みモデル
url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
save_path = "./data/wiki-news-300d-1M.vec.zip"
if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)
if not os.path.exists("./data/wiki-news-300d-1M.vec"):
    zip = zipfile.ZipFile("./data/wiki-news-300d-1M.vec.zip")
    zip.extractall("./data/")
    zip.close()

# IMDbデータセット
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
save_path = "./data/aclImdb_v1.tar.gz"
if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)
if not os.path.exists("./data/aclImdb"):
    tar = tarfile.open("./data/aclImdb_v1.tar.gz")
    tar.extractall("./data")
    tar.close()

# fastTextの学習済みモデル
if not os.path.exists("./data/vector_neologd"):
    zip = zipfile.ZipFile("./data/vector_neologd.zip")
    zip.extractall("./data/vector_neologd")
    zip.close()

