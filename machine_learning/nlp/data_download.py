import os
import urllib.request
import tarfile
import zipfile


def report_hook(block_num, block_size, total_size):
    if total_size <= 0:
        return

    part_size = block_size * block_num
    if part_size > total_size:
        part_size = total_size
    progress = (part_size / total_size) * 100.0
    print("\r{:.0f}%({}/{})".format(progress, part_size, total_size), end="")


data_dir = "data"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

# Word2Vecの日本語学習済みモデル
url = "http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/data/20170201.tar.bz2"
save_path = "./data/20170201.tar.bz2"
extract_path = "./data/entity_vector"
if not os.path.exists(save_path):
    print("Download: {0}".format(url))
    urllib.request.urlretrieve(url, save_path, report_hook)
    print()
if not os.path.exists(extract_path):
    print("Extract: {0}".format(save_path))
    tar = tarfile.open(save_path, "r|bz2")
    tar.extractall("./data")
    tar.close()

# fastTextの英語学習済みモデル
url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
save_path = "./data/wiki-news-300d-1M.vec.zip"
extract_path = "./data/wiki-news-300d-1M"
if not os.path.exists(save_path):
    print("Download: {0}".format(url))
    urllib.request.urlretrieve(url, save_path, report_hook)
    print()
if not os.path.exists(extract_path):
    print("Extract: {0}".format(save_path))
    zip = zipfile.ZipFile(save_path)
    zip.extractall(extract_path)
    zip.close()

# IMDbデータセット
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
save_path = "./data/aclImdb_v1.tar.gz"
extract_path = "./data/aclImdb"
if not os.path.exists(save_path):
    print("Download: {0}".format(url))
    urllib.request.urlretrieve(url, save_path, report_hook)
    print()
if not os.path.exists(extract_path):
    print("Extract: {0}".format(save_path))
    tar = tarfile.open(save_path)
    tar.extractall("./data")
    tar.close()

# fastTextの日本語学習済みモデル
# https://drive.google.com/open?id=0ByFQ96A4DgSPUm9wVWRLdm5qbmc
save_path = "./data/vector_neologd.zip"
extract_path = "./data/vector_neologd"
if not os.path.exists(extract_path):
    print("Extract: {0}".format(save_path))
    zip = zipfile.ZipFile(save_path)
    zip.extractall(extract_path)
    zip.close()
