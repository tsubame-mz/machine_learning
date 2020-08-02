import re
import string
import random
import torchtext
from torchtext.vocab import Vectors

def get_IMDb_DataLoaders_and_TEXT(max_length=256, batch_size=32):
	# 前処理
	def preprocessing_text(text):
	    text = re.sub("<br />", "", text)  # 改行コードの削除

	    for p in string.punctuation:
	        if (p == ".") or (p == ","):    # カンマとピリオドはそのまま
	            continue
	        else:
	            text = text.replace(p, " ")

	    # カンマとピリオドの前後に半角スペースをいれる
	    text = text.replace(".", " . ")
	    text = text.replace(",", " , ")
	    return text

	# 単語分割
	def tokenizer_janome(text):
	    # 英語なので半角スペースで分割するのみ
	    return text.strip().split()

	# 前処理＋単語分割
	def tokenizer_with_preprocessing(text):
	    return tokenizer_janome(preprocessing_text(text))

	# Datasetの作成
	TEXT = torchtext.data.Field(
	    sequential=True,    # 可変データか
	    tokenize=tokenizer_with_preprocessing,  # 単語分割の関数
	    use_vocab=True,     # 単語を辞書に追加するか
	    lower=True,         # アルファベットを小文字にするか
	    include_lengths=True,   # 単語数を保持するか
	    batch_first=True,       # バッチの次元が先にくるか
	    fix_length=max_length,  # 各文章をパディングして同じ長さにする
	    init_token="<cls>",     # 文章の最初の単語
	    eos_token="<eos>"       # 文章の最後の単語
	)
	LABEL = torchtext.data.Field(
	    sequential=False,    # 可変データか
	    use_vocab=False,     # 単語を辞書に追加するか
	)
	train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
	    path="/content/drive/My Drive/Transformer/data/",
	    train="IMDb_train.tsv",
	    test="IMDb_test.tsv",
	    format="tsv",
	    fields=[("Text", TEXT), ("Label", LABEL)]
	)
	# 学習データと検証データを分割する
	train_ds, val_ds = train_val_ds.split(split_ratio=0.8, random_state=random.seed(1234))

	# 単語ベクトルを生成
	english_fasttext_vectors = Vectors("/content/drive/My Drive/Transformer/data/wiki-news-300d-1M.vec")
	TEXT.build_vocab(train_ds, vectors=english_fasttext_vectors, min_freq=1)

	# DataLoaderの作成
	train_dl = torchtext.data.Iterator(train_ds, batch_size=batch_size, train=True)
	val_dl = torchtext.data.Iterator(val_ds, batch_size=batch_size, train=False, sort=False)
	test_dl = torchtext.data.Iterator(test_ds, batch_size=batch_size, train=False, sort=False)

	return train_dl, val_dl, test_dl, TEXT





