import array
import csv
import gzip
import logging
import re
import shutil
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from tqdm import tqdm

import click
import sentencepiece as spm
import torch

logging.basicConfig(level=logging.INFO)


MAINDIR = Path("./")
DATA_PATH = MAINDIR.joinpath("data")
SRC_PATH = DATA_PATH.joinpath("training.1600000.processed.noemoticon.csv")
RE_URL = re.compile(r"(?:\@|https?\://)\S+")
RE_MENTION = re.compile(r"(?:@)\S+")
RE_NOT  = re.compile('[^\w\s@:,;]+')

# 定义数据读取器
def datareader(path: Path):
    with open(path, "rt", encoding="utf-8", errors='ignore') as fp:
        for row in csv.reader(fp):
            """
            row[5]是数据集中的推文列
            将RE_URL替换为空字符,RE_MENTION替换为@,RE_NOT替换为空格
            即删除推文中的URL和不需要的符号,保留@ 字母和数字
            将处理后的字符串和row[0]一起输出
            """
            yield RE_NOT.sub(' ', RE_MENTION.sub('@', RE_URL.sub('',row[5]))), row[0]

# 清理数据
def cleanup(src, target):
    """Nettoyage du jeu de tweet"""
    if not target.is_file():
        logging.info("Creating the text data file from %s", src)
        target_tmp = target.with_suffix(".tmp")  # 创建临时文件
        with target_tmp.open("wt", encoding="utf-8") as out:
            for tweet, klass in datareader(src):
                out.write(tweet)
                out.write("\n")

        shutil.move(target_tmp, target)  # 将临时文件移动到目标文件中

# 定义一个命名元组，包含 text（推文文本）和 labels（标签）
Batch = namedtuple("Batch", ["text", "labels"])

# 定义TextDataset 类，包含推文文本、文本长度、情感标签
class TextDataset(torch.utils.data.Dataset):

    def __init__(self, text: torch.LongTensor, sizes: torch.LongTensor, labels: torch.LongTensor):
        self.text = text
        self.sizes = sizes
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.text[self.sizes[index]:self.sizes[index+1]], self.labels[index].item()

    # 用于批量加载数据，使用 pad_sequence 填充文本，使其对齐成相同长度，方便进行训练。
    @staticmethod
    def collate(batch):
        data = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return Batch(torch.nn.utils.rnn.pad_sequence(data, batch_first=True), torch.LongTensor(labels))

# 数据集处理函数
def process(mode: str, fn, map: dict):
    """
    Process the dataset 首先判断缓存文件是否存在，如果存在则加载；否则对原始数据进行编码、存储
    """
    datapath = MAINDIR / f"{mode}.pth"
    if datapath.is_file():
        logging.info("Loading %s", mode)
        with gzip.open(datapath, "rb") as fp:
            return torch.load(fp)

    text = array.array('L')  # 创建数组对象，元素类型码为L，即长整型
    sizes = array.array('L')
    labels = array.array('B')  # 创建数组对象，元素类型码为B，即字符
    sizes.append(0)
    for tweet, label in tqdm(datareader(fn), unit=" sentences"):
        for tokenid in tokenizer.encode_as_ids(tweet):
            # 每条推文通过 tokenizer 编码为 ID，然后保存这些 ID 和对应的标签
            text.append(tokenid)
        sizes.append(len(text))
        labels.append(int(label))

    data = TextDataset(torch.LongTensor(text), torch.LongTensor(sizes), torch.LongTensor(labels))
    with gzip.open(datapath, "wb") as fp:
        torch.save(data, fp)
    return data

# 类似于 process 函数，但额外接受词汇表大小参数，并根据指定映射 (map) 处理标签数据
def generatedata(mode: str, tokenizer, vocab_size: int, fn, map):
    datapath = MAINDIR / f"{mode}-{vocab_size}.pth"
    if datapath.is_file():
        return

    text = array.array('L')
    sizes = array.array('L')
    labels = array.array('B')
    sizes.append(0)
    for tweet, label in tqdm(datareader(fn), unit=" sentences"):
        for tokenid in tokenizer.encode_as_ids(tweet):
            text.append(tokenid)
        label = int(label)
        if label in map:
            sizes.append(len(text))
            labels.append(map[label])

    data = TextDataset(torch.LongTensor(text), torch.LongTensor(sizes), torch.LongTensor(labels))
    with gzip.open(datapath, "wb") as fp:
        torch.save(data, fp)


@click.option("--vocab-size", default=1000, type=int, help="Vocabulary size")
@click.command()
def cli(vocab_size: int):
    """
    通过命令行获取参数，如 vocab_size(词汇表大小)
    """
    # Création du jeu de données et du modèle
    TRAINPATH = DATA_PATH.joinpath("sentiment140-train.txt") # 定义训练数据路径


    # Création du vocabulaire
    wpmodel = Path("wp{}.model".format(vocab_size))  # 将vocab_size变量的值插入到{}位置，生成词汇表模型的文件路径wp1000.model
    if not wpmodel.is_file():
        logging.info("Did not find the wordpiece model %s", wpmodel)
        cleanup(SRC_PATH, TRAINPATH)   # 调用cleanup函数，删除临时文件
        logging.info("Création du vocabulaire avec sentencepiece")
        spm.SentencePieceTrainer.train(
            input=str(TRAINPATH),
            model_prefix=f"wp{vocab_size}",
            vocab_size=vocab_size
        )
        TRAINPATH.unlink()


    # Création des jeux de données
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(f"wp{vocab_size}.model")  # 读取并加载wp1000.model模型，用于后续的文本处理

    CLASSMAP = { 0: 0, 4: 1 }
    # 调用generatedata生成训练数据集
    generatedata("train", tokenizer, vocab_size, SRC_PATH, CLASSMAP)


if __name__ == "__main__":
    cli()


"""
训练好的 SentencePiece 模型会被保存为 .model 和 .vocab 文件(vocab中每行的第一个部分表示一个token;每行第二个部分表示该token的对数概率log-probability)
分割后的数据会被保存为 .pth 文件,推文文本编码为ID后的数据集,每条推文由一个整数列表表示,每个整数代表一个token的ID
"""