# 安装需要的库
# pip install wordcloud numpy matplotlib
import os.path

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

TRAIN_TEXT_PATH = 'autodl-tmp/EMGCN-v2/EMCGCN-v1.1/data/D2/lap14/train.txt'


def main():
    if not os.path.exists(TRAIN_TEXT_PATH):
        print(f"Error: Train text ({TRAIN_TEXT_PATH}) not found.")
        return

    # 读取文本数据
    with open(TRAIN_TEXT_PATH, 'r', encoding='utf-8') as file:
        text_data = file.read()

    # 数据清洗
    translator = str.maketrans("", "", string.punctuation)
    text_data = text_data.translate(translator)

    # 使用英文停用词，中文可替换为中文停用词表
    stop_words = set(stopwords.words('english'))

    # 分词并转为小写
    words = word_tokenize(text_data.lower())
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

    # 生成词云图
    wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(
        ' '.join(filtered_words))

    # 绘制词云图
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
