# 安装需要的库
# pip install wordcloud numpy matplotlib

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# 读取文本数据
with open('autodl-tmp/EMGCN-v2/EMCGCN-v1.1/data/D2/lap14/train.txt', 'r', encoding='utf-8') as file:
    text_data = file.read()

# 数据清洗
translator = str.maketrans("", "", string.punctuation)
text_data = text_data.translate(translator)
stop_words = set(stopwords.words('english'))  # 使用英文停用词，中文可替换为中文停用词表
words = word_tokenize(text_data.lower())  # 分词并转为小写
filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

# 生成词云图
wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(' '.join(filtered_words))

# 绘制词云图
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
