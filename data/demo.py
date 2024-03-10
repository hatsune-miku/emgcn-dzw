from senticnet.senticnet import SenticNet
import torch
import numpy as np
from transformers import BertTokenizer


def get_sentic(args, sentences, token_ids, sn=SenticNet()):
    """
    返回情感矩阵
    Return: [batch_size, max_seq_len, max_seq_len]: Tensor
    """

    tokenizer = BertTokenizer("bert-base-uncased")
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    results = torch.zeros((16, 102, 102), device=args.device)
    for batch_id, sentence in enumerate(sentences):
        tokenized_tokens = tokens[batch_id]
        word_list = sentence.split()
        # 确保长度不能超过 max_seq_len，且考虑 [CLS] [SEP] 标签
        length = min(len(word_list), args.max_sequence_len - 2)
        matrix = np.zeros((length, length)).astype('float32')
        for i in range(length):
            word = word_list[i]
            if word in sn.data.keys():
                sentic = float(sn.concept(word)["polarity_value"]) + 1.0  # +1 保证平滑
            else:
                sentic = 0
            # 相加合并
            for j in range(length):
                matrix[i][j] += sentic
                matrix[j][i] += sentic
        for i in range(length):
            if matrix[i][i] == 0:
                matrix[i][i] = 1  # 0 项全部 +1 保证平滑

        # 对 PLM BPE 划分过的 token 要保证对齐
        j = 1
        for i in range(length):
            # 实际 token 和 tokenize 后的 token 不同，说明划分了 BPE，要执行对齐
            results[batch_id, j, :] = torch.from_numpy(matrix[i, :])
            if word_list[i] != tokenized_tokens[j]:
                j += 1
                while '##' in tokenized_tokens[j]:  # 确定切分 BPE 的跨度
                    results[batch_id, j, :] = torch.from_numpy(matrix[i, :])
                    results[batch_id, :, j] = torch.from_numpy(matrix[:, i])
                    j += 1

        # 从 1 开始取，空出 [CLS] 的位置
        # results[batch_id, 1:length+1, 1:length+1] = torch.from_numpy(matrix)

    return results


if __name__ == "__main__":
    # sents = [
    # 'importantortantandme and thebun , the additional menu items are only written in Chinese .',
    # 'Great atmoshere and worth every bit .',
    # 'Pair you food with the excellent beers on tap or their well priced wine list .',
    # 'Go here for a romantic dinner but not for an all out wow dining experience .',
    # 'Spreads and toppings are great - though a bit pricey .', 'Food is excellent .',
    # 'I have been a longtime fan of Holy Basil in the East Village , and while I do believe their food has slightly slipped in quality , I have been hesitant to be disloyal .',
    # "When he finally did , he was unable to make a gin and tonic -- could n't find tonic .",
    # 'I would not have been so disappointed with the portions if the qualities were good enough to make up for it , but they were not !',
    # 'May , the owner always has a smile on her and will warmly greet you .',
    # 'Plus , on Wednesday nights the house wine is unlimited !',
    # 'delicious simple food in nice outdoor atmosphere .',
    # "The one vegetarian entree ( Abby 's treasure ) was actually quite a surprise - it was delicious and had wintermelon covering an assortment of fresh mushrooms and vegetables .",
    # 'After all that , they complained to me about the small tip .',
    # 'The food is so cheap and the waiters are nice .',
    # 'All the food was hot tasty .'
    # ]
    # r = get_sentic(sents)
    # print(r)

    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # r = tokenizer.tokenize("importantortantandme and thebun")
    # print(r)
    pass
