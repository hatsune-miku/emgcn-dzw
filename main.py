# coding utf-8

import os
import json
import random
import argparse

import torch

from tqdm import trange
from argparse import Namespace
from typing import List, NamedTuple, Union, Tuple
from data import load_data_instances, DataIterator, label2id, Instance
from model import EMCGCN
from torch import Tensor
import utils
import pickle
import numpy as np

from prepare_vocab import VocabHelp
from transformers import AdamW, BertTokenizer, BatchEncoding, BertModel
from senticnet.senticnet import SenticNet
from torch.optim import Optimizer

# Typealias for Namespace.
Arguments = Namespace


def get_bert_optimizer(model: torch.nn.Module, args: Arguments) -> Optimizer:
    # # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    diff_part = ["bert.embeddings", "bert.encoder"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
            "weight_decay": args.weight_decay,
            "lr": args.bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.learning_rate
        },
    ]

    return AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)


def get_perturbed_matrix(args: Arguments, sentence_ids: List[str], mode: str) -> Tensor:
    file = open(args.prefix + args.dataset + '/' + mode + '.json_' + args.pm_model_class + '_matrix.pickle', 'rb')
    matrix_dict = pickle.load(file)

    # 将对应句子的 matrix 转为 tensor
    matrix = [torch.Tensor(v).to(args.device) for k, v in matrix_dict.items() if str(k) in sentence_ids]

    # 对不同长te的 tensor 做 zero pad 补成 batch_size * max_sequence_len * max_sequence_len
    results = torch.zeros((len(matrix), args.max_sequence_len, args.max_sequence_len), device=args.device)

    for i in range(len(results)):
        # NOTE 把 CLS 和 SEP 行设为全 0
        matrix[i][0, :] = torch.zeros(len(matrix[i][0, :]))
        matrix[i][-1, :] = torch.zeros(len(matrix[i][-1, :]))
        results[i, :len(matrix[i]), :len(matrix[i])] = matrix[i]

    return results


def get_sentence_embeddings(sentence: str, model: EMCGCN) -> Tensor:
    tokenizer: BertTokenizer = model.tokenizer
    bert_model: BertModel = model.bert_model
    input_ids = tokenizer(sentence, return_tensors='pt')['input_ids']

    with torch.no_grad():
        outputs = bert_model.forward(input_ids=input_ids)
        sentence_embeddings: torch.FloatTensor = outputs.last_hidden_state

    # Implicit conversion to Tensor - FloatTensor is a subclass of Tensor
    return sentence_embeddings


def create_unperturbed_matrix(
    sentence: str,
    device: str,
    max_sequence_len: int,
    model: EMCGCN,
) -> Tensor:
    # 得到词向量
    sentence_embeddings: Tensor = get_sentence_embeddings(sentence, model)

    # Squeezed to zero because we only have one sentence
    unprocessed_unperturbed_matrix: Tensor = sentence_embeddings.squeeze(0)
    matrix_list = [unprocessed_unperturbed_matrix.to(device)]

    # One because only one sentence
    processed_unperturbed_matrix = torch.zeros((1, max_sequence_len, max_sequence_len), device=device)

    # NOTE 把 CLS 和 SEP 行设为全 0
    matrix_list[0][0, :] = torch.zeros(len(matrix_list[0][0, :]))
    matrix_list[0][-1, :] = torch.zeros(len(matrix_list[0][-1, :]))
    processed_unperturbed_matrix[0, :len(matrix_list[0]), :len(matrix_list[0])] = matrix_list[0]

    return processed_unperturbed_matrix


class GetSenticArguments(NamedTuple):
    tokenizer: BertTokenizer
    max_sequence_len: int
    device: str


def get_sentic(
    args: Union[Arguments, GetSenticArguments],
    sentences: List[str],
    token_ids: Union[List[int], Tensor],
    sn: SenticNet = SenticNet()
) -> Tensor:
    """
    返回情感矩阵
    Return: [batch_size, max_seq_len, max_seq_len]: Tensor
    """
    tokenizer: BertTokenizer = args.tokenizer

    results = torch.zeros((len(sentences), args.max_sequence_len, args.max_sequence_len), device=args.device)
    for batch_id, sentence in enumerate(sentences):
        tokenized_tokens = tokenizer.convert_ids_to_tokens(token_ids[batch_id])
        word_list = sentence.split()

        # 确保长度不能超过 max_seq_len，且考虑 [CLS] [SEP] 标签
        length = min(len(word_list), args.max_sequence_len - 2)

        # matrix = np.zeros((length, length)).astype('float32')
        # 初始化matrix变量时，将其设为全1，而不是全0，因为乘以0会得到全0的结果。
        matrix = np.ones((length, length)).astype('float32')

        for i in range(length):
            word = word_list[i]

            if word in sn.data.keys():
                # +1 保证平滑
                sentic = float(sn.concept(word)["polarity_value"]) + 1.0
            else:
                # 不加1
                sentic = 0

            # 相加合并
            for j in range(length):
                matrix[i][j] *= sentic
                matrix[j][i] *= sentic

        # 乘法删除将对角线元素设为1的代码，因为现在不再需要这样的操作
        # 如果你希望矩阵对角线上的元素不受乘法和加法的影响，仍然保持为1，可以在乘法操作之后将对角线元素重新设置为1。
        for i in range(length):
            matrix[i][i] = 1

        # 对 PLM BPE 划分过的 token 要保证对齐
        j = 1

        for i in range(length):
            # 实际 token 和 tokenize 后的 token 不同，说明划分了 BPE，要执行对齐
            results[batch_id, j, 1:length + 1] = torch.from_numpy(matrix[i, :])
            results[batch_id, 1:length + 1, j] = torch.from_numpy(matrix[:, i])

            if word_list[i].lower() != tokenized_tokens[j].lower():
                # 确定切分 BPE 的跨度
                while '##' in tokenized_tokens[j + 1]:
                    j += 1
                    results[batch_id, j, 1:length + 1] = torch.from_numpy(matrix[i, :])
                    results[batch_id, 1:length + 1, j] = torch.from_numpy(matrix[:, i])
            j += 1

    return results


def train(args: Arguments) -> None:
    """
    练！
    """

    # 从json文件中读取数据 训练集
    train_sentence_packs = json.load(open(args.prefix + args.dataset + '/train.json'))

    # 打乱句子顺序
    random.shuffle(train_sentence_packs)

    # 验证集
    dev_sentence_packs = json.load(open(args.prefix + args.dataset + '/dev.json'))

    # load 四个句子特征各自的类别。这四个都是什么意思？应该是四个句子特征各自的标签类别信息
    # 加载文件
    post_vocab: VocabHelp = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_post.vocab')
    deprel_vocab: VocabHelp = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_deprel.vocab')
    postag_vocab: VocabHelp = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_postag.vocab')
    synpost_vocab: VocabHelp = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_synpost.vocab')

    args.post_size = len(post_vocab)  # 81
    args.deprel_size = len(deprel_vocab)  # 45
    args.postag_size = len(postag_vocab)  # 855
    args.synpost_size = len(synpost_vocab)  # 7

    # 传入训练数据，构建模型用到的实例
    instances_train = load_data_instances(
        train_sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args)

    # 传入验证集数据，构建模型用到的实例
    instances_dev = load_data_instances(
        dev_sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args)

    # 打乱训练集形成的实例的顺序
    random.shuffle(instances_train)

    # 这个DataIterator的作用应该是进行分批操作batch
    trainset = DataIterator(instances_train, args)

    # 同上，只不过是验证集
    devset = DataIterator(instances_dev, args)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # 初始化模型，装载模型到device
    model = EMCGCN(args).to(args.device)

    # 优化器
    optimizer = get_bert_optimizer(model, args)

    # label = ['N', 'B-A', 'I-A', 'A', 'B-O', 'I-O', 'O', 'negative', 'neutral', 'positive']
    # 这个weight是干什么用的？
    weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).float().to(args.device)

    best_joint_f1 = 0
    best_joint_epoch = 0

    for i in range(args.epochs):
        # 第几次循环数据集epoch
        print(f'Epoch: {i}')

        # 第几个批次batch。trange()用来显示进度条以及展示每一轮（iteration)所耗费的时间。
        for j in trange(trainset.batch_count):

            # sentences: 句子, tokens: 句子分词之后的结果,
            # lengths: 词经过toid操作之后的tokens列表长度,
            # masks: 一个max_sequence_lenght的列表，对应句子token长度的位置为1，其他位置为0
            # aspect_tags: 列表，padding部分为-1，其他部分为0，
            # aspect部分是aspect第一个词的第一个token为1，中间词的第一个token为2
            # tags: 矩阵，十种关系的哪一种，不是都是-1
            # tags_symmetry: 对称矩阵，对角线为-1，其他位置为0

            batch = trainset.get_batch(j)
            sentic_matrixs = get_sentic(args, batch.sentences, batch.bert_tokens)
            tags_flatten = batch.tags.reshape([-1])  # 将矩阵展平 16 x 102 x 102 -> 16*102*102
            tags_symmetry_flatten = batch.tags_symmetry.reshape([-1])  # 将矩阵展平

            perturbed_matrix = get_perturbed_matrix(args, batch.sentence_ids, 'train')

            # 真实的矩阵和通过模型计算预测得到的矩阵之间进行求交叉熵进行拟合
            # 这个过程在算loss
            if args.relation_constraint:
                predictions = model.forward(
                    batch.bert_tokens,
                    batch.masks,
                    sentic_matrixs,
                    perturbed_matrix,
                    batch.word_pair_position,
                    batch.word_pair_deprel,
                    batch.word_pair_pos,
                    batch.word_pair_synpost
                )
                sentic_pred, biaffine_pred, post_pred, deprel_pred, postag, synpost, final_pred = predictions[0], \
                    predictions[1], predictions[2], predictions[3], predictions[4], predictions[5], predictions[6]

                with torch.nn.functional as f:
                    l_se = 0.10 * f.cross_entropy(
                        sentic_pred.reshape([-1, sentic_pred.shape[3]]),
                        tags_symmetry_flatten,
                        ignore_index=-1
                    )

                    l_ba = 0.10 * f.cross_entropy(
                        biaffine_pred.reshape([-1, biaffine_pred.shape[3]]),
                        tags_symmetry_flatten,
                        ignore_index=-1
                    )

                    l_rpd = 0.01 * f.cross_entropy(
                        post_pred.reshape([-1, post_pred.shape[3]]),
                        tags_symmetry_flatten,
                        ignore_index=-1
                    )

                    l_dep = 0.01 * f.cross_entropy(
                        deprel_pred.reshape([-1, deprel_pred.shape[3]]),
                        tags_symmetry_flatten,
                        ignore_index=-1
                    )

                    l_psc = 0.01 * f.cross_entropy(
                        postag.reshape([-1, postag.shape[3]]),
                        tags_symmetry_flatten,
                        ignore_index=-1
                    )

                    l_tbd = 0.01 * f.cross_entropy(
                        synpost.reshape([-1, synpost.shape[3]]),
                        tags_symmetry_flatten,
                        ignore_index=-1
                    )

                # 意思是仅仅解码对角线的值，对角线的值无非是方面词和意见词，没有情感极性的预测
                # 默认是进行元组的抽取 16 x 102 x 102 x 10 -> reshape()操作 -> 16*102*102 x 10
                l_p = f.cross_entropy(
                    final_pred.reshape([-1, final_pred.shape[3]]),
                    tags_symmetry_flatten if args.symmetry_decoding else tags_flatten,
                    weight=weight,
                    ignore_index=-1
                )
                loss = l_se + l_ba + l_rpd + l_dep + l_psc + l_tbd + l_p
            else:
                # 这个就是仅仅只有Biaffine Attention的那个Loss
                preds = model.forward(
                    batch.bert_tokens,
                    batch.masks,
                    sentic_matrixs,
                    perturbed_matrix,
                    batch.word_pair_position,
                    batch.word_pair_deprel,
                    batch.word_pair_pos,
                    batch.word_pair_synpost
                )
                preds = preds[-1]
                preds_flatten = preds.reshape([-1, preds.shape[3]])
                loss = f.cross_entropy(
                    preds_flatten,
                    tags_symmetry_flatten if args.symmetry_decoding else tags_flatten,
                    weight=weight,
                    ignore_index=-1
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        eval_result = model_eval(model, devset, args, test_dev='dev')

        if eval_result.f1 > best_joint_f1:  # 得到最优的f1值
            model_path = args.model_dir + 'bert' + args.task + '.pt'
            torch.save(model, model_path)
            best_joint_f1 = eval_result.f1
            best_joint_epoch = i

    print('best epoch: {}\tbest dev {} f1: {:.5f}\n\n'.format(
        best_joint_epoch, args.task, best_joint_f1))  # 输出


class ModelEvalResult(NamedTuple):
    precision: float
    recall: float
    f1: float


def model_predict_single(
    text: str,
    model: EMCGCN,
    post_vocab: VocabHelp,
    deprel_vocab: VocabHelp,
    postag_vocab: VocabHelp,
    synpost_vocab: VocabHelp,
    device: str = 'cuda',
    max_sequence_len: int = 102
) -> List[Tuple[List[int], List[int], List[int]]]:
    """
    预测单个句子，返回预测结果
    """
    tokenizer: BertTokenizer = model.tokenizer

    # 对输入的文本进行分词
    encoded_dict: BatchEncoding = tokenizer(text, return_tensors='pt')

    # 提取结果（ids和mask）
    input_ids: List[int] = encoded_dict['input_ids']
    mask = encoded_dict['attention_mask']
    masks = torch.tensor(mask)

    # ids转为tokens
    tokens: List[str] = tokenizer.convert_ids_to_tokens(input_ids)

    # 只有一个句子呀，那就一个吧
    sentences = [text]
    sentic_matrixs = get_sentic(
        args=GetSenticArguments(
            tokenizer=tokenizer,
            max_sequence_len=max_sequence_len,
            device=device
        ),
        sentences=sentences,
        token_ids=input_ids
    )

    # 预测需要的是unperturbed_matrix
    perturbed_matrix = create_unperturbed_matrix(text, device, max_sequence_len, model)

    token_range = Instance.get_token_range(
        tokens=tokens,
        tokenizer=tokenizer
    )

    word_pair_position = Instance.get_word_pair_position(
        max_sequence_len=max_sequence_len,
        tokens=tokens,
        token_range=token_range,
        post_vocab=post_vocab
    )

    # TODO: head和deprel
    word_pair_deprel = Instance.get_word_pair_deprel(
        max_sequence_len=max_sequence_len,
        head=[0] * len(tokens),
        tokens=tokens,
        deprel=[''] * len(tokens),
        token_range=token_range,
        deprel_vocab=deprel_vocab
    )

    # TODO: postag
    word_pair_pos = Instance.get_word_pair_pos(
        max_sequence_len=max_sequence_len,
        tokens=tokens,
        postag=[''] * len(tokens),
        token_range=token_range,
        postag_vocab=postag_vocab
    )

    # TODO: head
    word_pair_synpost = Instance.get_word_pair_synpost(
        max_sequence_len=max_sequence_len,
        head=[0] * len(tokens),
        tokens=tokens,
        token_range=token_range,
        synpost_vocab=synpost_vocab
    )

    tokens: Tensor = torch.tensor(tokens)

    with torch.no_grad():
        predict_list: List[Tensor] = model.forward(
            tokens=tokens,
            masks=masks,
            sentic_matrixs=sentic_matrixs,
            perturbed_matrix=perturbed_matrix,
            word_pair_position=word_pair_position,
            word_pair_deprel=word_pair_deprel,
            word_pair_pos=word_pair_pos,
            word_pair_synpost=word_pair_synpost
        )

        predict: Tensor = predict_list[-1]
        predict = torch.nn.functional.softmax(predict, dim=-1)
        predict = torch.argmax(predict, dim=3)

    # sen_length说的是token的数量
    sen_length = len(tokens)
    predict_cpu: List[List[int]] = predict.cpu().tolist()

    opinions = utils.get_opinions(
        tags=predict_cpu,
        length=sen_length,
        token_range=token_range
    )

    aspects = utils.get_aspects(
        tags=predict_cpu,
        length=sen_length,
        token_range=token_range
    )

    polarities = utils.get_polarities(
        tags=predict_cpu,
        length=sen_length,
        token_range=token_range
    )

    triple_list = list(zip(opinions, aspects, polarities))
    return triple_list


def model_eval(
    model: EMCGCN,
    dataset: DataIterator,
    args: Arguments,
    should_report: bool = False,
    test_dev: str = 'test'
) -> ModelEvalResult:
    # 在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。
    with torch.no_grad():
        all_ids = []
        all_sentences = []
        all_preds = []
        all_labels = []
        all_lengths = []
        all_sens_lengths = []
        all_token_ranges = []
        for i in range(dataset.batch_count):
            batch = dataset.get_batch(i)
            sentic_matrixs = get_sentic(args, batch.sentences, batch.bert_tokens)
            perturbed_matrix = get_perturbed_matrix(args, batch.sentence_ids, test_dev)

            preds = model.forward(
                tokens=batch.bert_tokens,
                masks=batch.masks,
                sentic_matrixs=sentic_matrixs,
                perturbed_matrix=perturbed_matrix,
                word_pair_position=batch.word_pair_position,
                word_pair_deprel=batch.word_pair_deprel,
                word_pair_pos=batch.word_pair_pos,
                word_pair_synpost=batch.word_pair_synpost
            )
            preds = preds[-1]
            preds = torch.nn.functional.softmax(preds, dim=-1)
            preds = torch.argmax(preds, dim=3)
            all_preds.append(preds)
            all_labels.append(batch.tags)
            all_lengths.append(batch.lengths)
            all_sens_lengths.extend(batch.sens_lens)
            all_token_ranges.extend(batch.token_ranges)
            all_ids.extend(batch.sentence_ids)
            all_sentences.extend(batch.sentences)

        all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
        all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
        all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()

        metric = utils.Metric(
            args=args,
            predictions=all_preds,
            goldens=all_labels,
            bert_lengths=all_lengths,
            sen_lengths=all_sens_lengths,
            tokens_ranges=all_token_ranges
        )
        precision, recall, f1 = metric.score_uniontags()
        aspect_results = metric.score_aspect()
        opinion_results = metric.score_opinion()
        print('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1],
                                                                  aspect_results[2]))
        print('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1],
                                                                   opinion_results[2]))
        print(args.task + '\t\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

        if should_report:
            metric.tagReport()

    model.train()
    return ModelEvalResult(precision, recall, f1)


def test(args: Arguments):
    print("Evaluation on testset:")
    model_path = args.model_dir + 'bert' + args.task + '.pt'
    model: EMCGCN = torch.load(model_path).to(args.device)

    sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
    post_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_post.vocab')
    deprel_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_deprel.vocab')
    postag_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_postag.vocab')
    synpost_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_synpost.vocab')
    instances = load_data_instances(sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args)
    testset = DataIterator(instances, args)
    model_eval(model, testset, args, False)


def main():
    parser = argparse.ArgumentParser()

    # 数据集文件路径前缀
    parser.add_argument('--prefix', type=str, default="data/D1/",
                        help='dataset and embedding path prefix')

    # 模型地址路径前缀
    parser.add_argument('--model_dir', type=str, default="savemodel/",
                        help='model path prefix')

    # 任务类型
    parser.add_argument('--task', type=str, default="triplet", choices=["triplet"],
                        help='option: pair, triplet')

    # 模式：训练 or 测试
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test')

    # 数据集类型
    parser.add_argument('--dataset', type=str, default="res14", choices=["res14", "lap14", "res15", "res16"],
                        help='dataset')

    # 输入的句子的最大长度
    parser.add_argument('--max_sequence_len', type=int, default=102,
                        help='max length of a sentence')

    # 训练设备：gpu or cpu
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')

    parser.add_argument('--bert_model_path', type=str,
                        default="bert-base-uncased",
                        help='pretrained bert model path')  # 预训练的模型路径

    parser.add_argument('--bert_feature_dim', type=int, default=768,
                        help='dimension of pretrained bert feature')  # bert的输入向量的维度
    parser.add_argument('--pm_model_class', type=str, default='bert', choices=['bert', 'robert'])
    parser.add_argument('--batch_size', type=int, default=16,
                        help='bathc size')  # 但批输入数据量
    parser.add_argument('--epochs', type=int, default=100,
                        help='training epoch number')  # 轮次
    parser.add_argument('--class_num', type=int, default=len(label2id),
                        help='label number')  # 标签个数，10种关系
    parser.add_argument('--seed', default=1000, type=int)  # 随机种子
    parser.add_argument('--learning_rate', default=1e-3, type=float)  # 学习率
    parser.add_argument('--bert_lr', default=2e-5, type=float)  # bert学习率
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")  # 优化器
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")  #

    parser.add_argument('--emb_dropout', type=float, default=0.5)  #
    parser.add_argument('--num_layers', type=int, default=1)  #
    parser.add_argument('--pooling', default='avg', type=str, help='[max, avg, sum]')  # 池化类型
    parser.add_argument('--gcn_dim', type=int, default=300, help='dimension of GCN')  # GCN向量的维度
    parser.add_argument('--relation_constraint', default=False, action='store_true')  #
    parser.add_argument('--symmetry_decoding', default=False, action='store_true')  #

    args: Namespace = parser.parse_args()  # 解析参数

    print(args)

    args.tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)

    # 固定随机数的数值，使每次产生的随机数是一致的，进而保证相同输入下，输出是相同的，
    # 因为初始化权重矩阵的随机参数可能有很多种，我们要每次实验都要他一样，才能进行实验
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

        # 为cpu设置种子，产生随机数
        torch.manual_seed(args.seed)

        # 为gpu设置种子，产生随机数
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

        # 如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的
        torch.backends.cudnn.benchmark = False

    if args.task == 'triplet':
        # 任务为三元组： 类别个数设置为10
        args.class_num = len(label2id)

    if args.mode == 'train':
        train(args)
    test(args)


if __name__ == '__main__':
    main()
