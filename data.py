import argparse
import math
import torch
from torch import Tensor
from typing import List, Any, NamedTuple
from collections import OrderedDict, defaultdict
from transformers import BertTokenizer
from prepare_vocab import VocabHelp

# TODO: unused
# sentiment2id = {'negative': 3, 'neutral': 4, 'positive': 5}

# A type alias for JSON.
JSON = Any
Arguments = argparse.Namespace

# A token range is list of 2-tuples, the `i`th element represents
# the (start and end index pair) of the `i`th token.
TokenRange = List[List[int]]


def make_label_id_maps() -> (OrderedDict, OrderedDict):
    """
    Returns label2id and id2label ordered maps
    """
    label = ['N', 'B-A', 'I-A', 'A', 'B-O', 'I-O', 'O', 'negative', 'neutral', 'positive']
    l2id = OrderedDict()
    ld2l = OrderedDict()
    for i, v in enumerate(label):
        l2id[v] = i
        ld2l[i] = v
    return l2id, ld2l


label2id, id2label = make_label_id_maps()


# 返回标签标记的index范围
def get_spans(tags):
    """for BIO tag"""

    tags = tags.strip().split()
    length = len(tags)
    spans = []
    start = -1

    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i

        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1

    if start != -1:
        spans.append([start, length - 1])

    return spans


def get_evaluate_spans(tags, length, token_range):
    """for BIO tag"""

    spans = []
    start = -1

    for i in range(length):
        l, r = token_range[i]
        if tags[l] == -1:
            continue

        elif tags[l] == 1:
            if start != -1:
                spans.append([start, i - 1])
            start = i

        elif tags[l] == 0:
            if start != -1:
                spans.append([start, i - 1])
                start = -1

    if start != -1:
        spans.append([start, length - 1])

    return spans


# 需要处理的单个句子的各种值的初始化
class Instance(object):
    # TODO: 这构造函数的逻辑太复杂了，考虑是不是真的要这样做
    def __init__(
        self,
        tokenizer: BertTokenizer,
        sentence_pack: JSON,
        post_vocab: VocabHelp,
        deprel_vocab: VocabHelp,
        postag_vocab: VocabHelp,
        synpost_vocab: VocabHelp,
        args
    ):
        # 句子id
        self.id: str = sentence_pack['id']

        # 句子
        self.sentence: str = sentence_pack['sentence']

        # 分词，将句子进行分词得到的词列表;
        # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        self.tokens: List[str] = self.sentence.strip().split()

        # 词性标注信息，目的是确定词的词性
        self.postag: List[str] = sentence_pack['postag']

        # 每个词在依赖树中依赖谁，0代表root
        self.head: List[int] = sentence_pack['head']

        # 每个词在依赖树中和其依赖的head的关系
        self.deprel: List[str] = sentence_pack['deprel']

        # 句子长度
        self.sen_length: int = len(self.tokens)

        # 每个词对应的形成的数字的开始地址和结束地址，有的时候一次词由两个数字才可以表示
        self.token_range: TokenRange = []

        # 将句子序列转换成数字序列，并默认自动加入开始CLS和SEP
        self.bert_tokens: List[int] = tokenizer.encode(self.sentence)

        # 初始化bert的输入的各种用到的序列
        # 转换成数字序列并加入CLS和SEP之后的长度
        self.length: int = len(self.bert_tokens)

        # bert的输入序列统一长度max_sequence
        self.bert_tokens_padding: Tensor = torch.zeros(args.max_sequence_len).long()

        # 用来记录序列中属于方面词的位置
        self.aspect_tags: Tensor = torch.zeros(args.max_sequence_len).long()

        # 用来记录序列中属于意见词的位置
        self.opinion_tags: Tensor = torch.zeros(args.max_sequence_len).long()

        # 最后预测的矩阵的真实标签
        self.tags: Tensor = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()

        # 最后预测的矩阵上对角线位置上的真实标签
        self.tags_symmetry: Tensor = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()

        # 对输入的序列没有用的部分进行mask
        self.mask: Tensor = torch.zeros(args.max_sequence_len)

        for i in range(self.length):
            # 句子长度未达到最大长度对后面的位置而进行的padding补0操作，最后形成前面是句子对应的数字序列，后面都是0
            self.bert_tokens_padding[i] = self.bert_tokens[i]

        # 将mask序列全置为1，一样的道理，不mask的位置为1， mask的位置为0
        self.mask[:self.length] = 1

        # 计算每个token对应的数字序列的开始地址和结束地址,因为有的词需要占用两个位置
        self.token_range = Instance.get_token_range(self.tokens, tokenizer)
        assert self.length == self.token_range[-1][-1] + 2  # 多了CLS, SEP，

        # padding部分设置为-1
        self.aspect_tags[self.length:] = -1

        # 第一个为cls，不可能为aspect,设置为-1
        self.aspect_tags[0] = -1

        # 最后一个为seq结束符，不可能为aspect, 设置为-1
        self.aspect_tags[self.length - 1] = -1

        # 和aspect进行同样的操作
        self.opinion_tags[self.length:] = -1
        self.opinion_tags[0] = -1
        self.opinion_tags[self.length - 1] = -1

        # 先将标签矩阵全部初始化为-1，然后有用的地方再赋值成相应的值， tags对应的是triplet任务
        self.tags[:, :] = -1

        # 同上tags_symmetry对应pairs任务
        self.tags_symmetry[:, :] = -1

        # 句子所在index的范围都变成0
        for i in range(1, self.length - 1):

            # 上三角就够用了，为了减少计算量
            for j in range(i, self.length - 1):
                self.tags[i][j] = 0

        # 虽然我们从json中读取的数据，但是我们要将其转换成机器识别的形式，
        # 所以我们要对各种用到的矩阵，数组进行赋值，初始化
        # 迭代单个句子中的每组三元组
        for triple in sentence_pack['triples']:
            # 方面词标记序列
            aspect = triple['target_tags']

            # 意见词标记序列
            opinion = triple['opinion_tags']

            # 通过get_spans()方法获取方面词所在index范围
            aspect_span = get_spans(aspect)

            # 意见词所在index范围
            opinion_span = get_spans(opinion)

            # set tag for aspect
            # aspect_span说的是句子中的方面词开始index和结束index
            for left, right in aspect_span:
                # 为什么这样弄，因为将词转换为bert id时，有可能一个词用两个id才可以表示，所以index不相互对应
                start = self.token_range[left][0]
                end = self.token_range[right][1]

                # 对tags矩阵进行贴标签
                for i in range(start, end + 1):
                    for j in range(i, end + 1):
                        if j == start:
                            # j == start 说明它时方面词词组的第一个词
                            self.tags[i][j] = label2id['B-A']
                        elif j == i:
                            # j != start && j == i 说明它是属于方面词的内部
                            # i == j 要不是B-A, 要不是I-A
                            self.tags[i][j] = label2id['I-A']
                        else:
                            # 同属于一个方面词, i != j就是同属于一个方面词组
                            self.tags[i][j] = label2id['A']

                # 给aspect_tags, tags贴标签，
                # 是aspect的设为1或2，
                # 不是的设为-1，
                # 若一个词由多个token表示，则设该词的第一个token为1，其他的token为-1
                for i in range(left, right + 1):
                    # i == l 代表方面词的第一个词
                    set_tag = 1 if i == left else 2

                    # 方面词中词i的开始位置和结束位置
                    al, ar = self.token_range[i]
                    self.aspect_tags[al] = set_tag

                    # 除方面词第一个token之外的其他方面词tokens设置为-1
                    self.aspect_tags[(al + 1):(ar + 1)] = -1

                    # mask positions of sub words
                    # 行和列全都设置为-1
                    self.tags[(al + 1):(ar + 1), :] = -1
                    self.tags[:, (al + 1):(ar + 1)] = -1

            # set tag for opinion
            for left, right in opinion_span:
                # 开始那个词对应的tokens的开始的token
                # 到结束那个词对应的tokens的结束的token
                start = self.token_range[left][0]
                end = self.token_range[right][1]

                # 对tags矩阵进行贴标签
                for i in range(start, end + 1):
                    for j in range(i, end + 1):
                        # i = j时，判断token属于什么类别， i != j时，判断他们之间的关系
                        if j == start:
                            # B-O, I-O都是位于对角线的
                            self.tags[i][j] = label2id['B-O']
                        elif j == i:
                            self.tags[i][j] = label2id['I-O']
                        else:
                            # 同属于一个方面词组对应的tokens里面
                            self.tags[i][j] = label2id['O']

                for i in range(left, right + 1):
                    # 方面词组的第一个词为1，其他的词为2
                    set_tag = 1 if i == left else 2

                    # 获取词对应的token的开始index和结束index
                    pl, pr = self.token_range[i]

                    # 方面词组的第一个词的第一个token设为1，第一个词的其他token设为-1；
                    # 方面词组的除了第一个方面词的词对应的token的第一个token设为2， 其他token设为-1
                    self.opinion_tags[pl] = set_tag
                    self.opinion_tags[(pl + 1):(pr + 1)] = -1

                    # tags矩阵的上述说的位置也是同样的规则进行初始化
                    self.tags[(pl + 1):(pr + 1), :] = -1
                    self.tags[:, (pl + 1):(pr + 1)] = -1

            # 一个方面词一个意见的这样的关系，在tags矩阵中对应的位置进行赋值，对应POS, NEG, NEU, T
            # 方面词组在句子中的跨度，如果有多个方面词组，就迭代多个
            for al, ar in aspect_span:

                # 意见词组在句子中的跨度
                for pl, pr in opinion_span:

                    # 方面词在句子中的跨度的迭代
                    for i in range(al, ar + 1):

                        # 意见词在句子中的跨度的迭代
                        for j in range(pl, pr + 1):
                            # 方面词对应的token的开始index和结束index
                            sal, sar = self.token_range[i]

                            # 意见词对应的token的开始index和结束index
                            spl, spr = self.token_range[j]

                            # 一个是aspect,一个时opinion这样的关系的位置的值设为-1，
                            # 先全设为-1，下面对有关系的地方再进行赋值
                            self.tags[sal:(sar + 1), spl:(spr + 1)] = -1

                            # 任务类型为输出pair, 不是triplet
                            if args.task == 'pair':
                                # 方面词在意见词后面
                                if i > j:
                                    # 设为7，7对应的label为negative,这个是随便设的，
                                    # 不为-1就说明他们两个具备a-o关系，不需要识别情感
                                    self.tags[spl][sal] = 7
                                else:
                                    # 方面词在意见词前面
                                    self.tags[sal][spl] = 7
                            elif args.task == 'triplet':
                                # 这个大小关系的目的是因为我仅仅用到上三角，
                                # 所以遇到这个标签在下三角的我把它调到上三角
                                if i > j:
                                    self.tags[spl][sal] = label2id[triple['sentiment']]
                                else:
                                    self.tags[sal][spl] = label2id[triple['sentiment']]

        # 将tags_symmetry赋值成和tags矩阵对角线一样的内容，其他位置用不到
        for i in range(1, self.length - 1):
            for j in range(i, self.length - 1):
                self.tags_symmetry[i][j] = self.tags[i][j]
                self.tags_symmetry[j][i] = self.tags_symmetry[i][j]

        # 1. generate position index of the word pair
        self.word_pair_position: Tensor = Instance.get_word_pair_position(
            max_sequence_len=args.max_sequence_len,
            tokens=self.tokens,
            token_range=self.token_range,
            post_vocab=post_vocab
        )

        # 2. generate deprel index of the word pair
        self.word_pair_deprel: Tensor = Instance.get_word_pair_deprel(
            max_sequence_len=args.max_sequence_len,
            head=self.head,
            tokens=self.tokens,
            deprel=self.deprel,
            token_range=self.token_range,
            deprel_vocab=deprel_vocab
        )

        # 3. generate POS tag index of the word pair
        # 词性标注
        self.word_pair_pos: Tensor = Instance.get_word_pair_pos(
            max_sequence_len=args.max_sequence_len,
            tokens=self.tokens,
            postag=self.postag,
            token_range=self.token_range,
            postag_vocab=postag_vocab
        )

        # 4. generate synpost index of the word pair
        # 基于句法的相对位置
        self.word_pair_synpost: Tensor = Instance.get_word_pair_synpost(
            max_sequence_len=args.max_sequence_len,
            head=self.head,
            tokens=self.tokens,
            token_range=self.token_range,
            synpost_vocab=synpost_vocab
        )

    @staticmethod
    def get_token_range(
        tokens: List[str],
        tokenizer: BertTokenizer,
    ) -> TokenRange:
        token_start = 1
        ret: TokenRange = []
        for i, w, in enumerate(tokens):
            token_end = token_start + len(tokenizer.encode(w, add_special_tokens=False))
            ret.append([token_start, token_end - 1])
            token_start = token_end
        return ret

    @staticmethod
    def get_word_pair_position(
        max_sequence_len: int,
        tokens: List[str],
        token_range: TokenRange,
        post_vocab: VocabHelp
    ) -> Tensor:
        ret: Tensor = torch.zeros(max_sequence_len, max_sequence_len).long()

        # 句子中词的个数的迭代
        for i in range(0, len(tokens)):
            # 每个词对应的token的开始index和结束index
            start = token_range[i][0]
            end = token_range[i][1]

            for j in range(0, len(tokens)):
                s = token_range[j][0]
                e = token_range[j][1]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        ret[row][col] = post_vocab.stoi.get(abs(row - col), post_vocab.unk_index)
        return ret

    @staticmethod
    def get_word_pair_deprel(
        max_sequence_len: int,
        head: List[int],
        tokens: List[str],
        deprel: List[str],
        token_range: TokenRange,
        deprel_vocab: VocabHelp
    ) -> Tensor:
        # 2. generate deprel index of the word pair
        # 依赖关系
        ret: Tensor = torch.zeros(max_sequence_len, max_sequence_len).long()
        for i in range(0, len(tokens)):
            start = token_range[i][0]
            end = token_range[i][1]

            for j in range(start, end + 1):
                if head[i] != 0:
                    s, e = token_range[head[i] - 1]
                else:
                    s, e = 0, 0

                for k in range(s, e + 1):
                    ret[j][k] = deprel_vocab.stoi.get(deprel[i])
                    ret[k][j] = deprel_vocab.stoi.get(deprel[i])
                    ret[j][j] = deprel_vocab.stoi.get('self')
        return ret

    @staticmethod
    def get_word_pair_pos(
        max_sequence_len: int,
        tokens: List[str],
        postag: List[str],
        token_range: TokenRange,
        postag_vocab: VocabHelp
    ) -> Tensor:
        ret: Tensor = torch.zeros(max_sequence_len, max_sequence_len).long()
        for i in range(0, len(tokens)):
            start, end = token_range[i][0], token_range[i][1]

            for j in range(0, len(tokens)):
                s, e = token_range[j][0], token_range[j][1]

                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        ret[row][col] = postag_vocab.stoi.get(
                            tuple(sorted([postag[i], postag[j]]))
                        )
        return ret

    @staticmethod
    def get_word_pair_synpost(
        max_sequence_len: int,
        head: List[int],
        tokens: List[str],
        token_range: TokenRange,
        synpost_vocab: VocabHelp
    ) -> Tensor:
        ret: Tensor = torch.zeros(max_sequence_len, max_sequence_len).long()
        tmp = [[0] * len(tokens) for _ in range(len(tokens))]

        for i in range(0, len(tokens)):
            j = head[i]
            if j == 0:
                continue
            tmp[i][j - 1] = 1
            tmp[j - 1][i] = 1

        tmp_dict = defaultdict(list)
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)

        word_level_degree = [[4] * len(tokens) for _ in range(len(tokens))]

        for i in range(len(tokens)):
            node_set = set()
            word_level_degree[i][i] = 0
            node_set.add(i)

            for j in tmp_dict[i]:
                if j not in node_set:
                    word_level_degree[i][j] = 1
                    node_set.add(j)

                for k in tmp_dict[j]:
                    if k not in node_set:
                        word_level_degree[i][k] = 2
                        node_set.add(k)

                        for g in tmp_dict[k]:
                            if g not in node_set:
                                word_level_degree[i][g] = 3
                                node_set.add(g)

        for i in range(len(tokens)):
            start, end = token_range[i][0], token_range[i][1]
            for j in range(len(tokens)):
                s, e = token_range[j][0], token_range[j][1]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        ret[row][col] = synpost_vocab.stoi.get(
                            word_level_degree[i][j],
                            synpost_vocab.unk_index)
        return ret


# 加载实例，实例中包括了四种类型的句子特征的矩阵 + biaffine attention用到的相关数据
def load_data_instances(
    sentence_packs: List[JSON],
    post_vocab: VocabHelp,
    deprel_vocab: VocabHelp,
    postag_vocab: VocabHelp,
    synpost_vocab: VocabHelp,
    args: Arguments,
) -> List[Instance]:
    instances = list()
    # 加载分词器
    # TODO: `args` consistency issue
    # (1/3) args应当是argparse解析出来的输入参数，但args.tokenizer是个BertTokenizer对象，这是不应该的。
    # (2/3) 出现这个情况是因为某处存在 `args.tokenizer = ...`，让args成了帮忙承载BertTokenizer对象的工具，
    # (3/3) 这违背了args的初衷，和单一职责原则。
    tokenizer: BertTokenizer = args.tokenizer

    # 处理每一个句子，将处理之后的封装之后的句子对象放到列表中
    for sentence_pack in sentence_packs:
        instances.append(
            Instance(
                tokenizer=tokenizer,
                sentence_pack=sentence_pack,
                post_vocab=post_vocab,
                deprel_vocab=deprel_vocab,
                postag_vocab=postag_vocab,
                synpost_vocab=synpost_vocab,
                args=args
            )
        )
    return instances


class Batch(NamedTuple):
    """
    类型，类型，类型！这才对嘛
    原先什么量都没有类型，这人怎么敢用的，这也能上ACL 2022
    """
    sentence_ids: List[str]
    sentences: List[str]
    bert_tokens: Tensor
    lengths: Tensor
    masks: Tensor
    sens_lens: List[int]
    token_ranges: List[TokenRange]
    aspect_tags: Tensor
    tags: Tensor
    word_pair_position: Tensor
    word_pair_deprel: Tensor
    word_pair_pos: Tensor
    word_pair_synpost: Tensor
    tags_symmetry: Tensor


class DataIterator(object):
    def __init__(self, instances: List[Instance], args):
        self.instances = instances
        self.args = args

        # math.ceil()向上取整
        self.batch_count = math.ceil(len(instances) / args.batch_size)

    # 这个函数什么时候调用的呢？？？在main.py第203行调用的
    def get_batch(self, index: int) -> Batch:
        # 因为清晰标注了类型，这里更容易看出，get_batch 进行了从标量到张量的转换。
        sentence_ids: List[str] = []
        sentences: List[str] = []
        sens_lens: List[int] = []
        token_ranges: List[List[List[int]]] = []
        bert_tokens: List[Tensor] = []
        lengths: List[int] = []
        masks: List[Tensor] = []
        aspect_tags: List[Tensor] = []
        opinion_tags: List[Tensor] = []
        tags: List[Tensor] = []
        tags_symmetry: List[Tensor] = []
        word_pair_position: List[Tensor] = []
        word_pair_deprel: List[Tensor] = []
        word_pair_pos: List[Tensor] = []
        word_pair_synpost: List[Tensor] = []

        # 获取一个batchsize大小的数据,相同属性的值放到一个列表中
        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            sentence_ids.append(self.instances[i].id)
            sentences.append(self.instances[i].sentence)
            sens_lens.append(self.instances[i].sen_length)
            token_ranges.append(self.instances[i].token_range)
            bert_tokens.append(self.instances[i].bert_tokens_padding)
            lengths.append(self.instances[i].length)
            masks.append(self.instances[i].mask)
            aspect_tags.append(self.instances[i].aspect_tags)
            opinion_tags.append(self.instances[i].opinion_tags)
            tags.append(self.instances[i].tags)
            tags_symmetry.append(self.instances[i].tags_symmetry)

            word_pair_position.append(self.instances[i].word_pair_position)
            word_pair_deprel.append(self.instances[i].word_pair_deprel)
            word_pair_pos.append(self.instances[i].word_pair_pos)
            word_pair_synpost.append(self.instances[i].word_pair_synpost)

        # 每个batch作为一个统一的输入
        bert_tokens_ret: Tensor = torch.stack(bert_tokens).to(self.args.device)
        lengths_ret = torch.tensor(lengths).to(self.args.device)
        masks_ret = torch.stack(masks).to(self.args.device)
        aspect_tags_ret = torch.stack(aspect_tags).to(self.args.device)

        # TODO: unused
        # opinion_tags_ret = torch.stack(opinion_tags).to(self.args.device)

        tags_ret = torch.stack(tags).to(self.args.device)
        tags_symmetry_ret = torch.stack(tags_symmetry).to(self.args.device)

        word_pair_position_ret = torch.stack(word_pair_position).to(self.args.device)
        word_pair_deprel_ret = torch.stack(word_pair_deprel).to(self.args.device)
        word_pair_pos_ret = torch.stack(word_pair_pos).to(self.args.device)
        word_pair_synpost_ret = torch.stack(word_pair_synpost).to(self.args.device)

        return Batch(
            sentence_ids,
            sentences,
            bert_tokens_ret,
            lengths_ret,
            masks_ret,
            sens_lens,
            token_ranges,
            aspect_tags_ret,
            tags_ret,
            word_pair_position_ret,
            word_pair_deprel_ret,
            word_pair_pos_ret,
            word_pair_synpost_ret,
            tags_symmetry_ret
        )
