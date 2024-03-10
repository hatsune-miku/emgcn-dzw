import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as nnf
from torch import Tensor
from transformers import BertModel, BertTokenizer
from arguments import Arguments
from typing import List, Tuple


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""
    a_2: nn.Parameter
    b_2: nn.Parameter

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class RefiningStrategy(nn.Module):
    def __init__(self, hidden_dim, edge_dim, dim_e, dropout_ratio=0.5):
        super(RefiningStrategy, self).__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.dim_e = dim_e
        self.dropout = dropout_ratio
        self.W = nn.Linear(self.hidden_dim * 2 + self.edge_dim * 3, self.dim_e)

    def forward(self, edge, node1, node2):
        batch, seq, seq, edge_dim = edge.shape
        node = torch.cat([node1, node2], dim=-1)

        edge_diag = torch.diagonal(edge, offset=0, dim1=1, dim2=2).permute(0, 2, 1).contiguous()
        edge_i = edge_diag.unsqueeze(1).expand(batch, seq, seq, edge_dim)
        edge_j = edge_i.permute(0, 2, 1, 3).contiguous()
        edge = self.W(torch.cat([edge, edge_i, edge_j, node], dim=-1))

        return edge


class GATLayer(nn.Module):
    """ A GAT module operated on dependency graphs. """

    def __init__(self, device, input_dim, edge_dim, dep_embed_dim, num_heads):
        super(GATLayer, self).__init__()
        self.hidden_dim = input_dim // num_heads
        self.edge_dim = edge_dim
        self.input_dim = input_dim
        self.dep_embed_dim = dep_embed_dim
        self.device = device
        self.layernorm = LayerNorm(self.hidden_dim * num_heads)
        self.W = nn.Linear(self.hidden_dim * num_heads, self.hidden_dim * num_heads)
        self.highway = RefiningStrategy(
            self.hidden_dim * num_heads,
            self.edge_dim,
            self.dep_embed_dim,
            dropout_ratio=0.5)
        self.linear = nn.Linear(input_dim, self.hidden_dim * num_heads)
        self.fc_w1 = nn.Parameter(torch.empty(size=(1, 1, num_heads, self.hidden_dim)))
        nn.init.xavier_uniform_(self.fc_w1.data, gain=1.414)
        self.fc_w2 = nn.Parameter(torch.empty(size=(1, 1, num_heads, self.hidden_dim)))
        nn.init.xavier_uniform_(self.fc_w2.data, gain=1.414)
        self.num_heads = num_heads
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_heads),
        )

    def forward(self, weight_prob_softmax, weight_adj, gcn_inputs, self_loop):
        batch, seq, dim = gcn_inputs.shape
        weight_prob_softmax = self_loop + weight_prob_softmax

        # 哪个地方有边
        mask = weight_prob_softmax.sum(-1) == 0

        # 维度对齐
        feature = self.linear(gcn_inputs).reshape(batch, seq, self.num_heads, self.hidden_dim)

        attn_src = torch.sum(feature * self.fc_w1, dim=-1).permute(0, 2, 1).unsqueeze(-1)
        attn_dst = torch.sum(feature * self.fc_w2, dim=-1).permute(0, 2, 1).unsqueeze(-2)

        matrix_a = self.edge_mlp(weight_prob_softmax).permute(0, 3, 1, 2)
        attn = nnf.leaky_relu(attn_src + attn_dst + matrix_a)

        # 把没有边的地方设置为 负无穷
        attn = torch.masked_fill(attn, mask.unsqueeze(-3), float("-inf"))

        # TODO: 没有axis这个参数，需要检查
        attn = torch.softmax(attn, axis=-1)

        # 可以加 pooling
        gcn_outputs = torch.matmul(attn, feature.permute(0, 2, 1, 3))
        gcn_outputs = gcn_outputs.permute(0, 2, 1, 3).reshape(batch, seq, -1)
        gcn_outputs = self.W(gcn_outputs)
        gcn_outputs = self.layernorm(gcn_outputs)

        weights_gcn_outputs = nnf.relu(gcn_outputs)

        node_outputs = weights_gcn_outputs
        # TODO: unused
        # weight_prob_softmax = weight_prob_softmax.permute(0, 2, 3, 1).contiguous()
        node_outputs1 = node_outputs.unsqueeze(1).expand(batch, seq, seq, dim)
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()
        edge_outputs = self.highway(weight_adj, node_outputs1, node_outputs2)

        return node_outputs, edge_outputs


# TODO: unused
"""
class GraphConvLayer(nn.Module):
    ""A GCN module operated on dependency graphs. ""

    def __init__(self, device, gcn_dim, edge_dim, dep_embed_dim, pooling='avg'):
        super(GraphConvLayer, self).__init__()
        self.gcn_dim = gcn_dim
        self.edge_dim = edge_dim
        self.dep_embed_dim = dep_embed_dim
        self.device = device
        self.pooling = pooling
        self.layernorm = LayerNorm(self.gcn_dim)
        self.W = nn.Linear(self.gcn_dim, self.gcn_dim)
        self.highway = RefiningStrategy(gcn_dim, self.edge_dim, self.dep_embed_dim, dropout_ratio=0.5)

    def forward(self, weight_prob_softmax, weight_adj, gcn_inputs, self_loop):
        # [B, S, S, 50]
        # 16 x 102 x 300
        batch, seq, dim = gcn_inputs.shape

        # 16 x 102 x 102 x 50 -> 16 x 50 x 102 x 102
        weight_prob_softmax = weight_prob_softmax.permute(0, 3, 1, 2)

        # 16 x 102 x 300 -> 16 x 1 x 102 x 300 -> 16 x 50 x 102 x 300
        gcn_inputs = gcn_inputs.unsqueeze(1).expand(
            batch, self.edge_dim, seq, dim
        )

        weight_prob_softmax += self_loop

        # torch.matmul()是tensor乘法 16x102x102x50 X 16x50x102x300
        matrix_ax = torch.matmul(weight_prob_softmax, gcn_inputs)
        if self.pooling == 'avg':
            matrix_ax = matrix_ax.mean(dim=1)

        elif self.pooling == 'max':
            matrix_ax, _ = matrix_ax.max(dim=1)

        elif self.pooling == 'sum':
            matrix_ax = matrix_ax.sum(dim=1)

        # Ax: [batch, seq, dim] 16 x 102 x 300
        gcn_outputs = self.W(matrix_ax)
        gcn_outputs = self.layernorm(gcn_outputs)
        weights_gcn_outputs = nnf.relu(gcn_outputs)

        # 最终的经过图卷积神经网络的经过特征提取之后的词向量
        node_outputs = weights_gcn_outputs

        # 16 x 50 x 102 x 102 -> 16 x 102 x 102 x 50
        # TODO: unused
        # weight_prob_softmax = weight_prob_softmax.permute(0, 2, 3, 1).contiguous()

        # 16 x 102 x 300 -> 16 x 1 x 102 x 300 -> 16 x 102 x 102 x 300
        node_outputs1 = node_outputs.unsqueeze(1).expand(batch, seq, seq, dim)

        # 16 x 102 x 102 x 300 -> 16 x 102 x 102 x 300
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()

        # 这个就是节点和边关系进行拼接
        edge_outputs = self.highway(weight_adj, node_outputs1, node_outputs2)

        return node_outputs, edge_outputs
"""


class Biaffine(nn.Module):
    args: Arguments
    in1_features: int
    in2_features: int
    out_features: int
    bias: Tuple[bool, bool]

    def __init__(
        self,
        args: Arguments,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: Tuple[bool, bool] = (True, True)
    ):
        super(Biaffine, self).__init__()
        self.args: Arguments = args
        self.in1_features = in1_features  # 300
        self.in2_features = in2_features  # 300
        self.out_features = out_features  # 10
        self.bias = bias

        # 301
        self.linear_input_size = in1_features + int(bias[0])

        # 3010
        self.linear_output_size = out_features * (in2_features + int(bias[1]))

        # 301 -> 3010
        self.linear = torch.nn.Linear(
            in_features=self.linear_input_size,
            out_features=self.linear_output_size,
            bias=False
        )

    # input1和input2分别是300维的词向量a和词向量o
    def forward(
        self,
        input1: Tensor,
        input2: Tensor
    ) -> Tensor:
        # 16 x 句子1的token长度 x dim
        batch_size, len1, dim1 = input1.size()

        # 16 x 句子2的token长度 x dim
        # TODO: 前面的batch_size被忽略了，检查是否是特意这样做的
        batch_size, len2, dim2 = input2.size()

        if self.bias[0]:
            ones = torch.ones(batch_size, len1, 1).to(self.args.device)
            input1 = torch.cat((input1, ones), dim=2)  # dim变成301
            dim1 += 1
        if self.bias[1]:
            ones = torch.ones(batch_size, len2, 1).to(self.args.device)
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1

        # 线性变换301 -> 3010 : 16 x len1 x 3010
        affine = self.linear(input1)

        # 16 x len1*10 x 301
        affine = affine.view(batch_size, len1 * self.out_features, dim2)

        # 16 x len2 x 301 -> 16 x 301 x len2
        input2 = torch.transpose(input2, 1, 2)

        # 矩阵乘法 16 x len1*10 x len2
        biaffine = torch.bmm(affine, input2)

        # 16 x len2 x len1*10
        biaffine = torch.transpose(biaffine, 1, 2)

        # 16 x len2 x len1 x 10
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)

        return biaffine


# 定义 Multi-Head Attention 模块
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.query_fc = nn.Linear(input_dim, input_dim, bias=False)
        self.key_fc = nn.Linear(input_dim, input_dim, bias=False)
        self.value_fc = nn.Linear(input_dim, input_dim, bias=False)
        self.out_fc = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        # 将输入张量拆分为 num_heads 个头部，并进行线性变换
        query_heads = self.query_fc.forward(query) \
            .view(-1, self.num_heads, query.shape[-2], self.head_dim)

        key_heads = self.key_fc.forward(key) \
            .view(-1, self.num_heads, key.shape[-2], self.head_dim)

        value_heads = self.value_fc.forward(value) \
            .view(-1, self.num_heads, value.shape[-2], self.head_dim)

        # 将每个头部的 query, key 和 value 分别拆分，并计算得分矩阵
        attention_scores = torch.matmul(query_heads, key_heads.transpose(-2, -1)) / self.head_dim ** 0.5

        # 将得分矩阵进行 softmax 归一化，并根据 value_heads 进行加权求和
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        context_heads = torch.matmul(attention_probs, value_heads)
        context = context_heads.view(-1, self.output_dim)

        # 对上下文向量进行线性变换，并返回结果
        output = self.out_fc(context)
        return output


# 用pytorch建立的EMCGCN神经网络模型框架
# 原来这玩意儿就是model啊
class EMCGCN(torch.nn.Module):
    # 对一些模型、参数进行初始化
    def __init__(self, args: Arguments):
        # 固定写法，目的是向其父类中传递参数进行初始化其父类
        super(EMCGCN, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_model_path)  # 加载用到的bert模型
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)  # 加载分词器
        self.dropout_output = nn.Dropout(args.emb_dropout)

        # 这里并没有生成向量，而是初始化了一个对象，这个对象是干生成向量这个活的
        # [batch_size, max_len, 10]
        # 相对位置嵌入
        self.post_emb = torch.nn.Embedding(args.post_size, args.class_num, padding_idx=0)

        # 依赖关系嵌入
        self.deprel_emb = torch.nn.Embedding(args.deprel_size, args.class_num, padding_idx=0)

        # 词性标注嵌入
        self.postag_emb = torch.nn.Embedding(args.postag_size, args.class_num, padding_idx=0)

        # 基于依赖树的相对位置距离嵌入
        self.synpost_emb = torch.nn.Embedding(args.synpost_size, args.class_num, padding_idx=0)

        # 将情感矩阵升维
        self.sentic_dense = nn.Linear(1, 10)

        # 初试权重为 1
        self.sentic_dense.weight = nn.Parameter(torch.ones(10).reshape(10, 1))
        self.pm_dense = nn.Linear(1, 10)

        # 初试权重为 1，等同于复制
        self.pm_dense.weight = nn.Parameter(torch.ones(10).reshape(10, 1))

        # 初始化Biaffine Attention对象
        self.triplet_biaffine = Biaffine(
            args, args.gcn_dim, args.gcn_dim, args.class_num,
            bias=(True, True)
        )
        # MLPa, 768->300前向全链接层，
        # 输入bert_feature_dim的向量维度768，
        # 输出args.gcn_dim300维度
        self.ap_fc = nn.Linear(
            args.bert_feature_dim,
            args.gcn_dim
        )

        # MLPo, 768>300
        self.op_fc = nn.Linear(args.bert_feature_dim, args.gcn_dim)

        # 压缩线性变换
        self.dense = nn.Linear(args.bert_feature_dim, args.gcn_dim)

        # 这里的数值要和特征类型数量对应
        self.attention = MultiHeadAttention(input_dim=70, output_dim=70, num_heads=7)

        # 图卷积神经网络的层
        self.num_layers = args.num_layers
        self.gcn_layers = nn.ModuleList()

        # 层次归一化
        self.layernorm = LayerNorm(args.bert_feature_dim)

        for i in range(self.num_layers):
            self.gcn_layers.append(
                GATLayer(args.device, args.gcn_dim, 7 * args.class_num, args.class_num, num_heads=2))

    def forward(
        self,
        tokens: Tensor,
        masks: Tensor,
        sentic_matrixs: Tensor,
        perturbed_matrix: Tensor,
        word_pair_position: Tensor,
        word_pair_deprel: Tensor,
        word_pair_pos: Tensor,
        word_pair_synpost: Tensor
    ) -> List[Tensor]:
        bert_feature, _ = self.bert.forward(tokens, masks, return_dict=False)
        bert_feature = self.dropout_output.forward(bert_feature)

        # 16 x 102
        batch, seq = masks.shape

        # 16 x 102 -> 16 x 102 x 102 -> 16 x 102 x 102 x 1
        tensor_masks = masks.unsqueeze(1).expand(batch, seq, seq).unsqueeze(-1)

        # * multi-feature 16 x 102 x 102 x 10也就是说这个时候进行向量化
        word_pair_post_emb: Tensor = self.post_emb.forward(word_pair_position)
        word_pair_deprel_emb: Tensor = self.deprel_emb.forward(word_pair_deprel)
        word_pair_postag_emb: Tensor = self.postag_emb.forward(word_pair_pos)
        word_pair_synpost_emb: Tensor = self.synpost_emb.forward(word_pair_synpost)

        # BiAffine
        # 对应论文中的MLPa
        ap_node = nnf.relu(self.ap_fc.forward(bert_feature))

        # 对应论文中MLPo
        op_node = nnf.relu(self.op_fc.forward(bert_feature))

        # 16 x 102 x 102 x 10 gcn的边关系输入
        biaffine_edge: Tensor = self.triplet_biaffine.forward(ap_node, op_node)

        # Sentic
        sentic_matrixs = sentic_matrixs.unsqueeze(len(sentic_matrixs.shape))
        sentic_matrixs_emb: Tensor = self.sentic_dense.forward(sentic_matrixs)

        # Impact Matrix
        perturbed_matrix = perturbed_matrix.unsqueeze(len(perturbed_matrix.shape))
        perturbed_matrix_emb: Tensor = self.pm_dense.forward(perturbed_matrix)

        # 压缩，通过全链接网络变换一下维度 gcn的词向量输入
        gcn_input = nnf.relu(self.dense.forward(bert_feature))

        # 上一层的output是下一层的input
        gcn_outputs = gcn_input

        # 各种R
        weight_prob_list: List[Tensor] = [
            sentic_matrixs_emb,
            perturbed_matrix_emb,
            biaffine_edge,
            word_pair_post_emb,
            word_pair_deprel_emb,
            word_pair_postag_emb,
            word_pair_synpost_emb
        ]

        # 对类别分数进行归一化到0~1之间
        # TODO: 可是全都没用到啊
        # sentic_matrixs_emb_softmax = F.softmax(sentic_matrixs_emb, dim=-1) * tensor_masks
        # perturbed_matrix_emb_softmax = F.softmax(perturbed_matrix_emb, dim=-1) * tensor_masks
        # biaffine_edge_softmax = F.softmax(biaffine_edge, dim=-1) * tensor_masks
        # word_pair_post_emb_softmax = F.softmax(word_pair_post_emb, dim=-1) * tensor_masks
        # word_pair_deprel_emb_softmax = F.softmax(word_pair_deprel_emb, dim=-1) * tensor_masks
        # word_pair_postag_emb_softmax = F.softmax(word_pair_postag_emb, dim=-1) * tensor_masks
        # word_pair_synpost_emb_softmax = F.softmax(word_pair_synpost_emb, dim=-1) * tensor_masks

        self_loop: List[Tensor] = []
        for _ in range(batch):
            self_loop.append(torch.eye(seq))

        # torch.eye()生成对角钱全1， 其余部分为0的二维数组
        # batchsize = 16 -> 16个102 x 102 对角线为1 的多维矩阵
        # torch.stack()作用是将一个个二维数组进行拼接，形成一个三维矩阵。
        # .unsqueeze()函数作用是升维。
        # .expend()函数作用是：将张量广播到新形状16 x 5*10 x 102 x 102 。
        # .permute()函数的作用是重新排列，也就是重新建立形状，置换维度。
        # 16 x 102 x 102 -> 增加维度16 x 1 x 102 x 102 -> 16 x 5*10 x 102 x 102
        # tensor_masks: 16 x 102 x 102 x 1 -> 16 x 1 x 102 x 102 -> 经过contiguous()函数之后内存存储顺序也改变了
        # 最后self_loop是16 x 50 x 102 x 102, 且对多余部分进行了去除，都设为0
        self_loop = (
                        torch.stack(self_loop)
                        .to(self.args.device)
                        .unsqueeze(3)
                        .expand(batch, seq, seq, 7 * self.args.class_num)
                    ) * (
                        tensor_masks
                        .permute(0, 1, 2, 3)
                        .contiguous()
                    )
        # 拼接的是5个R 16 x 102 x 102 x 50
        weight_prob = torch.cat(
            [
                sentic_matrixs_emb,
                perturbed_matrix_emb,
                biaffine_edge,
                word_pair_post_emb,
                word_pair_deprel_emb,
                word_pair_postag_emb,
                word_pair_synpost_emb
            ],
            dim=-1
        )
        weight_prob = self.attention.forward(weight_prob, weight_prob, weight_prob)
        weight_prob = weight_prob.reshape(
            -1, self.args.max_sequence_len,
            self.args.max_sequence_len,
            7 * self.args.class_num
        )
        weight_prob_softmax = nnf.softmax(weight_prob, dim=-1) * tensor_masks

        # 图卷积神经网络
        for _layer in range(0, self.num_layers):
            # [batch, seq, dim]
            layer = self.gcn_layers[_layer]
            gcn_outputs, weight_prob = layer.forward(
                weight_prob_softmax,
                weight_prob,
                gcn_outputs,
                self_loop
            )
            weight_prob_list.append(weight_prob)

        return weight_prob_list
