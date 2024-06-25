"""For molecular graph processing."""
import os
import sys
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import rdkit.Chem as Chem
import torch
from rdkit.Chem import Descriptors
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from merging_operation_learning import merge_nodes
from model.mydataclass import batch_train_data, mol_train_data, train_data
from model.utils import (fragment2smiles, get_conn_list, graph2smiles,
                         networkx2data, smiles2mol)
from model.vocab import MotifVocab, SubMotifVocab, Vocab

RDContribDir = os.path.join(os.environ['CONDA_PREFIX'], 'share', 'RDKit', 'Contrib')
sys.path.append(os.path.join(RDContribDir, 'SA_Score'))

from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer
import sascorer

sys.path.pop()

ATOM_SYMBOL_VOCAB = Vocab(['*', 'N', 'O', 'Se', 'Cl', 'S', 'C', 'I', 'B', 'Br', 'P', 'Si', 'F'])
ATOM_ISAROMATIC_VOCAB = Vocab([True, False]) # 是否芳香
ATOM_FORMALCHARGE_VOCAB = Vocab(["*", -1, 0, 1, 2, 3]) #价态
ATOM_NUMEXPLICITHS_VOCAB = Vocab(["*", 0, 1, 2, 3]) # Atom 绑定到的显式 H 的数量
ATOM_NUMIMPLICITHS_VOCAB = Vocab(["*", 0, 1, 2, 3]) # Atom 绑定到的隐式 H 的数量
ATOM_FEATURES = [ATOM_SYMBOL_VOCAB, ATOM_ISAROMATIC_VOCAB, ATOM_FORMALCHARGE_VOCAB, ATOM_NUMEXPLICITHS_VOCAB, ATOM_NUMIMPLICITHS_VOCAB] # 原子特征
BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC] # 共价键
BOND_VOCAB = Vocab(BOND_LIST)

LOGP_MEAN, LOGP_VAR = 3.481587226600002, 1.8185146774225027 # LogP属性
MOLWT_MEAN, MOLWT_VAR = 396.7136355500001, 110.55283206754517 # MOLWT属性
QED_MEAN, QED_VAR = 0.5533041888502863, 0.21397359224960685 # QED属性
SA_MEAN, SA_VAR = 2.8882909807901354, 0.8059540682960904 # SA属性

class MolGraph(object):

    @classmethod
    # 为operation构建对象
    def load_operations(cls, operation_path: str, num_operations: int=500):
        MolGraph.NUM_OPERATIONS = num_operations
        MolGraph.OPERATIONS = [code.strip('\r\n') for code in open(operation_path)]
        MolGraph.OPERATIONS = MolGraph.OPERATIONS[:num_operations]
    
    @classmethod
    # load_vocab类方法——在MolGraph创建实例之前，就为该类创建通用的MOTIF_VOCAB和MOTIF_LIST对象
    def load_vocab(cls, vocab_path: str):
        pair_list = [line.strip("\r\n").split() for line in open(vocab_path)] # Motif词汇对（无连接点信息的smiles,有连接点信息的smiles）列表 Motif词汇对（无连接点信息的smiles,有连接点信息的smiles）列表
        '''
        合并两类
        '''
        MolGraph.MOTIF_VOCAB = MotifVocab(pair_list)
        MolGraph.MOTIF_LIST = MolGraph.MOTIF_VOCAB.motif_smiles_list # self.motif_smiles_list = [motif for _, motif in pair_list] 单纯的【有连接点信息的motif的smiles】列表

    def __init__(self,
        smiles: str,
        tokenizer: str="graph",
        methods: str="frequency_based"
    ):  
        assert tokenizer in ["graph", "motif"], \
            "The variable `process_level` should be 'graph' or 'motif'. "
        assert methods ["frequency_based", "connectivity_based"]
        "The variable `process_level` should be 'merging_based' or 'connectivity_based'. "
        self.smiles = smiles
        self.mol = smiles2mol(smiles, sanitize=True)
        self.mol_graph = self.get_mol_graph()
        self.init_mol_graph = self.mol_graph.copy()
        
        if tokenizer == "motif":
            if methods == "frequency_based":
                self.merging_graph = self.get_merging_graph_by_frequency()
                self.refragment()
                self.motifs = self.get_motifs()
            if methods == "connectivity_based":
                self.merging_graph = self.get_merging_graph_by_connectivity()
                self.refragment()
                self.motifs = self.get_motifs()

    def get_mol_graph(self) -> nx.Graph:
        graph = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))
        for atom in self.mol.GetAtoms():
            graph.nodes[atom.GetIdx()]['smarts'] = atom.GetSmarts()
            graph.nodes[atom.GetIdx()]['atom_indices'] = set([atom.GetIdx()])
            graph.nodes[atom.GetIdx()]['label'] = MolGraph.get_atom_features(atom)

        for bond in self.mol.GetBonds():
            atom1 = bond.GetBeginAtom().GetIdx()
            atom2 = bond.GetEndAtom().GetIdx()
            graph[atom1][atom2]['bondtype'] = bond.GetBondType()
            graph[atom1][atom2]['label'] = BOND_VOCAB[bond.GetBondType()]

        return graph

# Get motif
    # get merging_graph by frequency
    def get_merging_graph_by_frequency(self) -> nx.Graph: # 不改变mol_graph，主要更新了merging_graph的atom_indices信息
        mol = self.mol
        mol_graph = self.mol_graph.copy()
        merging_graph = mol_graph.copy()
        for code in self.OPERATIONS:
            for (node1, node2) in mol_graph.edges:
                if not merging_graph.has_edge(node1, node2):
                    continue
                # 合并，等价于对atom_indices集合求交集
                atom_indices = merging_graph.nodes[node1]['atom_indices'].union(merging_graph.nodes[node2]['atom_indices'])
                pattern = Chem.MolFragmentToSmiles(mol, tuple(atom_indices))
                if pattern == code:
                    merge_nodes(merging_graph, node1, node2)
            mol_graph = merging_graph.copy()
        return nx.convert_node_labels_to_integers(merging_graph)

    def get_merging_graph_by_connectivity(self) -> nx.Graph: # 不改变mol_graph，主要更新了merging_graph的atom_indices信息
        """
        依据Mol对象的连通性来构建merging_graph的节点及其['atom_indices']信息
        """
        mol = self.mol
        mol_graph = self.mol_graph.copy()
        merging_graph = mol_graph.copy()

        # 第一步：确定要断开的化学键
        # 这一步操作会找到所有应该断开的化学键的ID，这些键不在环中，但与环相连
        ids_of_bonds_to_cut = []
        for bond in mol.GetBonds():
            if bond.IsInRing():
                continue
            atom_begin = bond.GetBeginAtom()
            atom_end = bond.GetEndAtom()

            # 叶节点也分割
            # if min(atom_begin.GetDegree(), atom_end.GetDegree()) == 1:
            #     continue

            if not atom_begin.IsInRing() and not atom_end.IsInRing():
                continue
            ids_of_bonds_to_cut.append(bond.GetIdx())

        # 第二步：依据找到的化学键进行分割
        # 这一步操作将依据第一步找到的化学键，将分子进行分割，并得到分割后的新分子结构
        if ids_of_bonds_to_cut:
            fragmented_molecule = Chem.FragmentOnBonds(mol, ids_of_bonds_to_cut, addDummies=False)
        else:
            fragmented_molecule = mol

        # 第三步：获取分子碎片
        # 这一步操作将依据分割后的分子结构，得到各个碎片的信息
        frags = Chem.GetMolFrags(fragmented_molecule, asMols=False, sanitizeFrags=False)

        # 第四步：创建一个从原子索引到碎片ID的映射
        # 这一步操作为每个原子分配一个对应的碎片ID，方便后续合并操作
        atom_to_frag = {}
        for frag_id, frag in enumerate(frags):
            for atom_idx in frag:
                atom_to_frag[atom_idx] = frag_id

        # 第五步：依据碎片映射在merging_graph中合并节点
        # 这一步操作会依据第四步创建的映射，将属于同一个碎片的节点在merging_graph中合并
        for node1, node2 in mol_graph.edges:
            if node1 in atom_to_frag and node2 in atom_to_frag:
                if atom_to_frag[node1] == atom_to_frag[node2]:
                    if node1 not in merging_graph or node2 not in merging_graph:
                        continue
                    neighbors = [n for n in merging_graph.neighbors(node2)]
                    atom_indices = merging_graph.nodes[node1]["atom_indices"].union(
                        merging_graph.nodes[node2]["atom_indices"])
                    for n in neighbors:
                        if node1 != n and not merging_graph.has_edge(node1, n):
                            merging_graph.add_edge(node1, n)
                        merging_graph.remove_edge(node2, n)
                    merging_graph.remove_node(node2)
                    merging_graph.nodes[node1]["atom_indices"] = atom_indices

        # 最后一步：返回节点标签转换为整数的merging_graph
        return nx.convert_node_labels_to_integers(merging_graph)

    # 按照get_merging_graph_by_函数的结果分割分子，构建motif相关信息
    def refragment(self) -> None:
        mol_graph = self.mol_graph.copy()
        merging_graph = self.merging_graph

        for node in merging_graph.nodes:
            atom_indices = self.merging_graph.nodes[node]['atom_indices']
            # 不含断开点信息的片段smiles
            merging_graph.nodes[node]['motif_no_conn'] = fragment2smiles(self.mol, atom_indices)
            for atom_idx in atom_indices:
                mol_graph.nodes[atom_idx]['bpe_node'] = node

        # 断开分子，构造motif
        for node1, node2 in self.mol_graph.edges:
            bpe_node1, bpe_node2 = mol_graph.nodes[node1]['bpe_node'], mol_graph.nodes[node2]['bpe_node']
            if bpe_node1 != bpe_node2:
                # 创建motif1的断开点1
                conn1 = len(mol_graph)
                mol_graph.add_node(conn1)
                mol_graph.add_edge(node1, conn1)

                # 创建motif2的断开点2
                conn2 = len(mol_graph)
                mol_graph.add_node(conn2)
                mol_graph.add_edge(node2, conn2)

                # 为断开点1构造属性
                # 断开点1在mol_graph中的属性
                mol_graph.nodes[conn1]['smarts'] = '*' # 键的断开点
                mol_graph.nodes[conn1]['targ_atom'] = node2 # 断开点的原始配对点
                mol_graph.nodes[conn1]['merge_targ'] = conn2 # 断开点的合并配对点
                mol_graph.nodes[conn1]['anchor'] = node1 # 断开点所在片段的锚点
                mol_graph.nodes[conn1]['bpe_node'] = bpe_node1 # 断开点所在片段的基点
                mol_graph[node1][conn1]['bondtype'] = bondtype = mol_graph[node1][node2]['bondtype'] # 断开点与其锚点的键类型
                mol_graph[node1][conn1]['label'] = mol_graph[node1][node2]['label'] # 断开点与其锚点的键特征向量
                # 在为断开点1添加特征向量前，将断开点1添加到merging_graph其所在基点中的合并索引集中
                merging_graph.nodes[bpe_node1]['atom_indices'].add(conn1)
                # 为断开点1添加特征向量
                mol_graph.nodes[conn1]['label'] = MolGraph.get_atom_features(IsConn=True, BondType=bondtype)

                # 为断开点2构造属性
                # 断开点2在mol_graph中的属性
                mol_graph.nodes[conn2]['smarts'] = '*'
                mol_graph.nodes[conn2]['targ_atom'] = node1
                mol_graph.nodes[conn2]['merge_targ'] = conn1
                mol_graph.nodes[conn2]['anchor'] = node2
                mol_graph.nodes[conn2]['bpe_node'] = bpe_node2
                mol_graph[node2][conn2]['bondtype'] = bondtype = mol_graph[node1][node2]['bondtype']
                mol_graph[node2][conn2]['label'] = mol_graph[node1][node2]['label']
                # 在为断开点2添加特征向量前，将断开点1添加到merging_graph其所在基点中的合并索引集中
                merging_graph.nodes[bpe_node2]['atom_indices'].add(conn2)
                # 为断开点2添加特征向量
                mol_graph.nodes[conn2]['label'] = MolGraph.get_atom_features(IsConn=True, BondType=bondtype)

        for node in merging_graph.nodes:
            atom_indices = merging_graph.nodes[node]['atom_indices']
            # 在mol_graph中获取对应的子图
            motif_graph = mol_graph.subgraph(atom_indices)
            # 依据merging_graph生成motif：给merging_graph每个节点（即一个片段）赋motif值
            merging_graph.nodes[node]['motif'] = graph2smiles(motif_graph)

        self.mol_graph = mol_graph

    def get_motifs(self) -> Set[str]:
        return [(self.merging_graph.nodes[node]['motif_no_conn'], self.merging_graph.nodes[node]['motif']) for node in self.merging_graph.nodes]


# Get data
    def relabel(self):
        mol_graph = self.mol_graph # 会修改mol_graph
        bpe_graph = self.merging_graph # 会修改merging_graph

        # 对每一个motif（一个motif对应一个bpe_graph的节点）
        for node in bpe_graph.nodes:
            bpe_graph.nodes[node]['internal_edges'] = []
            atom_indices = bpe_graph.nodes[node]['atom_indices']

            fragment_graph = mol_graph.subgraph(atom_indices)
            motif_smiles_with_idx = graph2smiles(fragment_graph, with_idx=True) # graph2smiles将motif转为siles的过程中，对连接位点都打了同位素标记
            motif_with_idx = smiles2mol(motif_smiles_with_idx)
            # 获取【一个motif】的连接位点原子的【同位素标识列表】，及该列表到【原子排序列表】的映射
            conn_list, ordermap = get_conn_list(motif_with_idx, use_Isotope=True) # 由于graph2smiles中打了同位素标记，所以这里使用同位素标记，另外不考虑对称性

            bpe_graph.nodes[node]['conn_list'] = conn_list
            bpe_graph.nodes[node]['ordermap'] = ordermap
            bpe_graph.nodes[node]['label'] = MolGraph.MOTIF_VOCAB[ bpe_graph.nodes[node]['motif'] ] # getitem魔术方法
            bpe_graph.nodes[node]['num_atoms'] = len(atom_indices)

        for node1, node2 in bpe_graph.edges:
            self.merging_graph[node1][node2]['label'] = 0

        edge_dict = {}
        for edge, (node1, node2, attr) in enumerate(mol_graph.edges(data=True)):
            edge_dict[(node1, node2)] = edge_dict[(node2, node1)] = edge
            bpe_node1 = mol_graph.nodes[node1]['bpe_node']
            bpe_node2 = mol_graph.nodes[node2]['bpe_node']
            if bpe_node1 == bpe_node2:
                bpe_graph.nodes[bpe_node1]['internal_edges'].append(edge)

        for node, attr in mol_graph.nodes(data=True):
            if attr['smarts'] == '*':
                anchor = attr['anchor']
                targ_atom = attr['targ_atom']
                mol_graph.nodes[node]['edge_to_anchor'] = edge_dict[(node, anchor)]
                mol_graph.nodes[node]['merge_edge'] = edge_dict[(anchor, targ_atom)]

    def get_props(self) -> List[float]:
        mol = self.mol
        logP = (Descriptors.MolLogP(mol) - LOGP_MEAN) / LOGP_VAR
        Wt = (Descriptors.MolWt(mol) - MOLWT_MEAN) / MOLWT_VAR
        qed = (Descriptors.qed(mol) - QED_MEAN) / QED_VAR
        sa = (sascorer.calculateScore(mol) - SA_MEAN) / SA_VAR
        properties = [logP, Wt, qed, sa]
        return properties

### 构造训练数据，为GNN的嵌入与VAE的生成过程奠定了重要基础
    def get_data(self) -> mol_train_data:

        # 分子重新标注：把merging_graph引用为bpe_graph进行处理，重新标注了以下属性：
        # **节点属性：** ​`['internal_edges'] ：一条边的两个节点属于同一个 BPE 节点['conn_list']：连接列表`​和 ​`['ordermap']：排序映射`​→对分子中的原子进行规范化排序，依据【同位素标记】和【对称性】构建建排序映射​`['label']`​根据节点的 ​`motif`​ 属性，通过 ​`MotifVocab`​ 获取其标签，并设置为节点属性 ​`label`​**边属性：** ​`['label']`​对每条边，将边的标签 ​`label`​ 初始化为 0。
        self.relabel()

        # 生成这个分子整体的Data
        init_mol_graph, mol_graph, bpe_graph = self.init_mol_graph, self.mol_graph, self.merging_graph
        init_mol_graph_data, _ = networkx2data(init_mol_graph) #不涉及merging_grpah：与挖掘方式无关

        # 数据核心对象：1.motif 2.连接位点
        # 初始化
        motifs_list, conn_list = [], []
        '''
        class train_data:
            graph: Data
            query_atom: int
            cyclize_cand: List[int]
            label: Tuple[int, int]
        '''
        train_data_list: List[train_data] = []

        # 选择起始节点（起始motif）———分子中最大motif
        nodes_num_atoms = dict(bpe_graph.nodes(data='num_atoms'))
        node = max(nodes_num_atoms, key=nodes_num_atoms.__getitem__)
        start_label = bpe_graph.nodes[node]['label'] # 获取起始motif的标签（在motif词典里，唯一标识该motif词汇之smiles字符串的整数代号值）
        motifs_list.append(start_label) # 将起始motif加入motif_list
# 起始motif的连接位点列表
        conn_list.extend(self.merging_graph.nodes[node]['conn_list']) # 将起始motif的连接位点列表加入conn_list——relabel中通过get_conn_list函数建立了【该motif连接位点的原子标识列表】

# 获取起始motif的图表示形式
        subgraph = nx.Graph()
        subgraph = nx.union(subgraph, mol_graph.subgraph(bpe_graph.nodes[node]['atom_indices']))

        # 通过遍历连接位点，处理连接位点信息的同时，也可以处理连接位点所在的motif，但是一个motif可能对应多个连接位点，所以若某几个连接位点对应同一个motif，在处理这几个不同连接位点时，会反复对同一个motif进行处理，造成冗余
        # 每一个query_atom(是一个连接位点)对应一条训练数据
        while len(conn_list) > 0:
# 每一轮取一个连接位点
            # 获取连接位点列表中的第一个连接位点及其对应的断开目标点，即：一对断开点
            query_atom = conn_list[0]
            targ = mol_graph.nodes[query_atom]['merge_targ'] # 断开操作时创建的断开目标点属性（一般在另一个的motif上，也就是说一般不在conn_list里）
# 为当前【查询连接位点】对应的子图构造DATA、获取【当前子图所有原子的编号（偏移量）】
            # 第一次循环中，为【起始的motif子图】构造data；后续的循环中，为【逐渐连接起来的motif合成子图】构造data
            # 分子中的每一个连接位点对应一个data：连接位点的次序? 按照分子排名的motif分子片段里的排序后原子列表
            subgraph_data, mapping = networkx2data(subgraph) # ----【【mapping：给定具体的原子对象，返回该原子对象在【该子图中】的顺序编号】】-------
# 为当前【查询连接位点】构造【训练数据】【 标志：【查询连接位点】—— 特征：【当前子图的数据对象】—— 标签：【cyclize_cand】+【连接目标位置】 】
            # 如果另一个断开点和本连接位点都在conn_list里，说明这一对断开点在同一个motif上，断开了这个motif的一个环
            if targ in conn_list:
                cur_mol_smiles_with_idx = graph2smiles(subgraph, with_idx=True) # 调用了graph2smiles标记了连接位点的同位素
                motif_with_idx = smiles2mol(cur_mol_smiles_with_idx)
                _, ordermap = get_conn_list(motif_with_idx, use_Isotope=True) # ----【ordermap指示【该子图上】的【连接位点的同位素标记与其【原子等级次序】的对应关系】】----

# 目标连接点和其他点都是成环候选点，目标连接点是第一个成环候选点
                cyc_cand = [mapping[targ]] # 另一个连接位点【在该子图中的序号】【【优先】】放入可以成环的连接位点候选点
                for cand in conn_list[1:]:# 依次放入子图中的【其他（除了目标连接位点和查询连接位点以外的）连接位点】
                    if ordermap[cand] != ordermap[targ]: # 如果 cand 和 targ 的排序索引不同，表示它们不是同一个连接位点
                        cyc_cand.append(mapping[cand])
                # 成环类型训练数据：1，成环候选点【以成环目标点为[0]号排序】 2. 连接目标：连在【本motif（-1）】的【成环目标点——[0]号成环候选点】
                train_data_list.append(train_data(
                    graph = subgraph_data,
                    query_atom = mapping[query_atom], # 一个查询点query_atom（连接查询点）
                    cyclize_cand = cyc_cand, #【成环候选点】
                    label = (-1, 0), # 环类型，连在自己这个motif上成环
                ))
            # 如果另一个断开点不在本连接位点所在的conn_list里，说明断开目标点在另一个motif上
            else:
                # 找到断开目标点所在的motif
                node = mol_graph.nodes[targ]['bpe_node']
                # 获取这个新出现的motif的标签
                motif_idx = bpe_graph.nodes[node]['label']
                # 将这个新出现的motif加入motifs_list
                motifs_list.append(motif_idx) # 将起始motif之后的新的motif加入motif_list

                # 断开点的目标编号
                ordermap = bpe_graph.nodes[node]['ordermap']
                conn_idx = ordermap[targ]

# 除了查询连接位点以外，都可以是成环候选点，没有顺序
                cyc_cand = [mapping[cand] for cand in conn_list[1:]]

                # Motif类型训练数据：1，成环候选点【正常排序】 2. 连接目标：连在哪个motif上，连在motif的哪个连接位点上
                train_data_list.append(train_data( # 【“查询连接步骤”的训练数据】
                    graph = subgraph_data, # 【“查询连接步骤”的【过程状态子图】的特征矩阵】
                    query_atom = mapping[query_atom], # 【查询连接位点的【过程状态子图】编号】
                    cyclize_cand = cyc_cand, #【成环候选点】列表：【过程状态子图】中除了以外查询点以外的所有连接位点
                    label = (motif_idx, conn_idx), # 【连接目标位置指示】：连在哪个motif上，连在motif的哪个连接位点上
                ))
# 更新conn_list和subgraph
# 将另一个新的motif的连接位点加入conn_list：node对应断开目标点所在的另一个motif
                conn_list.extend(bpe_graph.nodes[node]['conn_list'])
# 将另一个另一个新的motif的子图融入subgraph
                subgraph = nx.union(subgraph, mol_graph.subgraph(bpe_graph.nodes[node]['atom_indices']))

# 更新conn_list和subgraph，进入下一轮迭代
            # 在本轮循环处理的两个连接位点【的锚点】之间加边，建立连接：若本轮处理的是同一motif的环类型则成环，若本轮处理的是不同motif的双motif类型则连接两个motif子图
            anchor1 = mol_graph.nodes[query_atom]['anchor']
            anchor2 = mol_graph.nodes[targ]['anchor']
            subgraph.add_edge(anchor1, anchor2)
            subgraph[anchor1][anchor2]['bondtype'] = mol_graph[anchor1][anchor2]['bondtype']
            subgraph[anchor1][anchor2]['label'] = mol_graph[anchor1][anchor2]['label']
            # 连接锚点后，连接位点可以删去
            subgraph.remove_node(query_atom)
            subgraph.remove_node(targ)

            # 处理连接位点列表
            conn_list.remove(query_atom)
            conn_list.remove(targ)

        props = self.get_props()
        motifs_list = list(set(motifs_list)) # 一个motif上可能存在多个连接位点，所以按照连接位点生成训练数据时，可能会有冗余
        return mol_train_data(
            mol_graph = init_mol_graph_data,
            props = props,
            start_label = start_label,
            train_data_list = train_data_list, # 连接位点对应的训练数据列表，【体现了从散落的motif，逐渐连接出一个完整分子的过程】
            motif_list = motifs_list,
        )


    @staticmethod
### 引入词汇表
    def preprocess_vocab() -> Batch:
        vocab_data = []
        for idx in tqdm(range(len(MolGraph.MOTIF_LIST))): # 遍历单纯的【有连接点信息的motif的smiles】列表
            graph, _, _ = MolGraph.motif_to_graph(MolGraph.MOTIF_LIST[idx])  # 针对motif的smiles，返回【更新了属性的Graph对象】(不考虑【按原子rank排序的连接位点列表】、【各个原子的rank列表】）
            '''
            由于data类型需要构造：1.【x: 所有原子标签(5维特征向量)构成的张量】；2. 【edge_attr：所有边标签(1维：[化学键类型]特征向量)构成的张量】
            所以motif_to_graph中专门更新了node和edge的['label'] 
            '''
            data, _ = networkx2data(graph) # 针对motif的graph，转换为data
            vocab_data.append(data)
        vocab_data = Batch.from_data_list(vocab_data)
        return vocab_data


    @staticmethod
    def get_atom_features(atom: Chem.rdchem.Atom=None, IsConn: bool=False, BondType: Chem.rdchem.BondType=None) -> Tuple[int, int, int, int, int]:
        # 连接位点的特征向量设定
        if IsConn:
            # 除了芳香性通过判断断键之外，其他特征都标注为0
            Symbol, FormalCharge, NumExplicitHs, NumImplicitHs = 0, 0, 0, 0       
            IsAromatic = True if BondType == Chem.rdchem.BondType.AROMATIC else False
            IsAromatic = ATOM_ISAROMATIC_VOCAB[IsAromatic]
        # 普通原子的特征向量设定
        else:
            Symbol = ATOM_SYMBOL_VOCAB[atom.GetSymbol()]
            IsAromatic = ATOM_ISAROMATIC_VOCAB[atom.GetIsAromatic()]
            FormalCharge = ATOM_FORMALCHARGE_VOCAB[atom.GetFormalCharge()]
            NumExplicitHs = ATOM_NUMEXPLICITHS_VOCAB[atom.GetNumExplicitHs()]
            NumImplicitHs = ATOM_NUMIMPLICITHS_VOCAB[atom.GetNumImplicitHs()]
        return (Symbol, IsAromatic, FormalCharge, NumExplicitHs, NumImplicitHs)


    @staticmethod
    def motif_to_graph(smiles: str, motif_list: Optional[List[str]] = None) -> Tuple[nx.Graph, List[int], List[int]]:
        # 将有连接点信息的motif的smiles转换为图，设定相关属性
        motif = smiles2mol(smiles) # 有连接点信息的motif的smiles转换为mol对象
        graph = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(motif)) # 获取图的邻接矩阵
        
        dummy_list = []
        for atom in motif.GetAtoms(): # 遍历每一个原子 【图节点】
            idx = atom.GetIdx()
            graph.nodes[idx]['smarts'] = atom.GetSmarts() # 设定原子【图节点】的【smarts表示】属性
            graph.nodes[idx]['motif'] = smiles  # 设定原子【图节点】的【所在motif】属性为——该motif有连接点信息的smiles
            # 如果原子【图节点】是dummy atom 【伪原子】——也就是【连接位点】，除了获取原子的特征向量【(0, IsAromatic, 0, 0, 0)】之外，还专门获取断键类型以构建dummy_bond_type属性，同时将编号加入dummy_list
            if atom.GetSymbol() == '*':
                graph.nodes[idx]['dummy_bond_type'] = bondtype = atom.GetBonds()[0].GetBondType()
                graph.nodes[atom.GetIdx()]['label'] = MolGraph.get_atom_features(IsConn=True, BondType=bondtype) #
                dummy_list.append(idx)
            # 如果原子是普通原子，只获取原子的特征向量【(Symbol, IsAromatic, FormalCharge, NumExplicitHs, NumImplicitHs)】
            else:
                graph.nodes[atom.GetIdx()]['label'] = MolGraph.get_atom_features(atom)

        # 【按照原子rank对连接位点列表重新排序】——数据集中的分子在Relabel中调用get_conn_list完成此步骤，motif则在motif_to_graph中完成此步骤
        ranks = list(Chem.CanonicalRankAtoms(motif, includeIsotopes=False))
        dummy_list = list(zip(dummy_list, [ranks[atom.GetIdx()] for atom in motif.GetAtoms() if atom.GetSymbol() == '*']))
        if len(dummy_list) > 0:
            dummy_list.sort(key=lambda x: x[1]) # 按照【原子rank】重新组织列表顺序
            dummy_list, _ = zip(*dummy_list)  # 解包，只获取【rank序的连接位点编号列表】

        # 标记化学键类型
        for bond in motif.GetBonds():
            atom1 = bond.GetBeginAtom().GetIdx()
            atom2 = bond.GetEndAtom().GetIdx()
            graph[atom1][atom2]['bondtype'] = bond.GetBondType()
            graph[atom1][atom2]['label'] = BOND_VOCAB[bond.GetBondType()]

        # 返回【更新了属性的Graph对象】、【按原子rank排序的连接位点列表】、【各个原子的rank列表】
        return graph, list(dummy_list), ranks