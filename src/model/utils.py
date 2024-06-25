from typing import Dict, List, Tuple

import networkx as nx
import rdkit.Chem as Chem
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

def smiles2mol(smiles: str, sanitize: bool=False) -> Chem.rdchem.Mol:
    if sanitize:
        return Chem.MolFromSmiles(smiles)
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None
    AllChem.SanitizeMol(mol, sanitizeOps=0)
    return mol

def graph2smiles(fragment_graph: nx.Graph, with_idx: bool=False) -> str:
    motif = Chem.RWMol()
    node2idx = {}
    for node in fragment_graph.nodes:
        idx = motif.AddAtom(smarts2atom(fragment_graph.nodes[node]['smarts']))
        if with_idx and fragment_graph.nodes[node]['smarts'] == '*':
            motif.GetAtomWithIdx(idx).SetIsotope(node) # 同位素标记设定
        node2idx[node] = idx
    for node1, node2 in fragment_graph.edges:
        motif.AddBond(node2idx[node1], node2idx[node2], fragment_graph[node1][node2]['bondtype'])
    return Chem.MolToSmiles(motif, allBondsExplicit=True)

def networkx2data(G: nx.Graph) -> Tuple[Data, Dict[int, int]]:
    num_nodes = G.number_of_nodes()
    # 词典：以【原子节点对象（分子图的节点是原子）】为键，以【编号】为值 —— 给定具体的原子对象，返回该原子对象在图中的顺序编号
    mapping = dict(zip(G.nodes(), range(num_nodes)))

    # 将整数表示应用到原子标识上
    G = nx.relabel_nodes(G, mapping)

    # 转为有向图
    G = G.to_directed() if not nx.is_directed(G) else G

    # 所有边构成的张量数据
    edges = list(G.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 所有节点【标签（5维：[Symbol, IsAromatic, FormalCharge, NumExplicitHs, NumImplicitHs]）特征向量】构成的张量数据
    x = torch.tensor([i for _, i in G.nodes(data='label')])

    # 所有边【标签(1维：[化学键类型])特征向量】构成的张量数据
    edge_attr = torch.tensor([[i] for _, _, i in G.edges(data='label')], dtype=torch.long)

    # 构造data类
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data, mapping

def fragment2smiles(mol: Chem.rdchem.Mol, indices: List[int]) -> str:
    smiles = Chem.MolFragmentToSmiles(mol, tuple(indices))
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=False))

def smarts2atom(smarts: str) -> Chem.rdchem.Atom:
    return Chem.MolFromSmarts(smarts).GetAtomWithIdx(0)

def mol_graph2smiles(graph: nx.Graph, postprocessing: bool=True) -> str:
    mol = Chem.RWMol()
    graph = nx.convert_node_labels_to_integers(graph)
    node2idx = {}
    for node in graph.nodes:
        idx = mol.AddAtom(smarts2atom(graph.nodes[node]['smarts']))
        node2idx[node] = idx
    for node1, node2 in graph.edges:
        mol.AddBond(node2idx[node1], node2idx[node2], graph[node1][node2]['bondtype'])
    mol = mol.GetMol()
    smiles = Chem.MolToSmiles(mol)
    return postprocess(smiles) if postprocessing else smiles
 
def postprocess(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol)
    except:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic() and not atom.IsInRing():
                atom.SetIsAromatic(False)   
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                if not (bond.GetBeginAtom().GetIsAromatic() and bond.GetEndAtom().GetIsAromatic()):
                    bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        
        for _ in range(100):
            problems = Chem.DetectChemistryProblems(mol)
            flag = False
            for problem in problems:
                if problem.GetType() =='KekulizeException':
                    flag = True
                    for atom_idx in problem.GetAtomIndices():
                        mol.GetAtomWithIdx(atom_idx).SetIsAromatic(False)
                    for bond in mol.GetBonds():
                        if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                            if not (bond.GetBeginAtom().GetIsAromatic() and bond.GetEndAtom().GetIsAromatic()):
                                bond.SetBondType(Chem.rdchem.BondType.SINGLE)
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol), sanitize=False)
            if flag: continue
            else: break
        
        smi = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        try:
            Chem.SanitizeMol(mol)
        except:
            print(f"{smiles} not valid")
            return "CC"
        smi = Chem.MolToSmiles(mol)
        return smi

def get_conn_list(motif: Chem.rdchem.Mol, use_Isotope: bool=False, symm: bool=False) -> Tuple[List[int], Dict[int, int]]:# 本项目中：使用同位素标记，不考虑对称性

    ranks = list(Chem.CanonicalRankAtoms(motif, includeIsotopes=False, breakTies=False)) # 生成按照分子排名的【motif】排序后原子列表
    # 提取【连接位点原子】构造【【连接位点的表示：连接位点的原子rank】的映射】：本项目使用同位素进行标记：{连接位点原子在原始分子图中的同位素质量数【连接位点的表示】 : 连接位点原子在排序原子列表中的编号【连接位点的原子rank】}
    if use_Isotope: # ordermap【连接位点的表示：连接位点的原子rank】映射
        ordermap = {atom.GetIsotope(): ranks[atom.GetIdx()] for atom in motif.GetAtoms() if atom.GetSymbol() == '*'} # 假设连接位点都用同位素标示出来了
    else:
        ordermap = {atom.GetIdx(): ranks[atom.GetIdx()] for atom in motif.GetAtoms() if atom.GetSymbol() == '*'}
    if len(ordermap) == 0:
        return [], {}
    # --------------按照【原子rank】重新组织连接位点在ordermap中的排列顺序--------------
    ordermap = dict(sorted(ordermap.items(), key=lambda x: x[1]))

    # （依据是否考虑对称性，采取不同的）连接位点列表提取方法
    # 不考虑对称性，则连接位点信息唯一，直接将【连接位点原子在原始分子图中的同位素质量数】构建为【连接位点原子列表】
    if not symm:
        conn_atoms = list(ordermap.keys()) #
    # 考虑对称性，如果当前原子的排名不同于前一个原子的排名，则将其添加到 conn_atoms 列表中，即如果 order 相同，只保留一个
    else:
        cur_order, conn_atoms = -1, []
        for idx, order in ordermap.items():
            if order != cur_order:
                cur_order = order
                conn_atoms.append(idx)

    # 连接位点(同位素标记编号)列表的次序：按照CanonicalRankAtoms分子排名进行组织
    return conn_atoms, ordermap


def label_attachment(smiles: str) -> str:

    mol = Chem.MolFromSmiles(smiles)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    dummy_atoms = [(atom.GetIdx(), ranks[atom.GetIdx()])for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    dummy_atoms.sort(key=lambda x: x[1])
    orders = []
    for (idx, order) in dummy_atoms:
        if order not in orders:
            orders.append(order)
            mol.GetAtomWithIdx(idx).SetIsotope(len(orders))
    return Chem.MolToSmiles(mol)

def get_accuracy(scores: torch.Tensor, labels: torch.Tensor):
    _, preds = torch.max(scores, dim=-1)
    acc = torch.eq(preds, labels).float()

    number, indices = torch.topk(scores, k=10, dim=-1)
    topk_acc = torch.eq(indices, labels.view(-1,1)).float()
    return torch.sum(acc) / labels.nelement(), torch.sum(topk_acc) / labels.nelement()

def sample_from_distribution(distribution: torch.Tensor, greedy: bool=False, topk: int=0):
    if greedy or topk == 1:
        motif_indices = torch.argmax(distribution, dim=-1)
    elif topk == 0 or len(torch.where(distribution > 0)) <= topk:
        motif_indices = torch.multinomial(distribution, 1)
    else:
        _, topk_idx = torch.topk(distribution, topk, dim=-1)
        mask = torch.zeros_like(distribution)
        ones = torch.ones_like(distribution)
        mask.scatter_(-1, topk_idx, ones)
        motif_indices = torch.multinomial(distribution * mask, 1)
    return motif_indices