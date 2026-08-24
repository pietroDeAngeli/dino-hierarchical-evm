import logging
import itertools
import numpy as np
# import torch
# from .memory import cc2clusters
from . import utils
from .utils import gen_parents, tree_from_list
import networkx as nx


class Supervisor(object):
    def __init__(self, dataset):
        self.dataset = dataset

        self.knowledge = tree_from_list(self.dataset.info["hierarchy"])

        self.global_mapper = {i: self.dataset.get_metadata("name", i)
                              for i in range(len(self.dataset))}

    def ask_supervision(self, s_id, pred, agent):
        root = utils.get_root(agent.obj_mem.T)
        mapper = {k: self.global_mapper[k]
                  for k in agent.obj_mem.T.nodes[root]["elem"]}
        inv_map = {self.global_mapper[k]: k
                   for k in agent.obj_mem.T.nodes[root]["elem"]}

        assert len(mapper) == len(inv_map)

        converted = convert_tree(agent.obj_mem.T, mapper)
        if pred[0] in mapper:
            pred = (mapper[pred[0]], pred[1])
        s_id = self.global_mapper[s_id]
        is_consistent = check_consistency(s_id, converted, pred[0],
                                          self.knowledge)

        if not is_consistent:
            start = get_first_consistent_node(s_id, converted, pred[0],
                                              self.knowledge)
        else:
            start = pred[0]

        sup = tree_supervise(s_id, converted, start, self.knowledge)
        stat = get_supervision_stat(converted, pred[0], sup[0])
        if sup[0] in inv_map:
            return inv_map[sup[0]], sup[1], stat
        else:
            return sup[0], sup[1], stat


def supervisor_factory(dataset):
    return Supervisor(dataset)


def convert_tree(original, mapper):
    converted = nx.DiGraph()

    for s, d in original.edges:
        converted.add_edge(mapper.get(s, s), mapper.get(d, d))

    for node in original.nodes:
        new_set = set(mapper[e] for e in original.nodes[node]["elem"])
        converted.add_node(mapper.get(node, node))
        converted.nodes[mapper.get(node, node)]["elem"] = new_set

    return converted


def gen_anchor_path(new, mem, node, knowledge):
    anchor = new
    yield anchor
    node_elem = mem.nodes[node]["elem"]
    while not knowledge.nodes[anchor]["elem"].issuperset(node_elem):
        anchor = next(iter(knowledge.pred[anchor].keys()))
        yield anchor


def get_anchor(new, mem, node, knowledge):
    gen = gen_anchor_path(new, mem, node, knowledge)
    anchor = None
    for anchor in gen:
        pass

    return anchor


def get_targets_anchor_node(new, mem, node, knowledge):
    anchor_path = list(gen_anchor_path(new, mem, node, knowledge))

    target_node = node

    stop = False
    target_anchor = anchor_path[-1]
    for a in reversed(anchor_path[:-1]):
        if stop:
            break
        stop = True
        elem = knowledge.nodes[a]["elem"]
        for child in mem.succ[target_node]:
            child_elem = mem.nodes[child]["elem"]
            if not utils.is_leaf(mem, child) or child in elem:
                if child_elem.issubset(elem):
                    stop = False
                    target_node = child
                    target_anchor = a

    return target_anchor, target_node


def tree_supervise(new, mem, node, knowledge):
    root = utils.get_root(mem)
    if new in mem.nodes[root]["elem"]:
        return new, False

    target_anchor, target_node = get_targets_anchor_node(new, mem,
                                                         node, knowledge)

    new_genus = False
    if len(mem.nodes) >= 1:
        for child in knowledge.succ[target_anchor]:
            target_elem = mem.nodes[target_node]["elem"]
            if target_elem.issubset(knowledge.nodes[child]["elem"]):
                new_genus = True

    return target_node, new_genus


def check_consistency(new, mem, node, knowledge, max_hops=100):
    anchor = get_anchor(new, mem, node, knowledge)

    consistent = True
    anchor_elem = knowledge.nodes[anchor]["elem"]
    node_elem = mem.nodes[node]["elem"]

    elem_diff = anchor_elem - node_elem
    for par, _ in zip(gen_parents(node, mem), range(max_hops)):
        if len(elem_diff & mem.nodes[par]["elem"]) > 0:
            consistent = False
            break

    return consistent


def get_first_consistent_node(new, mem, node, knowledge):
    node_path = [node] + list(gen_parents(node, mem))

    t_a, t_n = get_targets_anchor_node(new, mem, node_path[-1], knowledge)

    target_node_ancestors = set(gen_parents(t_n, mem))
    if not utils.is_leaf(mem, t_n):
        target_node_ancestors.add(t_n)

    for par in node_path:
        if par in target_node_ancestors:
            return par


def get_supervision_stat(tree, predicted, real_node, fail=False):
    if fail:
        from nose.tools import set_trace
        set_trace()
    if predicted == real_node:
        return (0, 0)

    pred_path = [predicted] + list(gen_parents(predicted, tree))
    real_path = [real_node] + list(gen_parents(real_node, tree))

    ancestor = min(len(pred_path), len(real_path))
    for i, (p, r) in enumerate(zip(reversed(pred_path), reversed(real_path))):
        if p != r:
            ancestor = i
            break

    go_up = len(pred_path) - ancestor

    assert go_up >= 0
    go_down = len(real_path) - ancestor
    assert go_down >= 0

    return (go_up, go_down)


def get_supervision_cost(go_up, go_down):
    return go_up + go_down * 3 + 1
