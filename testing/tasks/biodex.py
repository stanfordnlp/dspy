from __future__ import annotations
from .base_task import BaseTask
import sys
import dspy
from dspy.evaluate import Evaluate
from dsp.utils import deduplicate
import tqdm
import datasets
import sys
import math
from functools import lru_cache
import os
import re
import pickle
from pydantic import BaseModel
from rapidfuzz import process
from typing import Union, Optional, Dict, DefaultDict, List
from enum import Enum
from collections import defaultdict

# Must point to a MedDra download with mdhier.asc
data_dir = '/future/u/okhattab/data/2023/MedDraV2/meddra_23_0_english/MedAscii' # NOTE: EDIT THIS LINE 

global meddra_root
meddra_root = None

def _remove_non_llt(candidates):
    return [c for c in candidates if c.level == Level.LLT]


def get_path(node):
    path = f"under {node.parent.term} < {node.parent.parent.term} < {node.parent.parent.parent.term}"
    return path


def flatten(ll):
    return [x for l in ll for x in l]


def _matches_to_terms_and_paths(matches):
    s = []
    for n in matches:
        new_s = f"{n.node.term} ({n.score} score for query '{n.query}'; {n.node.count} counts; {get_path(n.node)})"
        s.append(new_s)
    return s


class Level(Enum):
    ROOT = 0
    SOC = 1
    HLTG = 2
    HLT = 3
    LLT = 4

    def __add__(self, other: int):
        if self.ROOT.value <= self.value + other <= self.LLT.value:
            return Level(self.value + other)
        else:
            return None

    def __lt__(self, other) -> bool:
        return self.value < other.value

    def __le__(self, other) -> bool:
        return self.value <= other.value

    def __gt__(self, other) -> bool:
        return self.value > other.value

    def __ge__(self, other) -> bool:
        return self.value >= other.value

    def __sub__(self, other) -> int:
        return self.value - other.value


class Node(BaseModel):
    id: str
    term: str
    level: Level
    count: int = 0

    parent: 'Node' = None
    children: Dict[str, 'Node'] = {}

    # maps for efficient nested children lookups
    # typically only used for the root node
    id_to_nodes: Optional[DefaultDict[str, List['Node']]] = None
    term_to_nodes: Optional[DefaultDict[str, List['Node']]] = None

    # text normalizer to build lookup tables
    # typically only used for the root node
    normalizer: function = lambda x: x

    def get_parent_at_level(self, target_level: Level) -> Union['Node', None]:
        # check if current node is deeper than target_level
        level_diff = target_level - self.level
        if level_diff > 0:
            return None
        else:
            node = self
            for _ in range(-level_diff):
                node = node.parent
            return node

    def get_children_at_level(self, target_level: Level) -> List['Node']:
        raise NotImplementedError

    def lookup_id(self, id: str) -> Union[List['Node'], None]:
        # create lookup tables if they do not exists
        if self.id_to_nodes == None:
            self.set_lookup_tables()
        # lookup

        if id in self.id_to_nodes:
            return self.id_to_nodes[id]
        else:
            return None

    def lookup_term(self, term: str) -> Union[List['Node'], None]:
        # create lookup tables if they do not exists
        if self.term_to_nodes == None:
            raise RuntimeError("First set the lookup tables using 'set_lookup_tables'.")
        # lookup
        term = self.normalizer(term)
        if term in self.term_to_nodes:
            return self.term_to_nodes[term]
        else:
            return None

    def match(
        self,
        query: str,
        fuzzy_limit: int = 20,
        fuzzy_score_cutoff: int = 80,
        return_nodes: bool = False,
    ):
        """Return a tuple containing exact match and best fuzzy matches of LLT terms in ontology.

        Args:
            query (str): String being matched into ontology
            fuzzy_limit (int, optional): Maximal limit of fuzzy matches to return. Defaults to 20.
            fuzzy_score_cutoff (int, optional): Minimal fuzzy matching score, between 0 and 100. Defaults to 80.
            return_nodes (bool, optional): If true, return Node objects instead of paths

        Returns:
            tuple: Tuple(exact match or None, list of fuzzy matches)
        """
        exact_candidate = []
        fuzzy_candidates = []

        # exact match
        exact_matches = self.lookup_term(query)
        if exact_matches is not None:
            exact_candidate = _remove_non_llt(exact_matches)
            exact_candidate = [
                Match(query=query, score=100, node=node) for node in exact_candidate
            ]

        # fuzzy match
        fuzzy_terms = process.extract(
            query,
            self.term_to_nodes.keys(),
            limit=fuzzy_limit,
            score_cutoff=fuzzy_score_cutoff,
        )

        fuzzy_matches = []
        for term, sim, _ in fuzzy_terms:
            applicable_nodes = _remove_non_llt(self.term_to_nodes[term])
            for n in applicable_nodes:
                fuzzy_matches.append(Match(node=n, score=sim, query=query))

        if return_nodes:
            return exact_candidate, fuzzy_matches
        else:
            exact_term = (
                _matches_to_terms_and_paths(exact_candidate) if exact_candidate else []
            )
            fuzzy_terms = _matches_to_terms_and_paths(fuzzy_matches)
            return exact_term, fuzzy_terms

    def set_lookup_tables(self, normalizer: function = lambda x: x) -> None:
        # set normalizer
        self.normalizer = normalizer

        # get a flat list of all nodes
        self.term_to_nodes = defaultdict(list)
        self.id_to_nodes = defaultdict(list)

        # breadth-first traversal
        candidates = [self]
        while len(candidates):
            candidate = candidates.pop(0)
            # track nodes
            self.term_to_nodes[self.normalizer(candidate.term)].append(candidate)
            self.id_to_nodes[candidate.id].append(candidate)
            # add candidate's children to candidate list
            if candidate.children:
                candidates.extend(candidate.children.values())
        return None

    def is_equivalent_node(self, other: 'Node') -> bool:
        # return true if two nodes are the same, siblings or in a parent-child relation
        return (
            (self == other)
            or (self.parent == other)
            or (self == other.parent)
            or (self.parent == other.parent)
        )

    def terms_equivalent(self, term1: str, term2: str) -> Union[bool, None]:
        # lookup both terms
        term1_nodes = self.lookup_term(term1)
        term2_nodes = self.lookup_term(term2)
        # if at least one term not found
        if not term1_nodes or not term2_nodes:
            return None
        # compare all nodes
        for node1 in term1_nodes:
            for node2 in term2_nodes:
                if node1.is_equivalent_node(node2):
                    return True
        return False

    def __key(self):
        return (self.id, self.term, self.parent, len(self.children))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other: 'Node') -> bool:
        # undeep equals operator, does not check children
        return self.__key() == other.__key()

    def __str__(self) -> str:
        return f"Node(id={self.id}, term={self.term}, level={self.level}, parent={self.parent.term if self.parent else None}, #children={len(self.children)}, count={self.count})"

    def __repr__(self) -> str:
        return self.__str__()

    def get_path(self):
        path = f"under {self.parent.term} < {self.parent.parent.term} < {self.parent.parent.parent.term}"
        return path

    @property
    def all_nodes(self):
        return flatten(list(self.id_to_nodes.values()))

    @property
    def all_llt_nodes(self):
        return _remove_non_llt(self.all_nodes)


class Match(BaseModel):
    node: Node
    score: int
    query: str

def _create_child_if_not_exists(
    parent: Node, child_id: str, child_term: str, term_to_count: dict
):
    parent_level = parent.level
    # get child level if still appropriate
    child_level = parent_level + 1
    if child_level is None:
        return None
    # get child if exists
    if child_id in parent.children:
        child = parent.children[child_id]
    # create child if not exists
    else:
        child_term_dict = child_term.lower().replace("'", "^")
        if child_term_dict in term_to_count:
            count = term_to_count[child_term_dict]
        else:
            count = 0
        child = Node(
            id=child_id, term=child_term, level=child_level, parent=parent, count=count
        )
        parent.children.update({child_id: child})
    return child

def parse_mdhier(data_dir):
    file = os.path.join(data_dir, "mdhier.asc")
    if not os.path.isfile(file):
        raise FileNotFoundError(f"File {file} not found.")
    count_file = os.path.join(data_dir, "priors.pickle")
    if not os.path.isfile(count_file):
        raise FileNotFoundError(f"File {count_file} not found.")

    # get regex
    id_regex = r"\d{8}"
    term_regex = r"[^$]*"
    regex = rf"({id_regex})\$({id_regex})\$({id_regex})\$({id_regex})\$({term_regex})\$({term_regex})\$({term_regex})\$({term_regex})\$({term_regex})\$\$({id_regex})\$[YN]\$"
    regex = re.compile(regex)

    # get data
    lines = []
    with open(file, "r") as fp:
        lines = fp.readlines()
    text = "\n".join(lines)

    matches = regex.findall(text)

    print(f"Read {len(lines)} lines.")
    if len(lines) != len(matches):
        raise RuntimeError(
            f"Error parsing mdhier.asc. Expected number of matches to match number of lines."
        )

    # load counts
    with open(count_file, "rb") as fp:
        counts = pickle.load(fp)

    # create ontology tree
    root = Node(id="root", term="root", level=Level.ROOT)
    for match in matches:
        (
            llt_id,
            hlt_id,
            hltg_id,
            soc_id,
            llt_term,
            hlt_term,
            hltg_term,
            soc_term_1,
            soc_term_2,
            _,
        ) = match
        # recursivelty add the nodes specified by this line
        soc_node = _create_child_if_not_exists(root, soc_id, soc_term_1, counts)
        hltg_node = _create_child_if_not_exists(soc_node, hltg_id, hltg_term, counts)
        hlt_node = _create_child_if_not_exists(hltg_node, hlt_id, hlt_term, counts)
        llt_node = _create_child_if_not_exists(hlt_node, llt_id, llt_term, counts)

    # print(llt_node.parent == hlt_node)
    # print(hlt_node.parent == hltg_node)
    # print(hltg_node.parent == soc_node)
    # print(soc_node.parent == root)

    # print(Level.SOC < Level.LLT)
    # print(Level.SOC > Level.LLT)
    # print(Level.SOC - Level.LLT)

    # print(soc_node.get_parent_at_level(Level.ROOT))
    # print(soc_node.get_parent_at_level(Level.SOC))
    # print(soc_node.get_parent_at_level(Level.LLT))
    # print(llt_node.get_parent_at_level(Level.SOC))

    # print(root.lookup_term("COVID-19 pneumonia"))
    # print(root.lookup_term("COVID-19 pneumonia")[0].parent)
    return root

@lru_cache(maxsize=100000)
def base_ground(reaction, K):
    global meddra_root
    kwargs = dict(fuzzy_limit=K, fuzzy_score_cutoff=75, return_nodes=True)
    exact_matches, fuzzy_matches = meddra_root.match(reaction, **kwargs)

    return exact_matches, fuzzy_matches


@lru_cache(maxsize=100000)
def ground_v1(reaction, K=3):
    """
        Prefers exact matches over fuzzy matches, when available.
    """

    exact_matches, fuzzy_matches = base_ground(reaction, K)

    matches = exact_matches[:1] or fuzzy_matches
    matches = [(match.score / 100.0, match.node.term) for match in matches]
    matches = sorted(list(set(matches)), reverse=True)

    return tuple(matches)

@lru_cache(maxsize=100000)
def ground_v2(reaction, K=1):
    """
        When K=1 (default), returns exact matches (if available) or the best fuzzy match.
    """

    exact_matches, fuzzy_matches = base_ground(reaction, K)

    matches = [(match.score / 100.0, match.node.term) for match in (exact_matches[:1] + fuzzy_matches)]
    matches = sorted(list(set(matches)), reverse=True)[:K]

    return tuple(matches)

# Returns the best three matches (including one exact match, if available).
ground_v3 = lambda reaction: ground_v2(reaction, K=3)

@lru_cache(maxsize=100000)
def ground_v4(reaction, K=3):
    """
        Returns the best three matches (including one exact match, if available) and applies a prior.
    """
    exact_matches, fuzzy_matches = base_ground(reaction, K)
    
    matches = [((match.score / 100.0) * math.sqrt(match.node.count + 0.1), match.node.term)
               for match in (exact_matches[:1] + fuzzy_matches)]
    matches = sorted(list(set(matches)), reverse=True)[:K]

    return tuple(matches)

@lru_cache(maxsize=100000)
def ground_v4b(reaction, K=3):
    """
        Returns the best three matches (including one exact match, if available) and applies a prior.
    """

    exact_matches, fuzzy_matches = base_ground(reaction, K)
    
    matches = [((match.score / 100.0) * math.log(match.node.count + 0.1), match.node.term)
               for match in (exact_matches[:1] + fuzzy_matches)]
    matches = sorted(list(set(matches)), reverse=True)[:K]

    return tuple(matches)

@lru_cache(maxsize=100000)
def ground_v4c(reaction, K=3):
    """
        Returns the best three matches (including one exact match, if available) and applies a prior.
    """

    exact_matches, fuzzy_matches = base_ground(reaction, K)
    
    matches = [((match.score / 100.0) * (match.node.count + 0.1), match.node.term)
               for match in (exact_matches[:1] + fuzzy_matches)]
    matches = sorted(list(set(matches)), reverse=True)[:K]

    return tuple(matches)


@lru_cache(maxsize=100000)
def ground_v5(reaction, K=3):
    exact_matches, fuzzy_matches = base_ground(reaction, K)
    
    matches = [((match.score / 100.0)**2 * math.log(match.node.count + 0.1), match.node.term)
               for match in (exact_matches[:1] + fuzzy_matches)]
    matches = sorted(list(set(matches)), reverse=True)[:K]

    return tuple(matches)

def preprocess_example(example):
    title = example['title']
    abstract = example['abstract']
    context = example['fulltext_processed'].split('\n\nTEXT:\n', 1)[-1]
    reactions = [r.strip().lower() for r in example['reactions'].split(',')]

    example = dict(title=title, abstract=abstract, context=context, reactions=reactions)
    example['labels'] = dspy.Example(reactions=reactions)

    return example

class Chunker:
    def __init__(self, context_window=3000, max_windows=5):
        self.context_window = context_window
        self.max_windows = max_windows
        self.window_overlap = 0.02
    
    def __call__(self, paper):
        snippet_idx = 0

        while snippet_idx < self.max_windows and paper:
            endpos = int(self.context_window * (1.0 + self.window_overlap))
            snippet, paper = paper[:endpos], paper[endpos:]

            next_newline_pos = snippet.rfind('\n')
            if paper and next_newline_pos != -1 and next_newline_pos >= self.context_window // 2:
                paper = snippet[next_newline_pos+1:] + paper
                snippet = snippet[:next_newline_pos]

            yield snippet_idx, snippet.strip()
            snippet_idx += 1

def get_set_precision_and_recalls(l1, l2):
    # lower case and strip
    l1 = [x.strip().lower() for x in l1]
    l2 = [x.strip().lower() for x in l2]

    s1, s2 = set(l1), set(l2)
    intersect = s1.intersection(s2)

    p, r = len(intersect) / max(1, len(s1)), len(intersect) / max(1, len(s2))
    f1 = 2 * (p * r) / max((p + r), 1)

    return dspy.Prediction(precision=p, recall=r, f1=f1)

def finegrained_metrics(gold, pred):
    reaction_metrics = get_set_precision_and_recalls(pred.reactions, gold.reactions)
    return reaction_metrics


class CompilationMetric:
    __name__ = 'compilation_metric'

    def __init__(self, field='reactions', metric='recall', perfect_precision=False):
        self.field = field
        self.metric = metric
        self.perfect_precision = perfect_precision

    def __call__(self, gold, pred, trace=None):
        field = self.field
        metric = finegrained_metrics(gold, pred) #[field]

        if trace is None: # eval
            return metric[self.metric]

        if metric.precision >= 0.5 and metric.recall >= 0.5:
            for trace_idx in range(len(trace)):
                *_, outputs = trace[trace_idx]

                values = outputs[field].split(', ')
                values = [r.lower().strip('.').strip() for r in values]
                values = deduplicate([r for r in values if r and r != 'n/a'])

                if self.perfect_precision:
                    values = [r for r in values if r in gold[self.field]]
                
                outputs[field] = ', '.join(values) or 'N/A'

                # print(gold[field], '\t\t\t', values)

            return True

        return False


metricR = CompilationMetric(field='reactions', metric='recall')
from collections import Counter

def reduce_grounded_reactions(grounded_reactions):
    scores = {}
    for score, reaction in grounded_reactions:
        scores[reaction] = scores.get(reaction, 0) + score
    reactions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    reactions = [r for r, _ in reactions]
    return reactions


def metricR10(gold, pred, trace=None):
    reactions = pred.reactions

    try:
        reactions = reduce_grounded_reactions(reactions)
    except Exception:
        reactions = Counter(reactions).most_common()
        reactions = [r for r, _ in reactions]

    return metricR(gold, pred.copy(reactions=reactions[:10]), trace)


def extract_reactions_from_strings(reactions):
    reactions = [r.strip().split('\n')[0].lower().strip('.').strip() for r in reactions]
    reactions = ', '.join(reactions)
    reactions = [r.lower().strip('.').strip() for r in reactions.split(',')]
    reactions = [r for r in reactions if r and r != 'n/a']

    return reactions

class PredictReactions(dspy.Signature):
    __doc__ = f"""Given a snippet from a medical article, identify the adverse drug reactions affecting the patient. If none are mentioned in the snippet, say N/A."""

    title = dspy.InputField()
    context = dspy.InputField()
    reactions = dspy.OutputField(desc="list of comma-separated adverse drug reactions", format=lambda x: ', '.join(x) if isinstance(x, list) else x)

HINT = "Is any of the following reactions discussed in the article snippet? The valid candidates are:"

class GroundedReactionExtractor(dspy.Module):
    def __init__(self, context_window=3000, max_windows=5, num_preds=1):
        super().__init__()

        self.chunk = Chunker(context_window=context_window, max_windows=max_windows)        
        self.predict = dspy.ChainOfThoughtWithHint(PredictReactions, n=num_preds)
    
    def forward(self, title, abstract, context, labels=None):
        hint = f"{HINT} {', '.join(labels.reactions)}." if labels else None
        reactions = []

        for _, snippet in self.chunk(abstract + '\n\n' + context):
            chunk_reactions = self.predict(title=title, context=[snippet], hint=hint)
            reactions.extend(extract_reactions_from_strings(chunk_reactions.completions.reactions))

        reactions = [r for sublist in [ground_v4b(r) for r in reactions] for r in sublist]
        return dspy.Prediction(reactions=reactions)

class BioDexTask(BaseTask):
    def __init__(self):
        global meddra_root

        meddra_root = parse_mdhier(data_dir)
        meddra_root.set_lookup_tables(normalizer=str.lower)

        # Load and configure the datasets.
        dataset = datasets.load_dataset("BioDEX/BioDEX-Reactions")
        official_trainset, official_devset = dataset['train'], dataset['validation']
        trainset, devset = [], []

        for example in tqdm.tqdm(official_trainset):
            if len(trainset) >= 1000: break
            trainset.append(preprocess_example(example))

        for example in tqdm.tqdm(official_devset):
            if len(devset) >= 500: break
            devset.append(preprocess_example(example))

        trainsetX = [dspy.Example(**x).with_inputs('title', 'abstract', 'context', 'labels') for x in trainset]
        trainset = [dspy.Example(**x).with_inputs('title', 'abstract', 'context') for x in trainset]
        devsetX = [dspy.Example(**x).with_inputs('title', 'abstract', 'context', 'labels') for x in devset]
        devset = [dspy.Example(**x).with_inputs('title', 'abstract', 'context') for x in devset]

        self.trainset = trainset
        self.devset = devset

        # Set up metrics
        NUM_THREADS = 16

        self.metric = metricR10

        kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=15)
        self.evaluarteR10 = Evaluate(devset=self.devset, metric=metricR10, **kwargs)

        self.set_splits(TRAIN_NUM=100, DEV_NUM=100, EVAL_NUM=100)

    def get_program(self):
        return GroundedReactionExtractor(context_window=3000, max_windows=5, num_preds=1) 
    
    def get_metric(self):
        return self.metric