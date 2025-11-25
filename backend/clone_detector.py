import re
import difflib
import time
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any

# Optional BERT / CodeBERT imports
HAS_BERT = True
try:
    import torch
    from transformers import RobertaTokenizer, RobertaModel
except Exception:  
    HAS_BERT = False
    torch = None  
    RobertaTokenizer = None  
    RobertaModel = None  

# Logging configuration

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# Utility helpers

COMMENT_SINGLE_LINE_CPP = re.compile(r"//.*")
COMMENT_MULTI_LINE_CPP = re.compile(r"/\*[\s\S]*?\*/", re.MULTILINE)


def strip_comments(text: str) -> str:
    #Remove C / C++ style comments from source code.
    
    without_single = COMMENT_SINGLE_LINE_CPP.sub("", text)
    without_multi = COMMENT_MULTI_LINE_CPP.sub("", without_single)
    return without_multi


def normalize_whitespace(text: str) -> str:
    # for type 1
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")
TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+|[^\s]")


def tokenize(text: str) -> List[str]:
    
    # help in type2
    return TOKEN_RE.findall(text)


def extract_keywords_structure(text: str) -> str:
    
    # extract keywords

    lowered = text.lower()
    keywords = re.findall(
        r"\b(if|else|for|while|switch|case|class|def|function|try|catch|return)\b",
        lowered,
    )
    brace_count = (
        lowered.count("{")
        + lowered.count("}")
        + lowered.count("(")
        + lowered.count(")")
    )
    return " ".join(keywords) + (" |brace|" * brace_count)


def language_hint_from_extension(filename: Optional[str]) -> Optional[str]:
    """
    Very lightweight language hint based on file extension.
    Currently unused by the detector logic but kept for future extension.
    """
    if not filename:
        return None
    filename = filename.lower()
    if filename.endswith((".c", ".h")):
        return "c"
    if filename.endswith((".cpp", ".hpp", ".cc")):
        return "cpp"
    if filename.endswith(".py"):
        return "python"
    if filename.endswith(".java"):
        return "java"
    if filename.endswith((".js", ".ts")):
        return "javascript"
    return None


# Data models

@dataclass
class CodeSnippet:
    
    raw: str
    filename: Optional[str] = None
    language: Optional[str] = None

    cleaned: Optional[str] = field(default=None, init=False)
    tokens: Optional[List[str]] = field(default=None, init=False)
    structure: Optional[str] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.language is None:
            self.language = language_hint_from_extension(self.filename)

    # Lazy computations 

    def get_cleaned(self) -> str:
        if self.cleaned is None:
            no_comments = strip_comments(self.raw)
            self.cleaned = normalize_whitespace(no_comments)
            logger.debug("Computed cleaned code (%d chars)", len(self.cleaned))
        return self.cleaned

    # same function as that of type 1 and 2

    def get_tokens(self) -> List[str]:
        if self.tokens is None:
            self.tokens = tokenize(self.raw)
            logger.debug("Tokenized code into %d tokens", len(self.tokens))
        return self.tokens

    def get_structure(self) -> str:
        if self.structure is None:
            self.structure = extract_keywords_structure(self.raw)
            logger.debug(
                "Computed structural representation (%d chars)",
                len(self.structure),
            )
        return self.structure


@dataclass
class TechniqueResult:
    # for calculating similarity percentage
    score: float
    elapsed: float


@dataclass
class CloneReport:
    
    type1: TechniqueResult
    type2: TechniqueResult
    type3: TechniqueResult
    type4: TechniqueResult
    weights: Dict[str, float]

    def overall_score(self) -> float:
        
        # Compute a weighted average of all technique scores.
        
        weighted_sum = (
            self.type1.score * self.weights.get("type1", 1.0)
            + self.type2.score * self.weights.get("type2", 1.0)
            + self.type3.score * self.weights.get("type3", 1.0)
            + self.type4.score * self.weights.get("type4", 1.0)
        )
        total_weight = (
            self.weights.get("type1", 1.0)
            + self.weights.get("type2", 1.0)
            + self.weights.get("type3", 1.0)
            + self.weights.get("type4", 1.0)
        )
        if total_weight == 0:
            return 0.0
        return round(weighted_sum / total_weight, 2)

    def to_legacy_dict(self) -> Dict[str, Any]:
      
        return {
            "type1": round(self.type1.score, 2),
            "type2": round(self.type2.score, 2),
            "type3": round(self.type3.score, 2),
            "type4": round(self.type4.score, 2),
            "overall": self.overall_score(),
            "times": {
                "type1": round(self.type1.elapsed, 6),
                "type2": round(self.type2.elapsed, 6),
                "type3": round(self.type3.elapsed, 6),
                "type4": round(self.type4.elapsed, 6),
            },
        }


# Similarity strategies

class SimilarityStrategy:
   
    name: str = "base"

    def run(self, a: CodeSnippet, b: CodeSnippet) -> TechniqueResult:
        raise NotImplementedError


class Type1TextualStrategy(SimilarityStrategy):
    
    # Type-1: Textual similarity after comment removal and basic
    
    name = "type1"

    def run(self, a: CodeSnippet, b: CodeSnippet) -> TechniqueResult:
        start = time.time()
        cleaned_a = a.get_cleaned()
        cleaned_b = b.get_cleaned()
        matcher = difflib.SequenceMatcher(None, cleaned_a, cleaned_b)
        score = matcher.ratio() * 100.0
        end = time.time()
        return TechniqueResult(score=round(score, 2), elapsed=end - start)


class Type2TokenStrategy(SimilarityStrategy):
    # Type-2: Token-based similarity with identifier normalization.
    

    name = "type2"

    @staticmethod
    def _normalize_tokens(tokens: List[str]) -> str:
        mapping: Dict[str, str] = {}
        result: List[str] = []
        counter = 1

        for token in tokens:
            if IDENTIFIER_RE.match(token):
                key = token.lower()
                if key not in mapping:
                    mapping[key] = f"VAR{counter}"
                    counter += 1
                result.append(mapping[key])
            else:
                result.append(token)
        return " ".join(result)

    def run(self, a: CodeSnippet, b: CodeSnippet) -> TechniqueResult:
        start = time.time()
        tokens_a = a.get_tokens()
        tokens_b = b.get_tokens()

        norm_a = self._normalize_tokens(tokens_a)
        norm_b = self._normalize_tokens(tokens_b)

        matcher = difflib.SequenceMatcher(None, norm_a, norm_b)
        score = matcher.ratio() * 100.0
        end = time.time()
        return TechniqueResult(score=round(score, 2), elapsed=end - start)


class Type3StructuralStrategy(SimilarityStrategy):
        #Type-3: Structural similarity based on the control-flow keywords
    
    # and brace / parenthesis patterns. This technique is intentionally
    # language-agnostic and provides a high-level view of the program
    # structure.
    

    name = "type3"

    def run(self, a: CodeSnippet, b: CodeSnippet) -> TechniqueResult:
        start = time.time()
        struct_a = a.get_structure()
        struct_b = b.get_structure()

        matcher = difflib.SequenceMatcher(None, struct_a, struct_b)
        score = matcher.ratio() * 100.0
        end = time.time()
        return TechniqueResult(score=round(score, 2), elapsed=end - start)


#  Type-4: BERT / CodeBERT semantic similarity
_BERT_TOKENIZER = None
_BERT_MODEL = None


def _load_bert_models():
    """
    Lazy-load CodeBERT model and tokenizer, if available.
    """
    global _BERT_TOKENIZER, _BERT_MODEL
    if not HAS_BERT:
        logger.warning("BERT libraries not available; falling back from Type-4.")
        return None, None

    if _BERT_TOKENIZER is None or _BERT_MODEL is None:
        logger.info("Loading CodeBERT model (microsoft/codebert-base)...")
        _BERT_TOKENIZER = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        _BERT_MODEL = RobertaModel.from_pretrained("microsoft/codebert-base")
        logger.info("CodeBERT loaded successfully.")
    return _BERT_TOKENIZER, _BERT_MODEL


class Type4SemanticBERTStrategy(SimilarityStrategy):
    
    # Type-4: Semantic similarity based on CodeBERT embeddings.
    
    name = "type4"

    def __init__(self, fallback_strategy: Optional[Type2TokenStrategy] = None):
        self.fallback_strategy = fallback_strategy or Type2TokenStrategy()

    def _compute_cosine(self, v1, v2) -> float:
        # Using PyTorch cosine similarity
        return torch.nn.functional.cosine_similarity(v1, v2, dim=0).item()  # type: ignore

    def run(self, a: CodeSnippet, b: CodeSnippet) -> TechniqueResult:
        start = time.time()

        tokenizer, model = _load_bert_models()
        if tokenizer is None or model is None or torch is None:
            # Fallback to Type-2 if BERT is not available
            fallback_result = self.fallback_strategy.run(a, b)
            score = fallback_result.score * 0.9  # slightly penalize
            end = time.time()
            logger.debug(
                "Type-4 fallback to Type-2 (score=%.2f, elapsed=%.4f)",
                score,
                end - start,
            )
            return TechniqueResult(score=round(score, 2), elapsed=end - start)

        # Encode both snippets
        inputs = tokenizer(
            [a.raw, b.raw],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():  # type: ignore
            outputs = model(**inputs)

        # Mean pooling over sequence length
        embeddings = outputs.last_hidden_state.mean(dim=1)
        cos = self._compute_cosine(embeddings[0], embeddings[1])
        score = ((cos + 1) / 2.0) * 100.0  # map [-1, 1] -> [0, 100]

        end = time.time()
        return TechniqueResult(score=round(score, 2), elapsed=end - start)


# Clone detector core

@dataclass
class CloneDetectorConfig:
    
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "type1": 1.0,
            "type2": 1.0,
            "type3": 1.0,
            "type4": 1.0,
        }
    )
    enable_type1: bool = True
    enable_type2: bool = True
    enable_type3: bool = True
    enable_type4: bool = True


class CloneDetector:
    

    def __init__(self, config: Optional[CloneDetectorConfig] = None) -> None:
        self.config = config or CloneDetectorConfig()
        self._init_strategies()

    def _init_strategies(self) -> None:
        self.type1_strategy: Optional[Type1TextualStrategy] = (
            Type1TextualStrategy() if self.config.enable_type1 else None
        )
        self.type2_strategy: Optional[Type2TokenStrategy] = (
            Type2TokenStrategy() if self.config.enable_type2 else None
        )
        self.type3_strategy: Optional[Type3StructuralStrategy] = (
            Type3StructuralStrategy() if self.config.enable_type3 else None
        )
        self.type4_strategy: Optional[Type4SemanticBERTStrategy] = (
            Type4SemanticBERTStrategy(self.type2_strategy)
            if self.config.enable_type4
            else None
        )

    # Public API 

    def analyze(
        self,
        code1: str,
        code2: str,
        filename1: Optional[str] = None,
        filename2: Optional[str] = None,
    ) -> CloneReport:
        
        snippet1 = CodeSnippet(raw=code1, filename=filename1)
        snippet2 = CodeSnippet(raw=code2, filename=filename2)

        # Run strategies in sequence; each one measures its own time
        t1_result = (
            self.type1_strategy.run(snippet1, snippet2)
            if self.type1_strategy
            else TechniqueResult(score=0.0, elapsed=0.0)
        )
        t2_result = (
            self.type2_strategy.run(snippet1, snippet2)
            if self.type2_strategy
            else TechniqueResult(score=0.0, elapsed=0.0)
        )
        t3_result = (
            self.type3_strategy.run(snippet1, snippet2)
            if self.type3_strategy
            else TechniqueResult(score=0.0, elapsed=0.0)
        )
        t4_result = (
            self.type4_strategy.run(snippet1, snippet2)
            if self.type4_strategy
            else TechniqueResult(score=0.0, elapsed=0.0)
        )

        report = CloneReport(
            type1=t1_result,
            type2=t2_result,
            type3=t3_result,
            type4=t4_result,
            weights=self.config.weights,
        )
        logger.info(
            "Clone analysis completed: overall=%.2f (t1=%.2f, t2=%.2f, t3=%.2f, t4=%.2f)",
            report.overall_score(),
            report.type1.score,
            report.type2.score,
            report.type3.score,
            report.type4.score,
        )
        return report



# Create a single global detector instance used by detect_clones().

_GLOBAL_DETECTOR = CloneDetector()


def detect_clones(code1: str, code2: str) -> Dict[str, Any]:
    

    report = _GLOBAL_DETECTOR.analyze(code1, code2)
    return report.to_legacy_dict()
