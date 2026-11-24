import json
from pathlib import Path

import regex


# 预切分模式依据 GPT-2 官方编码器的 src/encoder.py。
_GPT2_PATTERN = regex.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
_ENDOFTEXT = "<|endoftext|>"


# 为 BPE 挑选要合并的相邻对提供频次。将原列表与从第二项开始的切片逐项配对，
# 再用每一对 ID 作字典键累计次数。
def count_pairs(ids: list[int]) -> dict[tuple[int, int], int]:
    """统计一维编号序列里每种有序相邻对的出现次数。

    每个相邻位置都参与统计，允许相邻对共享位置。
    少于两个编号时返回空字典。
    """
    counts: dict[tuple[int, int], int] = {}
    # 右侧序列错开一格，相邻两次配对可以共用中间的编号。
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


# 执行一条 BPE 合并规则。从左往右扫描，匹配时写入新 ID 并消耗两个位置，
# 其余时候复制当前 ID 并前进一格。
def merge_pair(
    ids: list[int], pair: tuple[int, int], new_id: int
) -> list[int]:
    """从左到右将指定的相邻对替换成新的编号。

    每个原始位置最多参与一次合并。未匹配的编号保持原顺序。
    new_id 由调用方分配。返回新列表，保留输入列表原样。
    """
    merged: list[int] = []
    i = 0
    n = len(ids)

    while i < n:
        # i 指向第一个还没处理的输入位置，merged 里的新编号不参与本轮匹配。
        if i + 1 < n and (ids[i], ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2  # 两个输入位置都已用掉，重叠的匹配只能取左边一次。
        else:
            merged.append(ids[i])
            i += 1  # 只用掉当前位置，右邻居仍有机会作为下一对的起点。

    return merged


# 从训练序列中学出编码时可复用的合并规则。每轮选出最高频相邻对并分配新 ID，
# 在合并后的序列上重新统计，最终按学习顺序返回规则。
def train_bpe(
    ids: list[int], num_merges: int, start_id: int
) -> dict[tuple[int, int], int]:
    """在一条编号序列上学习至多 num_merges 条 BPE 合并规则。

    start_id 是第一个新编号，应大于基础词表中的所有编号。
    频次并列时，选当前序列里最先出现的相邻对。
    返回按学习顺序插入的规则字典，保留原始输入列表。
    """
    if num_merges < 0:
        raise ValueError("num_merges must be non-negative")
    if ids and start_id <= max(ids):
        raise ValueError("start_id must be greater than all input token IDs")

    rules: dict[tuple[int, int], int] = {}
    for new_id in range(start_id, start_id + num_merges):
        counts = count_pairs(ids)
        if not counts:
            break

        # count_pairs 按首次出现顺序插入，max 在频次并列时保留第一个。
        pair = max(counts, key=lambda pair: counts[pair])
        rules[pair] = new_id
        ids = merge_pair(ids, pair, new_id)

    return rules


# 把训练时学到的 BPE 规则按顺序应用到一条新的编号序列。
# 每条规则都接着处理上一条规则的合并结果。
def encode_bpe(
    ids: list[int], rules: dict[tuple[int, int], int]
) -> list[int]:
    """按学习顺序应用合并规则，返回编码后的新列表。

    ids 是基础词表中的 token ID 序列，rules 来自 train_bpe。
    输入列表和规则字典保持原样。没有规则时返回输入列表的副本。
    """
    encoded = list(ids)
    for pair, new_id in rules.items():
        encoded = merge_pair(encoded, pair, new_id)
    return encoded


# 把合并出的 token ID 展开回基础 ID。逆着学习顺序拆开每条规则，
# 这样拆出的中间 ID 还能继续用较早的规则展开。
def decode_bpe(
    ids: list[int], rules: dict[tuple[int, int], int]
) -> list[int]:
    """将同一套规则编码后的 ID 序列还原为基础 ID 的新列表。

    rules 来自 train_bpe，基础 ID 在拆解时原样保留。
    输入列表和规则字典保持原样。没有规则时返回输入列表的副本。
    """
    decoded = list(ids)
    for pair, new_id in reversed(rules.items()):
        next_decoded: list[int] = []
        for token_id in decoded:
            if token_id == new_id:
                next_decoded.extend(pair)
            else:
                next_decoded.append(token_id)
        decoded = next_decoded
    return decoded


# 为 GPT-2 的 256 个字节值分配可逆的单字符表示。
# 能直接保留的字节沿用同编号字符，其余按字节值顺序分配新字符。
def bytes_to_unicode() -> dict[int, str]:
    """建立与 GPT-2 官方编码器一致的字节到字符映射。

    键覆盖 0 到 255，值是互不重复的单个 Unicode 字符。
    """
    direct_ranges = (range(33, 127), range(161, 173), range(174, 256))
    mapping = {byte: chr(byte) for interval in direct_ranges for byte in interval}

    next_codepoint = 256
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(next_codepoint)
            next_codepoint += 1
    return mapping


class GPT2Tokenizer:
    """从官方词表和合并规则构建的 GPT-2 字节级 tokenizer。"""

    # 读取词表和合并顺序，预先建好编码与解码用的查找表。
    def __init__(
        self, encoder_path: str | Path, merges_path: str | Path
    ) -> None:
        self.encoder: dict[str, int] = json.loads(
            Path(encoder_path).read_text(encoding="utf-8")
        )
        self.decoder = {token_id: symbol for symbol, token_id in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {symbol: byte for byte, symbol in self.byte_encoder.items()}

        lines = Path(merges_path).read_text(encoding="utf-8").splitlines()
        merge_lines = (line for line in lines[1:] if line.strip())
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        for rank, line in enumerate(merge_lines):
            left, right = line.split()
            self.bpe_ranks[left, right] = rank

    # 在单个预切分片段里反复合并优先级最高的相邻符号。
    def _bpe(self, symbols: str) -> list[str]:
        """返回片段合并后的符号列表。合并表里序号越小，优先级越高。"""
        if len(symbols) < 2:
            return list(symbols)

        word = list(symbols)
        while len(word) >= 2:
            pairs = set(zip(word, word[1:]))
            # 未收录的对排在所有有效规则之后。整轮无规则时，下面直接结束。
            bigram = min(
                pairs,
                key=lambda pair: self.bpe_ranks.get(pair, float("inf")),
            )
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            merged: list[str] = []
            i = 0
            n = len(word)
            # 只扫描本轮的输入序列，新拼出的符号从下一轮开始参与配对。
            while i < n:
                if i + 1 < n and word[i] == first and word[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged

        return word

    # 先按 GPT-2 的规则预切分，再逐段转字节、执行 BPE 和查词表。
    def encode(self, text: str, *, allow_endoftext: bool = False) -> list[int]:
        """编码文本。allow_endoftext 为真时，将结束标记作为特殊 token。"""
        parts = text.split(_ENDOFTEXT) if allow_endoftext else [text]
        ids: list[int] = []
        for index, part in enumerate(parts):
            if index:
                ids.append(self.encoder[_ENDOFTEXT])

            for match in _GPT2_PATTERN.finditer(part):
                chunk = match.group().encode("utf-8")
                symbols = "".join(self.byte_encoder[byte] for byte in chunk)
                ids.extend(self.encoder[token] for token in self._bpe(symbols))
        return ids

    # 先恢复完整字节序列，再做 UTF-8 解码，保留跨 token 的多字节字符。
    def decode(self, ids: list[int], *, errors: str = "replace") -> str:
        """把 token ID 还原成文本。errors 传给 UTF-8 解码器。"""
        symbols = "".join(self.decoder[token_id] for token_id in ids)
        raw_bytes = bytes(self.byte_decoder[symbol] for symbol in symbols)
        return raw_bytes.decode("utf-8", errors=errors)
