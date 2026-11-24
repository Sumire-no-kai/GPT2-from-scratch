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
