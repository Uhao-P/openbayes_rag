from typing import List, Optional, Union, Literal

'''
### 总结
这个类主要用于处理需要按特定分隔符分割的序列数据（如文本、令牌等），并且可以控制分割后的块大小和块之间的重叠，常见于自然语言处理任务中，如分割文本为处理单元等。
'''
class DocumentSeparatorSplitter:
    
    '''
    ### 类定义和初始化方法 `__init__`
    - **参数**:
    - `separators`: 可选，分隔符列表，每个分隔符是一个由整数构成的列表。
    - `keep_separator`: 指定如何处理分隔符。可以是布尔值或者字面量 "start" 或 "end"。
        - `True` 或 "end"：在分隔后的文本块末尾保留分隔符。
        - "start"：在下一个文本块的开始保留分隔符。
        - `False`：分隔符不保留。
    - `chunk_size`: 指定每个文本块的最大长度。
    - `chunk_overlap`: 指定文本块之间的重叠长度。
    - `length_function`: 用于计算文本块长度的函数，默认为 `len` 函数。
    '''
    def __init__(
        self,
        separators: Optional[List[List[int]]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = "end",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: callable = len,
    ):
        self._separators = separators or []
        self._keep_separator = keep_separator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        
    '''
    ### 方法 `split_tokens_with_separator`
    - 接受一个整数列表 `tokens`，代表待分割的文本（例如，可以是一系列词汇的索引）。
    - 调用 `_split_tokens_with_separators` 方法进行初步分割，然后使用 `_merge_splits` 方法合并分割结果。
    '''
    def split_tokens_with_separator(self, tokens: List[int]) -> List[List[int]]:
        splits = self._split_tokens_with_separators(tokens)
        return self._merge_splits(splits)
    
    '''
    ### 方法 `split_tokens`
    - 接受一个整数列表 `tokens`，代表待分割的文本（例如，可以是一系列词汇的索引）。
    - 直接进行切割
    '''
    def split_tokens(self, tokens: List[int]) -> List[List[int]]:
        return self._merge_splits([tokens])
    
    '''
    ### 内部方法 `_split_tokens_with_separators`
    - 遍历 `tokens`，使用 `separators` 列表中定义的分隔符进行分割。
    - 根据 `keep_separator` 参数的设置，决定分隔符的保留位置。
    - 返回一个列表，其中包含分割后的各个部分。
    '''
    def _split_tokens_with_separators(self, tokens: List[int]) -> List[List[int]]:
        splits = []
        current_split = []
        i = 0
        while i < len(tokens):
            separator_found = False
            for separator in self._separators:
                if tokens[i:i+len(separator)] == separator:
                    if self._keep_separator in [True, "end"]:
                        current_split.extend(separator)
                    if current_split:
                        splits.append(current_split)
                        current_split = []
                    if self._keep_separator == "start":
                        current_split.extend(separator)
                    i += len(separator)
                    separator_found = True
                    break
            if not separator_found:
                current_split.append(tokens[i])
                i += 1
        if current_split:
            splits.append(current_split)
        return [s for s in splits if s]
    
    '''
    ### 内部方法 `_merge_splits`
    - 基于 `chunk_size` 和 `chunk_overlap` 参数，合并 `_split_tokens_with_separators` 方法生成的分割列表。
    - 确保每个合并后的块不超过 `chunk_size`，并且根据 `chunk_overlap` 进行重叠处理。
    '''
    def _merge_splits(self, splits: List[List[int]]) -> List[List[int]]:
        if not splits:
            return []

        merged_splits = []
        current_chunk = []

        for split in splits:
            if not current_chunk:
                current_chunk = split
            elif self._length_function(current_chunk) + self._length_function(split) <= self._chunk_size:
                current_chunk.extend(split)
            else:
                merged_splits.append(current_chunk)
                current_chunk = split

        if current_chunk:
            merged_splits.append(current_chunk)

        if len(merged_splits) == 1 and self._length_function(merged_splits[0]) > self._chunk_size:
            return self._split_chunk(merged_splits[0])

        if self._chunk_overlap > 0:
            return self._enforce_overlap(merged_splits)
        
        return merged_splits

    '''
    ### 内部方法 `_split_chunk`
    - 如果单个块的大小超过 `chunk_size`，将其进一步分割。
    - 使用 `chunk_overlap` 作为步长减少量，确保分割后的每个新块不会丢失重要数据。
    '''
    def _split_chunk(self, chunk: List[int]) -> List[List[int]]:
        result = []
        for i in range(0, len(chunk), self._chunk_size - self._chunk_overlap):
            new_chunk = chunk[i:i + self._chunk_size]
            if len(new_chunk) > self._chunk_overlap:  # 只有当 chunk 长度大于 overlap 时才添加
                result.append(new_chunk)
        return result
    
    '''
    ### 内部方法 `_enforce_overlap`
    - 确保生成的块之间有适当的重叠，依据 `chunk_overlap` 参数。
    - 对于除第一个块外的每个块，从前一个块末尾取出重叠部分并与当前块合并。
    '''
    def _enforce_overlap(self, chunks: List[List[int]]) -> List[List[int]]:
        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
            else:
                overlap = chunks[i-1][-self._chunk_overlap:]
                new_chunk = overlap + chunk
                if self._length_function(new_chunk) > self._chunk_size:
                    new_chunk = new_chunk[:self._chunk_size]
                result.append(new_chunk)
        return result