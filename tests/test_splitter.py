import unittest
from typing import List
import tiktoken
from openbayes_rag.splitter import DocumentSeparatorSplitter


class TestDocumentSeparatorSplitter(unittest.TestCase):

    def setUp(self):
        self.tokenize = lambda text: [
            ord(c) for c in text
        ]  # Simple tokenizer for testing
        self.detokenize = lambda tokens: "".join(chr(t) for t in tokens)

    def test_split_with_custom_separator(self):
        splitter = DocumentSeparatorSplitter(
            separators=[self.tokenize("\n"), self.tokenize(".")],
            chunk_size=19,
            chunk_overlap=0,
            keep_separator="end",
        )
        text = "This is a test.\nAnother test."
        tokens = self.tokenize(text)
        expected = [
            self.tokenize("This is a test.\n"),
            self.tokenize("Another test."),
        ]
        result = splitter.split_tokens_with_separator(tokens)

        self.assertEqual(result, expected)
        
    def test_split_simple(self):
        splitter = DocumentSeparatorSplitter(
            separators=[self.tokenize("\n"), self.tokenize(".")],
            chunk_size=16,
            chunk_overlap=0,
            keep_separator="end",
        )
        text = "This is a test.\nAnother test."
        tokens = self.tokenize(text)
        expected = [
            self.tokenize("This is a test.\n"),
            self.tokenize("Another test."),
        ]
        result = splitter.split_tokens(tokens)

        self.assertEqual(result, expected)

    def test_chunk_size_limit(self):
        splitter = DocumentSeparatorSplitter(
            chunk_size=5, chunk_overlap=0, separators=[self.tokenize("\n")]
        )
        text = "1234567890"
        tokens = self.tokenize(text)
        expected = [self.tokenize("12345"), self.tokenize("67890")]
        result = splitter.split_tokens(tokens)
        self.assertEqual(result, expected)

    def test_chunk_overlap(self):
        splitter = DocumentSeparatorSplitter(
            chunk_size=5, chunk_overlap=2, separators=[self.tokenize("\n")]
        )
        text = "1234567890"
        tokens = self.tokenize(text)
        expected = [
            self.tokenize("12345"),
            self.tokenize("45678"),
            self.tokenize("7890"),
        ]
        result = splitter.split_tokens(tokens)
        self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()




'''
上面的 Python 代码是一个使用 `unittest` 框架编写的单元测试模块，旨在测试一个名为 `DocumentSeparatorSplitter` 的文档分割器类。这个类似乎专门用于将文本分割成特定大小的块，并考虑到指定的分隔符和块之间的重叠。下面是对代码的详细解释：

### 导入模块
- `unittest`: Python 的标准库中的单元测试框架。
- `List`: 从 `typing` 模块导入，用于类型注解。
- `tiktoken` 和 `openbayes_rag.splitter`: 这些看起来是自定义的模块，可能包含了 `DocumentSeparatorSplitter` 类的定义。

### 类 `TestDocumentSeparatorSplitter`
这是继承自 `unittest.TestCase` 的测试类，用于定义一系列针对 `DocumentSeparatorSplitter` 类的单元测试。

#### 方法 `setUp`
这是 `unittest` 框架中的一个特殊方法，每次执行测试方法前都会被调用。在这个方法中，定义了：
- `self.tokenize`: 一个简单的函数，将字符串转换为其对应的 ASCII 值列表。
- `self.detokenize`: 一个将 ASCII 值列表转换回字符串的函数。

#### 测试方法
1. **`test_split_with_custom_separator`**:
   - 创建一个 `DocumentSeparatorSplitter` 实例，分隔符设置为换行符和句点，块大小为19，不重叠，并保留分隔符在末尾。
   - 定义一个测试文本 `"This is a test.\nAnother test."` 并使用 `tokenize` 转换为 tokens。
   - 定义期望的结果，即文本按分隔符正确分割的 tokens 列表。
   - 使用 `split_tokens` 方法进行分割，然后断言结果与期望的结果相匹配。

2. **`test_chunk_size_limit`**:
   - 创建一个 `DocumentSeparatorSplitter` 实例，只使用换行符作为分隔符，块大小为5，没有重叠。
   - 定义一个测试文本 `"1234567890"` 并转换为 tokens。
   - 定义期望的结果，即按照块大小限制分割的 tokens 列表。
   - 使用 `split_tokens` 方法进行分割，然后断言结果与期望的结果相匹配。

3. **`test_chunk_overlap`**:
   - 创建一个 `DocumentSeparatorSplitter` 实例，块大小为5，重叠为2，使用换行符作为分隔符。
   - 定义一个测试文本 `"1234567890"` 并转换为 tokens。
   - 定义期望的结果，即按照块大小和重叠分割的 tokens 列表。
   - 使用 `split_tokens` 方法进行分割，然后断言结果与期望的结果相匹配。

### 运行测试
在文件的最后，如果这个脚本作为主程序运行，将调用 `unittest.main()` 来执行所有的测试方法。

这个测试模块的目的是确保 `DocumentSeparatorSplitter` 类能够按照预期处理不同的分割情况，包括处理不同的分隔符、遵守块大小限制，以及正确应用块之间的重叠。
'''
