"""
Math RAG (Retrieval-Augmented Generation) Module
从训练集中检索相似的数学题目作为 few-shot examples

支持两种检索方式:
1. BM25: 基于关键词的稀疏检索 (快速, 无需GPU)
2. Embedding: 基于语义的稠密检索 (更准确, 需要额外模型)

使用方式:
    retriever = MathRAGRetriever(dataset_name="gsm8k", method="bm25")
    examples = retriever.retrieve(question, top_k=3)
"""

import os
import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RetrievedExample:
    """检索到的例题"""
    question: str
    solution: str
    answer: str
    score: float
    index: int


class MathRAGRetriever:
    """
    数学题目检索器
    从训练集中检索相似题目作为 few-shot examples
    """
    
    def __init__(
        self,
        dataset_name: str = "gsm8k",
        method: str = "bm25",
        base_path: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        初始化检索器
        
        Args:
            dataset_name: 数据集名称 (gsm8k, math500)
            method: 检索方法 (bm25, embedding)
            base_path: 项目根目录
            cache_dir: 缓存目录
        """
        self.dataset_name = dataset_name
        self.method = method
        
        if base_path is None:
            base_path = str(Path(__file__).parent.parent)
        self.base_path = Path(base_path)
        
        if cache_dir is None:
            cache_dir = self.base_path / "cache" / "rag"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 知识库数据
        self.documents: List[Dict] = []
        self.index = None
        
        # 加载数据和构建索引
        self._load_knowledge_base()
        self._build_index()
    
    def _load_knowledge_base(self):
        """加载训练集作为知识库"""
        from datasets import load_from_disk
        
        dataset_path = self.base_path / "data" / self.dataset_name
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        dataset = load_from_disk(str(dataset_path))
        
        # 使用训练集作为知识库
        if "train" in dataset:
            train_data = dataset["train"]
        else:
            # 如果没有训练集，使用测试集的一部分 (不推荐)
            print(f"WARNING: No train split found for {self.dataset_name}, using test split")
            train_data = dataset["test"]
        
        # 解析数据
        for idx, example in enumerate(train_data):
            doc = self._parse_example(example, idx)
            if doc:
                self.documents.append(doc)
        
        print(f"Loaded {len(self.documents)} examples from {self.dataset_name} train set")
    
    def _parse_example(self, example: dict, idx: int) -> Optional[Dict]:
        """解析单个样本"""
        if self.dataset_name == "gsm8k":
            question = example.get("question", "")
            full_answer = example.get("answer", "")
            
            # GSM8K格式: solution #### answer
            if "####" in full_answer:
                parts = full_answer.split("####")
                solution = parts[0].strip()
                answer = parts[1].strip()
            else:
                solution = full_answer
                answer = ""
            
            return {
                "index": idx,
                "question": question,
                "solution": solution,
                "answer": answer,
                "text": question  # 用于检索的文本
            }
        
        elif self.dataset_name in ["math", "math500"]:
            question = example.get("problem", "")
            solution = example.get("solution", "")
            
            # 提取 \boxed{} 中的答案
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution)
            if boxed_match:
                answer = boxed_match.group(1)
            else:
                answer = ""
            
            return {
                "index": idx,
                "question": question,
                "solution": solution,
                "answer": answer,
                "text": question
            }
        
        return None
    
    def _build_index(self):
        """构建检索索引"""
        cache_file = self.cache_dir / f"{self.dataset_name}_{self.method}_index.pkl"
        
        # 尝试加载缓存
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self.index = pickle.load(f)
                print(f"Loaded cached {self.method} index from {cache_file}")
                return
            except Exception as e:
                print(f"Failed to load cache: {e}, rebuilding index...")
        
        # 构建新索引
        if self.method == "bm25":
            self._build_bm25_index()
        elif self.method == "embedding":
            self._build_embedding_index()
        else:
            raise ValueError(f"Unknown retrieval method: {self.method}")
        
        # 保存缓存
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self.index, f)
            print(f"Saved {self.method} index to {cache_file}")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def _build_bm25_index(self):
        """构建 BM25 索引"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Please install rank_bm25: pip install rank-bm25")
        
        # 分词 (简单的空格分词 + 小写)
        tokenized_docs = []
        for doc in self.documents:
            tokens = self._tokenize(doc["text"])
            tokenized_docs.append(tokens)
        
        self.index = BM25Okapi(tokenized_docs)
        print(f"Built BM25 index with {len(tokenized_docs)} documents")
    
    def _build_embedding_index(self):
        """构建 Embedding 索引"""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        
        # 使用轻量级模型
        model_name = "all-MiniLM-L6-v2"
        print(f"Loading embedding model: {model_name}")
        
        encoder = SentenceTransformer(model_name)
        
        # 编码所有文档
        texts = [doc["text"] for doc in self.documents]
        embeddings = encoder.encode(texts, show_progress_bar=True)
        
        self.index = {
            "encoder": encoder,
            "embeddings": embeddings
        }
        print(f"Built embedding index with {len(embeddings)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        # 转小写
        text = text.lower()
        # 移除标点 (保留数字)
        text = re.sub(r'[^\w\s]', ' ', text)
        # 分词
        tokens = text.split()
        # 移除停用词 (简单版本)
        stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'above',
                     'below', 'between', 'under', 'again', 'further', 'then', 'once',
                     'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
                     'not', 'only', 'own', 'same', 'than', 'too', 'very', 's', 't',
                     'can', 'just', 'don', 'now', 'her', 'his', 'she', 'he', 'it', 'they'}
        tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
        return tokens
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        exclude_indices: Optional[List[int]] = None
    ) -> List[RetrievedExample]:
        """
        检索相似题目
        
        Args:
            query: 查询问题
            top_k: 返回的例题数量
            exclude_indices: 要排除的索引 (避免检索到测试题本身)
        
        Returns:
            检索到的例题列表
        """
        if exclude_indices is None:
            exclude_indices = []
        
        if self.method == "bm25":
            return self._retrieve_bm25(query, top_k, exclude_indices)
        elif self.method == "embedding":
            return self._retrieve_embedding(query, top_k, exclude_indices)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _retrieve_bm25(
        self,
        query: str,
        top_k: int,
        exclude_indices: List[int]
    ) -> List[RetrievedExample]:
        """BM25 检索"""
        query_tokens = self._tokenize(query)
        scores = self.index.get_scores(query_tokens)
        
        # 排序并过滤
        scored_indices = [(i, s) for i, s in enumerate(scores) if i not in exclude_indices]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        # 取 top-k
        results = []
        for idx, score in scored_indices[:top_k]:
            doc = self.documents[idx]
            results.append(RetrievedExample(
                question=doc["question"],
                solution=doc["solution"],
                answer=doc["answer"],
                score=float(score),
                index=doc["index"]
            ))
        
        return results
    
    def _retrieve_embedding(
        self,
        query: str,
        top_k: int,
        exclude_indices: List[int]
    ) -> List[RetrievedExample]:
        """Embedding 检索"""
        import numpy as np
        
        encoder = self.index["encoder"]
        doc_embeddings = self.index["embeddings"]
        
        # 编码查询
        query_embedding = encoder.encode([query])[0]
        
        # 计算相似度 (余弦相似度)
        similarities = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # 排序并过滤
        scored_indices = [(i, s) for i, s in enumerate(similarities) if i not in exclude_indices]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        # 取 top-k
        results = []
        for idx, score in scored_indices[:top_k]:
            doc = self.documents[idx]
            results.append(RetrievedExample(
                question=doc["question"],
                solution=doc["solution"],
                answer=doc["answer"],
                score=float(score),
                index=doc["index"]
            ))
        
        return results
    
    def format_examples_as_prompt(
        self,
        examples: List[RetrievedExample],
        include_solution: bool = True,
        max_solution_length: int = 500
    ) -> str:
        """
        将检索到的例题格式化为 few-shot prompt
        
        Args:
            examples: 检索到的例题列表
            include_solution: 是否包含解题过程
            max_solution_length: 解题过程的最大长度
        
        Returns:
            格式化的 few-shot prompt
        """
        if not examples:
            return ""
        
        prompt_parts = ["Here are some similar problems and their solutions:\n"]
        
        for i, ex in enumerate(examples, 1):
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(f"Problem: {ex.question}")
            
            if include_solution:
                solution = ex.solution
                if len(solution) > max_solution_length:
                    solution = solution[:max_solution_length] + "..."
                prompt_parts.append(f"Solution: {solution}")
            
            prompt_parts.append(f"Answer: {ex.answer}")
            prompt_parts.append("")  # 空行分隔
        
        prompt_parts.append("Now solve this problem:\n")
        
        return "\n".join(prompt_parts)


# 便捷函数
def create_retriever(
    dataset_name: str = "gsm8k",
    method: str = "bm25",
    base_path: Optional[str] = None
) -> MathRAGRetriever:
    """创建检索器的便捷函数"""
    return MathRAGRetriever(
        dataset_name=dataset_name,
        method=method,
        base_path=base_path
    )


# 测试代码
if __name__ == "__main__":
    print("=" * 80)
    print("Math RAG Retriever - Test")
    print("=" * 80)
    
    # 创建检索器
    retriever = MathRAGRetriever(dataset_name="gsm8k", method="bm25")
    
    # 测试查询
    test_query = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    
    print(f"\nQuery: {test_query[:100]}...")
    print("\n" + "-" * 40)
    
    # 检索
    results = retriever.retrieve(test_query, top_k=3)
    
    print(f"\nTop {len(results)} similar examples:")
    for i, ex in enumerate(results, 1):
        print(f"\n[Example {i}] (score: {ex.score:.4f})")
        print(f"Question: {ex.question[:100]}...")
        print(f"Answer: {ex.answer}")
    
    # 格式化为 prompt
    print("\n" + "=" * 80)
    print("Formatted Few-Shot Prompt:")
    print("=" * 80)
    prompt = retriever.format_examples_as_prompt(results, include_solution=True, max_solution_length=200)
    print(prompt)














