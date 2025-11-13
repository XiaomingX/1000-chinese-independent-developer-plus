倒数排序融合（RRF）就像一个投票系统，用于合并多个搜索结果，让最重要的信息排在最前面。

**核心思想**

RRF 通过计算每个文档在不同搜索结果中的排名来给文档打分。简单来说，排名越靠前，分数越高。

**公式**

$$
\text{score} = \sum_{q \in \text{queries}} \left( \frac{1}{k + \text{rank}(result(q), d)} \right)
$$

*   `rank(result(q), d)`: 文档 *d* 在某个搜索结果中的排名。
*   *k*:  一个数字（通常设为 60）避免排名低的文档影响太大。
*   `score`: 文档的最终得分。得分越高，排名越靠前。

**工作流程**

1.  **用户提问**：用户输入想搜索的内容。
2.  **多路搜索**：问题同时发送给多个搜索引擎，它们使用不同的方法查找答案。
3.  **各自排名**：每个搜索引擎对找到的文档进行排名。
4.  **RRF 融合**：使用 RRF 算法，将所有排名结果合并成一个。
5.  **最终排名**：根据合并后的分数，生成最终的文档排名，展示给用户。

**RRF 的优势**

*   **简单高效**：容易理解和实现，效果好。
*   **适应性强**：不依赖于特定的评分标准，可以处理各种数据。

**实际应用**

*   **电商网站搜索**：假设你在淘宝搜索“手机”。
    *   一个搜索系统 A 按照销量排名。
    *   另一个搜索系统 B 按照相关度排名。
    *   RRF 将 A 和 B 的结果合并，既考虑了销量，也考虑了相关性，避免只看销量高但不太相关的商品，或者相关但没人买的商品。
*   **新闻聚合**：将多个新闻网站的搜索结果进行合并，避免信息单一。
*   **学术搜索**：合并多个学术数据库的搜索结果，提高查全率。

**代码示例 (Python)**

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    """
    使用倒数排序融合算法合并多个排名列表。

    Args:
        ranked_lists: 一个列表，其中每个元素是一个排名列表（文档ID列表）。
        k: RRF算法中的常量k，用于平滑。

    Returns:
        一个字典，包含每个文档ID的RRF得分。
    """
    document_scores = {}
    for ranked_list in ranked_lists:
        for rank, document_id in enumerate(ranked_list, 1):  # rank 从 1 开始
            if document_id not in document_scores:
                document_scores[document_id] = 0
            document_scores[document_id] += 1 / (k + rank)

    return document_scores

def get_reranked_results(document_scores):
  """根据 RRF 分数对文档进行重新排序。"""
  ranked_results = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
  return ranked_results


# 示例数据：来自两个不同搜索系统的排名列表
ranked_list_1 = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
ranked_list_2 = ['doc2', 'doc4', 'doc6', 'doc1', 'doc7']

# 使用 RRF 融合排名列表
ranked_lists = [ranked_list_1, ranked_list_2]
document_scores = reciprocal_rank_fusion(ranked_lists)

# 打印每个文档的 RRF 得分
print("文档 RRF 得分:")
for document_id, score in document_scores.items():
    print(f"{document_id}: {score:.4f}")

# 重新排序结果
reranked_results = get_reranked_results(document_scores)

# 打印重新排序后的结果
print("\n重新排序后的结果:")
for document_id, score in reranked_results:
    print(f"{document_id}: {score:.4f}")
```

**案例分析**

假设有两个搜索系统，它们对“苹果手机”的搜索结果如下：

*   **系统 A (按销量)**:  [“iPhone 14”, “iPhone 13”, “小米 12”, “华为 P50”]
*   **系统 B (按相关性)**: [“iPhone 14”, “iPhone SE”, “iPhone 13”, “三星 S22”]

使用 RRF 融合后，iPhone 14 会排在最前面，因为它在两个系统中都排名靠前。iPhone 13 也会排在前面，因为它的排名也比较高。 其他品牌的手机，虽然在一个系统中排名靠前，但在另一个系统中排名靠后，所以最终排名会下降。

**数值指标**

RRF 能够提升搜索结果的 *平均精度均值* (Mean Average Precision, MAP) 和 *归一化折损累计增益* (Normalized Discounted Cumulative Gain, NDCG)。  例如，在某些实验中，使用 RRF 后，MAP 值可以提升 5%-10%，NDCG 值可以提升 3%-8%。

**总结**

RRF 通过简单的算法，有效地融合多个搜索结果，提高搜索质量，让用户更快找到想要的信息。