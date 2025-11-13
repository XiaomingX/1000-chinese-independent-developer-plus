import numpy as np

# 链接关系，用邻接矩阵表示
# links[i][j] = 1 表示网页 i 链接到网页 j
links = np.array([
    [0, 1, 1, 0],  # A 链接到 B 和 C
    [0, 0, 1, 0],  # B 链接到 C
    [1, 0, 0, 0],  # C 链接到 A
    [0, 0, 1, 0]   # D 链接到 C
])

# 网页总数
N = links.shape[0]

# 阻尼系数
d = 0.85

# 初始化 PageRank 值
pagerank = np.ones(N) / N

# 迭代计算
for _ in range(100):
    new_pagerank = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if links[j][i] == 1:
                # 计算网页 j 的出链数
                out_degree = np.sum(links[j])
                new_pagerank[i] += pagerank[j] / out_degree
    new_pagerank = (1 - d) / N + d * new_pagerank
    pagerank = new_pagerank

print(pagerank)