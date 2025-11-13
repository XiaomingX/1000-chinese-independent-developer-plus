文本迁移学习就像**让一位经验丰富的老师（预训练模型）快速掌握新知识（新任务）**。它利用已有的语言知识来理解和处理新的文本任务，即使新任务的训练数据不多，也能取得不错的成果。

**核心思想**：利用在大规模文本数据上预先训练好的模型，让它把学到的通用语言知识应用到新的任务上。

**步骤**：

1.  **选择预训练模型**：选择像BERT、GPT这样已经在大规模文本上训练过的模型。中文任务选用中文BERT或RoBERTa，英文任务则选择英文BERT或GPT。
2.  **添加任务层**：在预训练模型的基础上，根据你的具体任务（比如分类、回归、标注）添加相应的层。例如，文本分类可以加一个全连接层，命名实体识别可以加一个CRF层。
3.  **微调模型**：用你的任务数据来调整预训练模型和新加的层，让模型适应你的任务。调整学习率和batch size等参数来控制调整的幅度和速度。
4.  **评估测试**：用测试数据评估模型效果，并根据结果调整参数或微调策略。关注准确率、精确率、召回率、F1值等指标。

**代码示例（使用Transformers库）**

以下是一个使用BERT模型进行情感分类的例子，判断一段文本是积极还是消极：

```python
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

# 1. 加载预训练的 BERT 模型和 Tokenizer
model_name = 'bert-base-chinese'  # 选择中文 BERT 模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # num_labels=2 表示 2 个类别（积极和消极）

# 2. 准备数据集
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 假设你已经有了训练数据集 train_texts, train_labels 和测试数据集 test_texts, test_labels
# 例如:
train_texts = ["这部电影太棒了！", "我非常喜欢这个产品。", "剧情有点无聊。", "不太推荐这家餐厅。"]
train_labels = [1, 1, 0, 0]  # 1: 积极, 0: 消极
test_texts = ["期待下一部！", "体验很差。", "还不错，可以试试。", "非常糟糕！"]
test_labels = [1, 0, 1, 0]

MAX_LEN = 128  # 设置最大文本长度
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)  # 使用 AdamW 优化器

# 4. 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 5. 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test texts: {100 * correct / total:.2f}%')
```

**实际应用案例**

*   **情感分析**：用BERT预训练模型，再用少量电影评论微调，可以高精度地判断评论的情感倾向（积极或消极）。在IMDB或ChnSentiCorp等数据集上，微调后的BERT模型准确率可达90%以上。
*   **文本分类**：使用RoBERTa模型，用少量新闻数据微调，可以将新闻分成体育、娱乐、科技等类别，准确率可达95%以上（在路透社新闻分类数据集上）。
*   **命名实体识别**：使用BERT模型，用少量医学文本微调，可以识别医学文本中的疾病、药物等实体，F1值可达90%以上（在CoNLL-2003数据集上）。

**小技巧**

*   **学习率调整**：采用学习率衰减策略（如线性衰减或余弦退火），避免训练初期震荡，并加速后期收敛。
*   **梯度裁剪**：使用梯度裁剪防止梯度爆炸，提高训练稳定性。
*   **数据增强**：使用随机替换、删除或插入词语等数据增强方法，增加数据多样性，提高模型泛化能力。

总而言之，迁移学习是一种强大的文本处理技术，它可以帮助我们利用预训练模型的知识，高效地解决各种文本任务，特别是在数据量有限的情况下。