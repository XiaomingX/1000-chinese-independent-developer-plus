## SynCheck：提升检索增强语言模型（RALM）可靠性的简单方法

SynCheck 是一种由加州大学洛杉矶分校 (UCLA) 研究人员开发的轻量级工具，旨在解决大型语言模型在生成内容时出现的可靠性问题，比如产生没有依据的信息或与检索到的上下文相矛盾的内容。SynCheck 通过在生成过程中同步检测不可靠的句子来实现这一目标。

**工作原理**

SynCheck 的核心在于实时监控大型语言模型的解码过程，主要分为两个部分：同步可靠性监控和可靠性导向解码。

**同步可靠性监控**

通过整合多个信号来评估每个句子的可靠性。关键信号包括：

*   **可能性 (Likelihood)**：计算句子的最小可能性和长度归一化可能性，以检测知识盲点。可能性越低，说明模型在生成响应时缺乏足够的知识支持。

    *   *实际应用举例：* 假设用 RALM 回答“iPhone 15 什么时候发布？”这个问题。如果模型生成的句子是“iPhone 15 在 1888 年发布”，因为时间上明显错误，这个句子的可能性就会非常低，SynCheck 就能检测出来。

    *   *Demo代码：*
        ```python
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        def calculate_likelihood(sentence):
          """计算句子的可能性"""
          inputs = tokenizer(sentence, return_tensors="pt")
          with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
          loss = outputs.loss
          return torch.exp(-loss).item()
        
        sentence1 = "iPhone 15 was released in September 2023."
        sentence2 = "iPhone 15 was released in 1888."
        
        likelihood1 = calculate_likelihood(sentence1)
        likelihood2 = calculate_likelihood(sentence2)
        
        print(f"'{sentence1}' 的可能性: {likelihood1}")
        print(f"'{sentence2}' 的可能性: {likelihood2}")
        ```

*   **不确定性 (Uncertainty)**：监控句子中每个 token 的熵以及中间层激活的局部内在维度，以捕捉模型在生成文本时的不确定性。熵越高，表示模型对生成的内容越不确定。

    *   *实际应用举例：* 如果模型生成的句子是“据我所知，iPhone 15 *可能* 在九月发布”，"可能"这个词表达了一种不确定性，SynCheck 会检测到这种不确定性。

*   **上下文影响 (Context Influence)**：通过计算包含上下文的分布和不包含上下文的分布之间的 Kullback-Leibler 散度，评估模型对检索到的上下文的依赖程度。KL 散度越大，说明模型对上下文的依赖程度越高。

    *   *实际应用举例：* 假设检索到的上下文是 "iPhone 15 发布于 2023 年 9 月 22 日"。如果模型生成的句子完全没有提及这个信息，或者生成了与此相反的信息，SynCheck 会检测到上下文影响不足。

*   **语义对齐 (Semantic Alignment)**：使用轻量级蕴含检查器来评估生成的句子是否在语义上与检索到的上下文一致。

    *   *实际应用举例：* 假设检索到的上下文是 "iPhone 15 具有 A17 芯片"。如果模型生成的句子是 "iPhone 15 采用了骁龙 8 Gen 3 芯片"，这在语义上是冲突的，SynCheck 能够检测出来。

**可靠性导向解码**

利用 SynCheck 提供的实时监控信号来指导解码过程，提高输出的可靠性。引入了面向可靠性的大模型解码算法 (FOD)，包括贪婪搜索和回溯以及可靠性引导的束搜索两个阶段。简单来说，就是根据 SynCheck 的评分，选择更可靠的生成结果。

**实验与评估**

实验结果表明，SynCheck 在可靠性检测方面优于其他方法，通过与 SPANEXTRACT、CRITICTOK、FLARE、ALIGNSCORE 和 MINICHECK 等基线对比，证明了其有效性。例如，在某个benchmark上，SynCheck 将生成内容的可靠性提高了 15%。