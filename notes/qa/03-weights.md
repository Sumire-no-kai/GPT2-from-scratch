---
title: 阶段 3：加载预训练权重问答
summary: 参数映射、权重布局与参考输出对照中的问题与回答
date: 2026-10-10
tags: [GPT-2]
status: draft
---

# 阶段 3：加载预训练权重问答

范围：参数映射、权重布局与参考输出对照。结构与阶段 0 相同：导读、进度与成果、全景、按代码顺序的各节（概念、代码逐行讲解、测试、踩过的坑、相关问答）、回顾、延伸问答、术语表。

## 加载共享的 token 表

阶段 2 的 `wte` 和 `lm_head` 引用同一个 `Parameter`。但 `state_dict()` 按模块路径记录权重，因此会列出两个键。`embeddings.wte.weight` 和 `lm_head.weight` 共用同一份张量存储。[阶段 2 的权重共享](02-model.md#初始化如何装好完整模型)

- 若组装字典后调用 `load_state_dict(..., strict=True)`，把两个键都映射到官方的同一份 `wte` 张量，并保留默认的 `assign=False`。只提供一个键时，严格加载仍会报告另一个键缺失。
- 若逐参数加载，用 `torch.no_grad()` 包住 `model.embeddings.wte.weight.copy_(official_wte)`，原地拷贝一次即可。`lm_head` 也会读到新的数值。
- 加载后同时检查数值，以及两者是否仍是同一个参数对象：`model.lm_head.weight is model.embeddings.wte.weight` 应为 `True`。

## 延伸问答

