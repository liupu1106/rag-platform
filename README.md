# RAG 全流程实操平台

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/liupu1106/rag-platform)

一个**可交互、可拆解、可验证**的 RAG（Retrieval-Augmented Generation）全流程实验平台。
不是又一个「调个 LangChain 就完事」的 demo —— 这里每一层都可替换、每个中间态都可见、每个环节都有量化评测。

零前端构建、零前端依赖，后端仅 Flask + numpy + jieba，**离线可跑通全链路**；接上任意 OpenAI 兼容 / Anthropic 接口即可升级为真实语义检索与流式生成。

> 仓库地址：https://github.com/liupu1106/rag-platform

---

## 快速开始

```bash
pip install -r requirements.txt
python app.py          # 默认 http://127.0.0.1:8080
```

打开浏览器即进入 7 步流水线引导。**不配置任何 Key 也能完整走完全流程**（Embedding 走 TF-IDF，生成走检索增强摘要）。

---

## 流水线：7 步全链路

| # | 步骤 | 做了什么 | 关键可观测物 |
|---|------|---------|-------------|
| 1 | **Load 载入** | 加载内置示例文档 / 上传 txt、md、json、pdf | 文档数、字符数 |
| 2 | **Chunk 切分** | 按 `chunk_size` / `overlap` 滑窗切分 | 每个 chunk 的原文与来源 |
| 3 | **Embed 向量化** | TF-IDF 或真实 Embedding 模型；PCA 降维到 2D | **2D 散点图**（文档向量 + 查询向量同框） |
| 4 | **Retrieve 检索** | 余弦相似度 Top-K；可选 BM25 混合 + RRF 融合；可选 Cross-Encoder Rerank | 命中 chunk、分数、2D 高亮 |
| 5 | **Assemble 组装** | 把 Top-K 拼进 Prompt 模板 | 完整可编辑的 Prompt 原文 |
| 6 | **Generate 生成** | SSE 流式输出（OpenAI / Ollama / Anthropic / 离线摘要） | 逐 token 流式渲染 |
| 7 | **Evaluate 评测** | 6 条标注问答对跑 Recall@K | Recall@K、命中明细表 |

---

## 你能动手调的每一个旋钮

| 旋钮 | 位置 | 说明 |
|------|------|------|
| `chunk_size` / `overlap` | 切分 | 滑窗大小与重叠，直接观察对召回的影响 |
| `top_k` | 检索 | 取回多少条上下文 |
| `hybrid` | 检索 | 开启 **BM25 + 向量，RRF 融合**。验证稀疏与稠密检索的互补性 |
| `rerank` | 检索 | 开启 **Cross-Encoder 重排**（装了 `sentence-transformers` 走真模型，否则退回 BM25 启发式） |
| `rewrite` | 检索前 | **查询改写**，用 LLM 把口语化问题改写成检索友好query |
| `emb_kind` | 向量化 | `tfidf`（离线） / `openai`（真实语义向量） |
| `temperature` | 生成 | 采样温度 |
| `eval_*` | 评测 | 评测用的 top_k / hybrid / rerank 组合 |

**建议的对照实验**：固定 query，分别跑 `baseline` → `+hybrid` → `+rerank` → `+rewrite`，看 Recall@K 与 Top-1 的变化。这是理解 RAG 优化手段性价比最快的方式。

---

## 架构与文件职责

```
rag-platform/
├── app.py            # Flask 主应用：13 个 API 路由 + 内置 FAQ 问答库
├── embed.py          # 向量化：TfidfEmbedder / OpenAIEmbedder / PCA 降维投影
├── retrieve.py       # 检索：余弦相似度 / BM25(Okapi) / RRF 融合 / Rerank
├── generate.py       # 生成：SSE 流式（OpenAI / Ollama / Anthropic）+ 离线摘要
├── rewrite.py        # 查询改写（LLM，未配置则原样返回）
├── datautil.py       # 示例数据、切分逻辑、6 条标注评测集 GOLD
├── evalutil.py       # Recall@K 评测
├── store.py          # 多集合持久化（matrix.npy + meta.json）
├── auth.py           # Bearer Token 鉴权 + 滑动窗口限流
├── static/           # 零依赖前端（原生 HTML/CSS/JS，无构建步骤）
├── data/docs.json    # 内置示例语料（"智学助手"产品文档）
└── tests/            # pytest 11 项，覆盖全链路 + 鉴权
```

### API

| 方法 | 路径 | 用途 |
|------|------|------|
| GET | `/` | 前端页面 |
| GET | `/healthz` | 健康检查（部署平台探活用） |
| GET | `/api/state` | 当前集合、配置、可用集合列表 |
| POST | `/api/config` | 更新 LLM / Embedding 配置 |
| POST | `/api/load` | 载入内置语料并切分 |
| POST | `/api/upload` | 上传 txt / md / json / pdf，建新集合 |
| POST | `/api/embed` | 向量化 + 返回 2D 投影 |
| POST | `/api/project` | 单独投影一个查询点到已有 2D 空间 |
| POST | `/api/query` | 检索（支持 hybrid / rerank / rewrite） |
| POST | `/api/assemble` | 组装 Prompt |
| POST | `/api/generate_stream` | **SSE 流式生成** |
| POST | `/api/eval` | 跑 Recall@K 评测 |
| POST | `/api/ask` | 内置 FAQ 助手（12 条关键词命中） |

---

## 配置（全部走环境变量，不落盘密钥）

复制 `.env.example` 为 `.env` 后按需填写：

| 变量 | 默认值 | 说明 |
|------|-------|------|
| `RAG_LLM_PROVIDER` | `openai` | `openai` / `ollama` / `anthropic` |
| `RAG_LLM_BASE` | 空 | 接口地址；Ollama 填 `http://localhost:11434/v1` |
| `RAG_LLM_KEY` | 空 | LLM API Key |
| `RAG_LLM_MODEL` | `gpt-3.5-turbo` | 模型名 |
| `RAG_TEMPERATURE` | `0.2` | 采样温度 |
| `RAG_EMB_KIND` | `tfidf` | `tfidf` / `openai` |
| `RAG_EMB_BASE` | `https://api.openai.com/v1` | Embedding 接口 |
| `RAG_EMB_KEY` | 空 | Embedding Key |
| `RAG_EMB_MODEL` | `text-embedding-3-small` | Embedding 模型 |
| `RAG_TOKEN` | 空 | **设了就开启 Bearer 鉴权** |
| `RAG_RATE_LIMIT` | `120` | 限流：每窗口请求数 |
| `RAG_RATE_WINDOW` | `60` | 限流窗口（秒） |
| `PORT` | `8080` | 监听端口 |

> 安全设计：密钥**只**从环境变量读取，不写入任何文件、不回显到前端；`store/` 与 `.env` 已在 `.gitignore` 中。

---

## 部署

### Docker（推荐）

```bash
docker build -t rag-platform .
docker run -p 8080:8080 --env-file .env rag-platform
```

多阶段构建，最终镜像以非 root 用户运行，内置 `HEALTHCHECK` 探活 `/healthz`。

### Render（一键）

点上方 **Deploy to Render** 按钮，Render 会自动读取根目录的 `render.yaml` 完成构建配置，
你只需要在向导里填需要的环境变量（`RAG_LLM_KEY` 等，全部可留空，留空则走离线模式）。

### Railway

仓库内已带 `railway.json`，在 Railway 选择 **Deploy from GitHub repo** 并选中本仓库即可，
之后在 Variables 面板填环境变量。

### 其他 PaaS

`Procfile` 已提供：`web: python app.py`。

---

## 测试

```bash
pip install pytest
pytest -q
```

11 项测试覆盖：健康检查、状态与默认集合、载入、向量化与投影、检索 Top-1、hybrid/rerank 开关、Prompt 组装、离线流式生成、Recall@K 评测、文件上传、鉴权拦截。

---

## 设计取舍（给有经验的工程师）

- **为什么默认 TF-IDF 而不是 bge/m3e？** 保证 `pip install` 之后零下载、零 GPU、离线可跑通。真实语义检索是**可选项**而非前置依赖 —— 教学平台的第一优先级是「一定能跑起来」。
- **为什么前端零构建？** 这是一个用来**读源码**的项目。原生 JS 意味着打开 DevTools 就能看懂每一行，不需要先学一套框架。
- **为什么 Rerank 有 BM25 退回？** `sentence-transformers` + `torch` 会把镜像体积推到 GB 级。退回实现让你在没有 GPU 的环境里依然能观察「重排改变了排序」这一**机制**，只是精度打折 —— 机制可见性优先于精度。
- **向量存储为什么是 .npy 而不是 FAISS/Chroma？** 本项目的重点是让你看清「余弦相似度到底怎么算的」。自己用 numpy 实现一遍，比调 `faiss.IndexFlatIP` 学到的多得多。

---

## 已知边界

- 面向**教学与实验**，不是生产级检索系统；语料规模建议控制在数千 chunk 以内。
- `hybrid` 的 RRF 权重固定 `k=60`，未做成可调参数。
- 评测目前只有 Recall@K，**没有**引入 faithfulness / answer relevancy 等 Ragas 指标。

---

## License

MIT
