# 智扫通机器人智能客服

一个面向扫地机器人和扫拖一体机器人的智能客服项目。项目使用 LangChain 构建带工具调用能力的 Agent，通过 Chroma 向量库检索本地产品资料，并使用 Streamlit 提供多轮聊天界面。

## 主要功能

- 基于本地 PDF、TXT 资料回答选购、使用、维护和故障排查问题
- 支持同一页面会话中的连续追问
- 根据用户提供的城市查询实时天气，并给出机器人使用建议
- 根据演示用户记录生成月度使用报告
- 记录模型调用、工具调用和异常日志

## 技术栈

- Python 3.11
- Streamlit
- LangChain / LangGraph Agent
- 通义千问（DashScope）
- Chroma
- Open-Meteo

## 项目结构

```text
Agent/
├── agent/                  # Agent 和工具调用逻辑
│   ├── react_agent.py
│   └── tools/
├── config/                 # 模型、提示词和向量库配置
├── data/                   # 本地知识库及演示业务数据
├── model/                  # 聊天模型和 Embedding 模型创建
├── prompts/                # 主提示词、RAG 提示词和报告提示词
├── rag/                    # 文档入库、检索和答案生成
├── utils/                  # 配置、文件、日志和路径工具
├── .env.example            # 环境变量示例
├── .gitignore
├── app.py                  # Streamlit 应用入口
├── requirements.txt       # Python 依赖清单
└── README.md
```

## 本地运行

### 1. 获取项目

```powershell
git clone <你的 GitHub 仓库地址>
cd Agent
```

如果已经在本地打开本项目，可以跳过这一步。

### 2. 创建虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止激活脚本，也可以继续使用已有的 Conda 环境。

### 3. 安装依赖

```powershell
python -m pip install -r requirements.txt
```

### 4. 配置 DashScope 密钥

复制环境变量示例：

```powershell
Copy-Item .env.example .env
```

打开 `.env`，填写自己的 DashScope API 密钥：

```dotenv
DASHSCOPE_API_KEY=你的密钥
```

`.env` 已被 `.gitignore` 忽略，不应提交到 GitHub。`.env.example` 只保留变量名，不应填写真实密钥。

### 5. 初始化本地知识库

```powershell
python rag/vector_store.py
```

程序会读取 `data/` 下的 PDF 和 TXT 文件，将分片写入本地 `chroma_db/`。已经处理过的文件会通过 MD5 记录跳过。

### 6. 启动应用

```powershell
python -m streamlit run app.py
```

终端显示访问地址后，在浏览器打开该地址即可使用。

## 演示问题

知识库问答：

```text
扫地机器人的滤网应该怎么保养？
```

收到回答后继续追问：

```text
多久清理一次？
```

天气场景：

```text
深圳现在的湿度适合开启拖地功能吗？
```

报告场景：

```text
帮我生成使用报告
```

## 配置说明

主要配置文件如下：

| 文件 | 作用 |
| --- | --- |
| `config/rag.yml` | 聊天模型和 Embedding 模型名称 |
| `config/chroma.yml` | 知识库路径、分片参数和召回数量 |
| `config/prompts.yml` | 提示词文件路径 |
| `config/agent.yml` | 演示业务数据路径 |

## 当前限制

- 城市由用户在对话中提供，项目不会自动读取浏览器位置。
- 月度报告使用本地 CSV 演示数据，当前演示用户为 `1001`，演示月份为 `2025-01`。
- 当前知识库采用固定长度分片和向量 Top-K 检索，尚未加入混合检索和重排序。
- 聊天历史只保存在当前 Streamlit 会话中，刷新会话或重启服务后不会持久化。
- 天气查询依赖外部 Open-Meteo 服务和网络连接。

## 安全说明

- 不要在代码、YAML、日志或提交记录中保存真实 API 密钥。
- 如果密钥曾经提交到公开仓库，应立即在服务商后台撤销并重新生成。
- 提交代码前，可以通过 `git status --short` 确认 `.env` 没有进入暂存区。

## 后续计划

- 建立人工审核的 RAG 评测集
- 优化知识分片并保留文档来源信息
- 对比向量检索与 BM25 混合检索
- 为回答增加可追溯引用和无答案处理
- 增加工具调用次数、超时和错误状态控制
