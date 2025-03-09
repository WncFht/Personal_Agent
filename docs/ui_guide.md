# RSS-RAG Gradio界面使用指南

本文档介绍如何使用RSS-RAG系统的Gradio界面，包括安装、配置和使用方法。

## 安装依赖

在使用Gradio界面前，请确保已安装所有必要的依赖：

```bash
pip install -r requirements.txt
pip install gradio pandas matplotlib
```

## 启动界面

有两种方式启动Gradio界面：

### 1. 使用启动脚本

```bash
python run_ui.py
```

可用的命令行参数：

- `--port`: 指定服务端口（默认：7860）
- `--host`: 指定服务主机（默认：0.0.0.0）
- `--share`: 创建公共链接，可通过互联网访问
- `--debug`: 启用调试模式

例如：

```bash
# 在8080端口启动，并创建公共链接
python run_ui.py --port 8080 --share

# 启用调试模式
python run_ui.py --debug
```

### 2. 直接运行app.py

```bash
python src/ui/app.py
```

## 界面功能

Gradio界面分为三个主要标签页：

### 1. 智能问答

功能：
- 输入问题并获取基于RSS数据的智能回答
- 可选择特定RSS源或时间范围
- 支持显示参考信息
- 支持流式输出（打字机效果）

使用方法：
1. 在文本框中输入您的问题
2. 可选择特定RSS源（默认为全部）
3. 可设置时间范围（默认为30天）
4. 点击"提问"按钮获取回答

### 2. RSS源管理

包含四个子标签页：

#### 2.1 添加RSS源

功能：
- 添加单个RSS源
- 可指定分类

使用方法：
1. 输入RSS源URL
2. 可选择输入分类
3. 点击"添加"按钮

#### 2.2 导入OPML

功能：
- 批量导入OPML格式的RSS源列表

使用方法：
1. 上传OPML文件
2. 点击"导入"按钮

#### 2.3 管理RSS源

功能：
- 删除RSS源
- 更新所有RSS源

使用方法：
1. 从下拉菜单选择要删除的RSS源
2. 点击"删除"按钮
3. 或点击"更新所有RSS源"按钮获取最新内容

#### 2.4 RSS源统计

功能：
- 显示RSS源统计信息
- 生成可视化图表

使用方法：
1. 点击"生成统计"按钮
2. 查看统计表格和图表

### 3. 系统配置

功能：
- 配置系统参数
- 选择LLM模型
- 设置API密钥
- 调整RAG参数

使用方法：
1. 调整各项配置参数
2. 点击"保存配置"按钮应用更改

配置项包括：
- 基础配置：设备、数据目录
- LLM配置：模型类型、模型路径、系统提示词
- API配置：OpenAI和DeepSeek API密钥
- RAG配置：Embedding模型、分块大小、检索结果数量等
- 高级选项：重排序、查询增强、缓存等

## 常见问题

### 1. 界面无法启动

检查：
- 是否已安装所有依赖
- 端口是否被占用
- 日志输出是否有错误信息

### 2. 无法添加RSS源

可能原因：
- RSS源URL格式不正确
- 网络连接问题
- RSS源需要特殊处理（如需要Cookie或JavaScript）

### 3. 问答结果不准确

改进方法：
- 添加更多相关RSS源
- 调整系统配置中的RAG参数
- 尝试使用不同的LLM模型

### 4. 系统响应缓慢

优化方法：
- 减少RSS源数量
- 调整分块大小和重叠
- 使用更轻量级的LLM模型
- 启用缓存功能

## 高级使用

### 自定义配置文件

系统配置保存在`config/app_config.json`文件中，您可以直接编辑此文件进行高级配置。

### 使用DeepSeek模型

1. 在系统配置中选择"deepseek"作为LLM类型
2. 输入您的DeepSeek API密钥
3. 选择模型（deepseek-chat或deepseek-reasoner）
4. 保存配置

### 批量导入RSS源

准备OPML文件示例：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<opml version="2.0">
  <head>
    <title>RSS订阅列表</title>
  </head>
  <body>
    <outline text="科技" title="科技">
      <outline text="36氪" title="36氪" type="rss" xmlUrl="https://36kr.com/feed" />
      <outline text="少数派" title="少数派" type="rss" xmlUrl="https://sspai.com/feed" />
    </outline>
    <outline text="新闻" title="新闻">
      <outline text="BBC中文网" title="BBC中文网" type="rss" xmlUrl="https://www.bbc.co.uk/zhongwen/simp/index.xml" />
    </outline>
  </body>
</opml>
```

然后在"导入OPML"标签页上传此文件。

## 参考资料

- [Gradio文档](https://www.gradio.app/docs/)
- [RSS-RAG项目文档](../README.md)
- [DeepSeek模型使用指南](./deepseek_usage.md) 