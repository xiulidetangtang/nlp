# nlp
this is for Fudan nlp final task

# 英语写作评分系统

基于DeBERTa+BiLSTM的英语作文自动评分系统，提供六维度评分和AI反馈。

## 快速开始

### 环境准备
```bash
pip install flask flask-cors torch transformers openai numpy
```
### 启动后端
```
python app.py
```
### 启动前段
```
python -m htttp.server 8080
```
### 打开网页
```
http://localhost:8080
```
### 配置api
在app.py 中修改api-key，就可以实现对大模型的调用
