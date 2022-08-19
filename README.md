# prompt-learning

## Requires
python >= 3.7.0
numpy >= 1.21.5
pandas >= 1.3.4
pytorch >= 1.9.0
transformers >= 4.15.0
pyprind == 2.11.3 （可以不要，如果不需要的话将相关的两行代码删除就行）

## Usage
### bert-base-uncased
- 如果你希望放在本地，那么将 bert-base-uncased (https://huggingface.co/bert-base-uncased) 放在 `static_data` 目录下。
- 如果你不想下载到本地，那么将 `prompt.py` 第32行替换为：
```python
model_path = 'bert-base-uncased'
```

### 数据
打开 `static_data`，将 `IMDB.rar` 解压到本目录下即可。

### 运行
可以直接编译器运行 `prompt.py`，也可以在命令行使用：
```python
cd run_models
python prompt.py
```
