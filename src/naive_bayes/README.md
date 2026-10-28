# Member 2 垃圾邮件检测

本包包含 TF-IDF 与 Multinomial Naive Bayes 的训练代码、已训练模型、
验证结果和普通版英文报告。使用 Member 1 提供的原始训练/验证划分，
不改动 CSV 内容。所有分数均来自验证集；测试集由 Member 5 保管并统一评估。

## 1 先看什么

1. 阅读 `Member2_Report.docx`，了解方法、结果和限制。
2. 按下面的命令试运行预测，确认本机环境正常。
3. 阅读 `preprocessing.py` 和 `train_model_1.py`，理解自己的代码。
4. 需要复现实验时再重新训练；保存好的模型可以直接使用。

## 2 安装和运行

使用 Python 3.12（实际运行版本为 3.12.14），解压后在本 README 所在文件夹
打开 PowerShell，依次运行。无需激活虚拟环境。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe predict.py --text "Subject: Team meeting. Please review the project agenda."
```

如果没有 `py` 命令，用已安装的 Python 3.12 的完整路径替换它。
在 macOS/Linux 上用 `python3.12 -m venv .venv`，后续解释器路径改为 `.venv/bin/python`。

重新训练并比较全部 32 组参数：

```powershell
.\.venv\Scripts\python.exe train_model_1.py
```

检查代码及保存的模型：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe verify_saved_model.py
```

训练会覆盖 `artifacts/` 中的结果；复核命令会更新 `verification.json`。
本次整理已经实际重跑全部实验并复核，报告中的耗时对应本次运行。
以后重跑的耗时可能不同；更换数据时要统一全组数据版本、更新
`input_manifest.json` 并重新生成实验记录和报告。

## 3 文件用途

| 文件或目录 | 用途 |
| --- | --- |
| `preprocessing.py` | 还原邮件正文，规范化文本，创建 TF-IDF |
| `train_model_1.py` | 训练、调参、选模型、导出结果和图表 |
| `predict.py` | 预测接口，供 Member 6 集成 |
| `verify_saved_model.py` | 独立复算指标，核对训练来源和模型文件 |
| `tests/test_member2.py` | 六项代码测试，使用合成样本，不使用测试集 |
| `requirements.txt` | 锁定实际运行环境中的库版本，包括间接依赖 |
| `data/train.csv`、`data/validation.csv` | 复现实验所需的两份原始数据 |
| `input_manifest.json` | 这两份输入文件的大小和 SHA256 |
| `Member2_Report.docx` | 普通黑白版英文报告 |
| `artifacts/naive_bayes_pipeline.joblib` | 完整模型，已包含文本处理和 TF-IDF |
| `artifacts/experiments.csv` | 32 组参数、验证指标、实测耗时 |
| `artifacts/validation_results.json` | 最佳配置、TF-IDF 设置、指标、版本和代码哈希 |
| `artifacts/validation_predictions.csv` | 所有验证样本的行号、标签、预测和概率 |
| `artifacts/validation_errors.csv` | 49 个错判案例，额外保留正文片段 |
| `artifacts/indicative_features.csv` | 支持报告中词汇分析的模型权重 |
| `artifacts/data_audit.json`、`normalization_overlap.csv` | 数据检查结果和跨集合重复证据 |
| `artifacts/verification.json` | 保存模型的独立复核结果 |
| `artifacts/validation_confusion_matrix.png`、`validation_tuning.png` | 混淆矩阵和调参结果图，供展示使用 |

## 4 实验方法与结果

训练集有 21,487 封邮件，验证集有 4,496 封。标签 `0=Legitimate`、`1=Spam`。
数据正文原本是 `['Subject: ...']` 形式的列表字符串，代码用
`ast.literal_eval` 还原，再解码 HTML 实体、转小写并统一空白。
保留主题和正文，不使用文件名、标签作为特征；不做词干提取或停用词删除。

TF-IDF 只在训练集拟合词汇表和 IDF，验证集只调用 `transform`。
分词保留至少两个字符的 Unicode 单词 token，也可能包含数字。
朴素贝叶斯使用类别先验与平滑后的词汇权重分类，`alpha` 控制加性平滑。
预测使用类别分数最大值，无额外阈值调优；输出概率未进行校准。

参数组合为：ngram `(1,1)`/`(1,2)` × min_df `2`/`5` ×
sublinear_tf `False`/`True` × alpha `0.1`/`0.5`/`1.0`/`2.0`，共 32 组。
词汇上限固定为 50,000。按验证集 spam F1 选择，平局时先比较 precision，
再选择实验编号较小的配置。最佳配置为 unigram、min_df=2、
sublinear_tf=False、alpha=0.1。

| 验证集指标 | 结果 |
| --- | --- |
| Accuracy | 98.9101% |
| Precision | 98.9179% |
| Recall | 98.8734% |
| F1 | 98.8957% |
| 正常邮件被误判为 spam | 24 |
| Spam 被漏判为正常邮件 | 25 |

同一 TF-IDF 配置会在四个 alpha 实验间复用，因此不要把实验表中的
`total_fit_seconds` 相加当作整个搜索耗时。实际总耗时另有记录。
模型在固定数据、软件版本和参数下是确定性的；本代码没有重新随机划分数据。

## 5 数据问题需要全组同步

还原并规范化正文后，训练集有 245 个额外重复行，验证集有 57 个，
组内没有发现标签冲突。训练索引 10939 和验证索引 2209 的正常邮件
正文相同。这些都是从 0 开始的 pandas 数据行号，不是 CSV 的物理文本行号。

现有共享划分保持原样。排除那一封验证邮件的补充检查得到
Accuracy=98.9099%，spam F1 不变。这只检查已识别的重叠样本，不能证明
其他重复情况没有影响。Member 1 与 Member 5 应使用还原后的正文统一
检查三份数据；如果更换划分，所有模型都需重跑。本包不执行最终测试评估。

## 6 交接方式

### Member 3 使用完全相同的特征

完整 Pipeline 已包含拟合好的 TF-IDF，不再单独保存重复的向量器文件。
在本目录运行，或确保 `preprocessing.py` 能被 Python 导入：

```python
from predict import load_model
from preprocessing import load_split

pipeline = load_model()
vectorizer = pipeline.named_steps["tfidf"]
train = load_split("data/train.csv")
validation = load_split("data/validation.csv")
X_train = vectorizer.transform(train.text)
X_validation = vectorizer.transform(validation.text)
# 用 X_train 和 train.label 训练逻辑回归。
# 用 X_validation 和 validation.label 调整逻辑回归参数。
```

如需重新拟合相同配置，读取 `validation_results.json` 中的 `tfidf_config`，
调用 `make_vectorizer(**config).fit(train.text)`。这个配置是根据朴素贝叶斯
的验证结果选择的；若另行调文本特征，需与“相同特征”的比较区别说明。

### Member 5 统一评估

交付本包中的模型、代码、实验记录、输入哈希和数据检查结果。
先与 Member 1 确认重复问题及最终数据版本，再对全部模型统一测试。
`pipeline.predict(texts)` 返回标签；`pipeline.predict_proba(texts)[:, 1]`
返回 spam 的模型估计概率。测试集仍使用 Member 1 提供的对应版本。

### Member 6 集成演示

```python
from predict import load_model, predict_emails

model = load_model()  # 应用启动时加载一次
result = predict_emails(["Subject: Meeting\nPlease review the agenda."], model)[0]
```

输出包含 `label`、`label_name`、`spam_probability`、`out_of_vocabulary`。
空文本会报错；无已知词汇时标记 `out_of_vocabulary=True`，此时预测依赖
类别先验。演示界面应处理模型加载错误，并把概率标为模型估计值。
需一起提供 `preprocessing.py`、`predict.py`、模型和匹配的依赖，
不能只复制 joblib 文件。本模型尚未验证中文邮件、图片、附件或当前真实邮件流量。

## 7 本次清理说明

- 使用普通版报告替换旧版；每个包内只保留一份报告。
- 将原来的 README 和中文指南合并为本文。
- 删除独立 TF-IDF joblib 副本，改从完整 Pipeline 中提取同一个对象。
- 把独立 TF-IDF JSON 配置合并到 `validation_results.json`。
- 所有验证预测仍保留，但移除其中重复的正文片段；错判表继续保留片段。
- 不打包终端日志，保留结构化实验和复核结果。
- 不打包测试集，它属于 Member 5 的最终评估输入。
- 不打包虚拟环境、缓存、`.pyc`、临时脚本、渲染图片、旧压缩包。
- 去掉额外的全包文件哈希清单；保留用于数据、模型和训练代码复核的哈希。
- 保留六项测试、两张实验图和词汇分析文件，它们支持检查和展示。
- 保留完整依赖版本清单；这些版本用于复现，不因包名不熟悉就删除。

## 8 资料来源

- Member 1 提供的 `Spam_Email_Detection.zip` 中的数据说明和准备代码。
- [MultinomialNB 文档](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- [TfidfVectorizer 文档](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [数据泄漏与常见错误](https://scikit-learn.org/stable/common_pitfalls.html)

数据来源按 Member 1 的说明记录为 bvk/ENRON-spam；其说明中尚未确认
明确的再分发许可。本次仅整理本地交付文件，没有公开发布数据。
