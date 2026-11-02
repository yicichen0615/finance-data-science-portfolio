# 金融与数据科学项目作品集
Finance & Data Science Portfolio

课程项目涵盖投资组合优化、跨市场策略研究、客户分群与深度学习。团队项目按团队成果展示，不代表全部由本人独立完成。

## 常见 ETF 投资组合优化
[研究报告](etf-portfolio-report.pdf)

STAT GR5261 团队项目。对15只ETF比较等权、均值方差、最小方差与CVaR组合，采用滚动窗口进行样本外评估，分析风险收益与尾部风险约束。当前提供报告，未附独立运行代码。

## 马科维茨与单指数模型的组合优化
[Excel模型](markowitz-single-index-model.xlsx)

比较Markowitz与Index Model，包含最小方差、最大Sharpe组合、有效前沿及ESG约束计算。工作簿保留原始表名和公式；存在外部工作簿链接，完整复现可能需要原始依赖及Excel Solver。尚未独立复算全部结果。

## 跨市场趋势跟踪策略
[研究演示文稿](cross-market-trend-following.pdf)

MATH GR5360 团队项目。研究美债期货与比特币市场的趋势特征，使用随机游走检验、趋势突破及回撤控制，采用滚动样本外验证。当前提供演示文稿，未附独立运行代码。

## 聚类方法下的客户分群
[研究报告](customer-segmentation-report.pdf) · [分析Notebook](customer-segmentation.ipynb)

STAT GR5291 团队项目。基于Mall Customer数据，对比K-Means、GMM、谱聚类及层次聚类，分析客户画像与分群稳定性。

运行需要Python、numpy、pandas、matplotlib、scikit-learn及scipy。原Notebook使用`/content/Mall_Customers.csv`，请获取对应数据并修改路径。数据未随仓库提供；代码尚未在新环境重跑。

## 基于CNN的街景数字识别
[研究报告](svhn-cnn-report.pdf) · [Python代码](svhn_cnn.py)

个人课程项目。使用TensorFlow/Keras构建CNN识别SVHN街景数字，结合卷积、池化、Dropout及Batch Normalization，评估分类效果与特征图。

运行需要Python、numpy、scipy、matplotlib、tensorflow及scikit-learn。请将SVHN的`train_32x32.mat`与`test_32x32.mat`放在`datasets/SVHN/format2/`，然后运行`python svhn_cnn.py`。数据未随仓库提供；代码尚未在新环境重跑。

## 材料说明
所有投资表现均来自课程研究或历史回测，不是实盘业绩。原报告保留团队署名，具体分工以实际参与情况为准。课程及数据材料的权利归各自权利人所有。本仓库不包含实习公司的内部材料。
