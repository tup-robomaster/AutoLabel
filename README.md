# TUP目标检测数据集自动标注Toolchain


## 1.简介
本仓库为TUP目标检测数据集自动标注工具Toolchain
包含多个工具脚本，方便进行自动化数据集样本筛选，标注。
Pipeline共由以下几部分组成:
- 自动标注脚本: `main.py`
- 图像去重脚本: `compare.py` 
- 人工审核脚本: `observe.py`

#TODO:
- 图像模糊检测，减少模糊样本比例
- Jupyter脚本
## 2.环境配置
在该目录下打开终端,执行如下命令
   ```shell
   pip install -r requirements.txt
   cd thirdparty/Galaxy_Linux_Python_2.0.2106.9041/api/
   python setup.py install
   ```
## 3.教程
见本仓库Wiki
<!-- 3. 新增分布分析功能，可以对先验数据集角点分布进行分析，便于进一步进行对比。
可在`dataset_cfg.yaml`配置分布分析参数。

4. 执行main.py,具体使用说明请参考--h参数
   ```shell
   cd ../..
   python main.py
   ``` -->
