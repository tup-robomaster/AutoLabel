# 数据集自动标注工具
## 使用说明
在该目录下打开终端,执行如下命令

1. 
   ```shell
   pip install -r requirements.txt
   ```
2. 
   ```shell
   cd Galaxy_Linux_Python_2.0.2106.9041/api/
   python setup.py install
   ```

3.新增分布分析功能，可以对先验数据集角点分布进行分析，便于进一步进行对比。
可在`dataset_cfg.yaml`配置分布分析参数。

4. 执行main.py,具体使用说明请参考--h参数
   ```shell
   cd ../..
   python main.py
   ```
