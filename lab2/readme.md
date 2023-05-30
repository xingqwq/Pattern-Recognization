## 程序运行

```shell
python main.py --mode test --pt pt模型文件路径
```

## 命名

- 时间戳
- testACC
- 模型类别
- 测试集上的正确率

## 模型文件

- 1684127162_testACC_sher_0.699.pt：仿射变换模型
- 1684127228_testACC_sher_0.702.pt
- 1684127854_testACC_none_0.651.pt：不带数据增强模型
- 1684136211_testACC_weight_0.714.pt：多种数据增强+权重衰减模型
- 1684136642_testACC_weight_0.728.pt
- 1684137729_testACC_non_weight_0.71.pt：不带数据增强+权重衰减模型