# NSLN Clinical–Radiomics Streamlit App

本项目是临床–DCE-MRI瘤内影像组学融合模型的科研演示版，用于预测cN0且SLN阳性乳腺癌患者的非前哨淋巴结转移概率。

## 已锁定的模型

- 影像组学输入：GLCM InverseVariance和GLSZM SizeZoneNonUniformityNormalized。
- 临床预测变量：阳性SLN枚数，编码为1枚、2枚和≥3枚。
- 融合模型内部验证：205例；OOF AUC 0.775（95%CI 0.707–0.840）；Brier score 0.156。
- 其他临床变量可以在网页录入并写入匿名报告，但不参与当前固定公式。

## 项目结构

```text
app.py                         Streamlit界面
model.py                       固定Rad-score和融合概率计算
model_config.json              模型参数唯一来源
radiomics_pipeline.py          NIfTI/Mask质控及PyRadiomics提取
radiomics_params.yaml          与原SlicerRadiomics一致的参数
clinical_input_template.csv    临床资料模板
verify_slicer_equivalence.py   与原Slicer结果逐例比较
tests/test_model.py            数学公式自动测试
requirements.txt               云端依赖
requirements-imaging.txt       本地NIfTI影像提取附加依赖
```

## 本地运行

建议新建独立Python 3.10环境：

```bash
conda create -n nsln-app python=3.10 -y
conda activate nsln-app
pip install -r requirements.txt
pip install -r requirements-imaging.txt
pip install --no-build-isolation PyRadiomics==3.0.1
streamlit run app.py
```

说明：`PyRadiomics==3.0.1`在部分平台仅提供源码包，其构建过程未正确声明
NumPy构建依赖，因此需在安装基础依赖后使用`--no-build-isolation`单独安装。
公开Streamlit Cloud版本默认锁定尚未完成等价性验证的NIfTI入口，不需要安装
PyRadiomics和SimpleITK。

浏览器访问：

```text
http://localhost:8501
```

## 运行公式测试

```bash
pip install pytest
pytest -q
```

## 验证自动提取与Slicer是否一致

在正式启用NIfTI＋Mask自动提取入口前，必须运行：

```bash
python verify_slicer_equivalence.py \
  --dataset /Users/mac/DL_Dataset \
  --slicer-csv /path/to/Radiomics_Tumor_ML.csv \
  --output slicer_equivalence_205.csv \
  --limit 205
```

建议预设判定标准：

- 两个核心特征逐例相对误差尽可能接近0；
- Rad-score最大绝对误差小于预设容差；
- 若PyRadiomics版本变化导致系统性差异，应优先使用SlicerRadiomics CSV入口，不能直接宣称自动提取流程与原模型等价。

当前状态：使用`/Users/mac/DL_Dataset`前3例进行初筛时，两个核心特征与原Slicer表的相对差异约为12%–41%；Patient 01当前Mask标签1体素数为10,154，而原Slicer diagnostics记录为13,282。因此公开版NIfTI自动预测入口默认锁定。

仅在本机调试自动提取流程时，可以临时运行：

```bash
NSLN_ENABLE_UNVALIDATED_NIFTI=1 streamlit run app.py
```

该调试结果不得写入论文主结果。取得与原建模完全相同的图像、Mask和软件环境，并完成205例等价性验证后，才可将`model_config.json`中的：

```json
"nifti_auto_extraction_validated_against_slicer": false
```

改为`true`。

## GitHub与Streamlit部署

1. 新建GitHub仓库。
2. 上传本目录中的代码和配置文件。
3. 不上传任何真实MRI、Mask、患者信息或密钥。
4. 在Streamlit Community Cloud中选择`app.py`作为入口。
5. Python版本选择3.10。
6. 云端仅使用`requirements.txt`中的轻量依赖；不要把本地影像依赖合并进去。
7. 部署完成后先用模拟数据以及一份去标识化SlicerRadiomics CSV测试。

公开云端版本只应用于匿名或模拟数据。真实患者影像建议在本机离线或医院内网运行。

## 重要方法学说明

部署使用的是全队列最终固定模型；APP预测概率不能被称为OOF预测。OOF结果仅用于估计模型内部验证性能。

本工具不提供治疗建议，不替代MDT或临床医生判断。
