from __future__ import annotations

import html
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from model import calculate_radscore, load_config, predict_from_features


ROOT = Path(__file__).resolve().parent
CONFIG = load_config()

st.set_page_config(
    page_title="NSLN Metastasis Clinical–Radiomics Predictor",
    page_icon="🧬",
    layout="wide",
)

st.markdown(
    """
    <style>
      .main-title {font-size:2.2rem;font-weight:800;color:#263248;margin-bottom:0.2rem;}
      .subtitle {color:#667085;margin-bottom:1rem;}
      .result-card {padding:1.4rem;border-radius:14px;background:#7030A0;color:white;text-align:center;}
      .result-card .prob {font-size:3.4rem;font-weight:800;line-height:1.1;}
      .small-note {font-size:0.86rem;color:#667085;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🧬 临床–DCE-MRI影像组学融合预测系统</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">预测cN0且SLN阳性乳腺癌患者的非前哨淋巴结转移概率｜科研演示版 v1.0</div>',
    unsafe_allow_html=True,
)

st.warning(
    "本工具仅用于科研与模型复现，不替代MDT或临床诊疗决策。模型来自205例单中心队列，"
    "虽经嵌套5折交叉验证，仍需独立外部验证。请勿在公开云端上传包含患者身份信息的文件。"
)


def clinical_record_form() -> dict:
    st.subheader("1. 临床资料")
    st.caption(
        "当前锁定的融合公式仅使用“阳性SLN枚数”；其余变量作为结构化研究记录保存，不参与本版概率计算。"
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        study_id = st.text_input("匿名研究编号", placeholder="例如 P001")
        age = st.number_input("Age（岁）", min_value=18, max_value=100, value=50)
        mri_size = st.number_input(
            "MRI tumor size（mm）", min_value=1.0, max_value=200.0, value=20.0
        )
    with c2:
        location = st.selectbox(
            "MRI tumor location",
            ["外上", "外下", "内上", "内下", "中央区", "不明确"],
        )
        grade = st.selectbox("CNB Grade", ["1", "2", "3", "缺失/不明确"])
        stils = st.number_input(
            "CNB sTILs（%）", min_value=0.0, max_value=100.0, value=10.0
        )
    with c3:
        lvi = st.selectbox("CNB-LVI", ["阴性", "阳性", "缺失/不明确"])
        positive_sln = st.number_input(
            "No. of positive SLNs（枚）",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
        )
        cn0 = st.checkbox("术前临床腋窝淋巴结阴性（cN0）", value=True)

    return {
        "Study_ID": study_id.strip(),
        "Age": int(age),
        "MRI_tumor_size_mm": float(mri_size),
        "MRI_location": location,
        "CNB_Grade": grade,
        "CNB_sTILs_percent": float(stils),
        "CNB_LVI": lvi,
        "Positive_SLNs": int(positive_sln),
        "cN0": bool(cn0),
    }


def feature_csv_to_mapping(uploaded_file) -> dict:
    frame = pd.read_csv(uploaded_file)
    if frame.empty:
        raise ValueError("CSV中没有数据。")
    if len(frame) > 1:
        st.info("CSV包含多行，本次演示使用第一行；批量预测建议使用独立批处理脚本。")
    return frame.iloc[0].to_dict()


def render_result(record: dict, features: dict, quality: dict | None) -> None:
    if not record["cN0"]:
        st.error("该患者不是cN0，不属于当前模型的开发人群，系统停止预测。")
        return

    result, warnings = predict_from_features(
        features, int(record["Positive_SLNs"]), CONFIG
    )
    for warning in warnings:
        st.warning(warning)

    st.subheader("3. 预测结果")
    left, middle, right = st.columns([1.2, 1, 1])
    with left:
        st.markdown(
            f"""
            <div class="result-card">
              <div>NSLN转移预测概率</div>
              <div class="prob">{result.fusion_probability:.1%}</div>
              <div>Clinical–radiomics fusion model</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with middle:
        st.metric("Rad-score", f"{result.rad_score:.4f}")
        st.metric("临床模型概率", f"{result.clinical_probability:.1%}")
    with right:
        cutoff = float(CONFIG["fusion_model"]["descriptive_youden_cutoff"])
        relative_group = "高于内部描述性截点" if result.fusion_probability >= cutoff else "低于内部描述性截点"
        st.metric("内部描述性截点", f"{cutoff:.3f}")
        st.metric("相对分层", relative_group)

    st.caption(
        "该截点来自内部OOF Youden分析，仅用于结果描述，尚未作为临床治疗阈值验证。"
    )

    with st.expander("查看影像组学计算明细"):
        rows = []
        for feature_name, z_value in result.standardized_features.items():
            spec = CONFIG["radscore"]["features"][feature_name]
            rows.append(
                {
                    "特征": feature_name,
                    "原始值": float(features.get(feature_name, features.get(spec["aliases"][1]))),
                    "开发队列均值": spec["mean"],
                    "开发队列SD": spec["sd_ddof0"],
                    "Z-score": z_value,
                    "系数": spec["coefficient"],
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.code(
            "Rad-score = -1.234808554 "
            "+ 0.446705937 × z(InverseVariance) "
            "− 0.120660900 × z(SizeZoneNonUniformityNormalized)"
        )
        st.code(
            "logit(P) = 0.280212649 "
            "+ 1.766272415 × Rad-score "
            "+ 0.804184831 × I(SLN=2) "
            "+ 1.630061723 × I(SLN≥3)"
        )

    if quality:
        with st.expander("查看MRI与Mask质量控制"):
            st.json(quality)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_version": CONFIG["model_version"],
        "clinical_record": record,
        "rad_score": result.rad_score,
        "clinical_probability": result.clinical_probability,
        "fusion_probability": result.fusion_probability,
        "fusion_linear_predictor": result.linear_predictor,
        "standardized_features": result.standardized_features,
        "image_mask_quality": quality,
        "research_only": True,
    }
    st.download_button(
        "下载匿名JSON结果报告",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name=f"{record['Study_ID'] or 'anonymous'}_NSLN_prediction.json",
        mime="application/json",
    )


record = clinical_record_form()
st.subheader("2. DCE-MRI影像组学输入")

input_mode = st.radio(
    "选择影像组学输入方式",
    [
        "上传SlicerRadiomics特征CSV（推荐用于严格复现）",
        "上传NIfTI图像＋人工勾画Mask（待一致性验证）",
        "手工输入2个核心特征（仅用于程序质控）",
    ],
)

feature_values = None
quality_values = None
ready = False

if input_mode.startswith("上传NIfTI"):
    nifti_validated = bool(
        CONFIG["deployment_flags"][
            "nifti_auto_extraction_validated_against_slicer"
        ]
    )
    local_override = os.getenv("NSLN_ENABLE_UNVALIDATED_NIFTI", "0") == "1"
    st.info(
        "请上传与模型开发相同DCE-MRI期相的image.nii.gz和标签值为1的mask.nii.gz。"
    )
    if not nifti_validated:
        st.error(
            "当前NIfTI自动提取尚未通过与原SlicerRadiomics结果的逐例等价性验证，"
            "公开部署默认锁定该入口。完成205例一致性验证后，方可在model_config.json中启用。"
        )
        if local_override:
            st.warning(
                "检测到本地研究调试开关；本次允许计算，但结果不得作为论文主分析或临床概率使用。"
            )
    a, b = st.columns(2)
    with a:
        image_file = st.file_uploader(
            "DCE-MRI image.nii.gz", type=["nii", "gz"], key="image_upload"
        )
    with b:
        mask_file = st.file_uploader(
            "Tumor mask.nii.gz", type=["nii", "gz"], key="mask_upload"
        )
    ready = (
        image_file is not None
        and mask_file is not None
        and (nifti_validated or local_override)
    )

elif input_mode.startswith("上传SlicerRadiomics"):
    feature_csv = st.file_uploader(
        "SlicerRadiomics导出CSV", type=["csv"], key="feature_csv"
    )
    ready = feature_csv is not None

else:
    a, b = st.columns(2)
    with a:
        inverse_variance = st.number_input(
            "original_glcm_InverseVariance",
            value=float(
                CONFIG["radscore"]["features"]["original_glcm_InverseVariance"]["mean"]
            ),
            format="%.10f",
        )
    with b:
        size_zone = st.number_input(
            "original_glszm_SizeZoneNonUniformityNormalized",
            value=float(
                CONFIG["radscore"]["features"][
                    "original_glszm_SizeZoneNonUniformityNormalized"
                ]["mean"]
            ),
            format="%.10f",
        )
    feature_values = {
        "original_glcm_InverseVariance": inverse_variance,
        "original_glszm_SizeZoneNonUniformityNormalized": size_zone,
    }
    ready = True

calculate_button = st.button(
    "⚡ 提取特征并计算NSLN转移概率",
    type="primary",
    disabled=not ready,
    width="stretch",
)

if calculate_button:
    try:
        if input_mode.startswith("上传NIfTI"):
            with st.spinner("正在进行图像/Mask质控并提取影像组学特征……"):
                # PyRadiomics/SimpleITK are optional imaging dependencies.
                # Import them only when the validated NIfTI workflow is actually used,
                # so the public CSV/manual-input app can deploy without compiling
                # PyRadiomics on Streamlit Community Cloud.
                from radiomics_pipeline import extract_original_features

                feature_values, quality_values = extract_original_features(
                    image_file, mask_file
                )
        elif input_mode.startswith("上传SlicerRadiomics"):
            feature_values = feature_csv_to_mapping(feature_csv)

        render_result(record, feature_values, quality_values)
    except Exception as error:
        st.error(f"无法完成预测：{html.escape(str(error))}")

st.divider()
metrics = CONFIG["internal_validation"]
st.caption(
    f"内部验证：n={metrics['n']}；融合模型OOF AUC="
    f"{metrics['fusion_auc']:.3f}（95%CI "
    f"{metrics['fusion_auc_ci95'][0]:.3f}–{metrics['fusion_auc_ci95'][1]:.3f}）；"
    f"Brier score={metrics['fusion_brier']:.3f}。"
)
