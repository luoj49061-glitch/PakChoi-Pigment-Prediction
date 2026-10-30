"""Export archived RGB-only vs RGB/HSV/Lab comparisons without fitting models."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from revision.common import ROOT,TRAITS,run_dir,load_manifest,output_dir,scores

RUN='20260921_210614_balanced128'
DL='20260921_093836_resnet_tuning'
MODELS=['ResNet18_RGB','ResNet18_Fusion']

def export(redraw=False,figure_source=None):
    rd=run_dir(RUN);dst=ROOT/'results/ablation';dst.mkdir(parents=True,exist_ok=True)
    for file in ['metrics.csv','test_metrics.csv','outer_cv_summary.csv','outer_cv_folds.csv','training_oof_metrics.csv','all_predictions.csv','outer_cv_predictions.csv']:
        d=pd.read_csv(rd/file,dtype={'sample_id':str} if 'predictions' in file else None)
        d=d[d.model.isin(MODELS)];d.to_csv(dst/file,index=False,float_format='%.17g')
    for file in ['paired_test_comparisons.csv','outer_cv_paired_differences.csv']:
        d=pd.read_csv(rd/file);d=d[(d.A=='ResNet18_RGB')&(d.B=='ResNet18_Fusion')]
        d.to_csv(dst/file,index=False,float_format='%.17g')
    shutil.copy2(rd/'RGB_ablation.csv',dst/'RGB_ablation.csv')
    m=load_manifest(RUN);pred=pd.read_csv(dst/'all_predictions.csv',dtype={'sample_id':str})
    table=pd.read_csv(dst/'metrics.csv')
    for trait in TRAITS:
        for split in ['train','test']:
            a=pred[(pred.trait==trait)&(pred.split==split)]
            for model in MODELS:
                part=a[a.model==model].sort_values('sample_id')
                ids=sorted(m.loc[m.split==split,'sample_id'])
                assert part.sample_id.tolist()==ids
                computed=scores(part.y_true,part.y_pred)
                row=table[(table.trait==trait)&(table.model==model)&(table.subset==split)].iloc[0]
                for key in ['R2','RMSE','MAE']:assert np.isclose(computed[key],row[key],rtol=0,atol=1e-12)
    index=[]
    for suffix in ['']+['_cv'+str(i) for i in range(1,6)]:
        for trait in TRAITS:
            folder=output_dir(DL+suffix,trait,'dl')
            for branch in ['RGB','HSV','Lab']:
                checkpoint=folder/('Best_Model_'+branch+'.pth');assert checkpoint.exists()
                index.append({'trait':trait,'evaluation':'fixed_test' if not suffix else suffix[1:],
                    'branch':branch,'checkpoint':str(checkpoint.relative_to(ROOT)),
                    'fusion_weights':str((folder/'fusion_weights.json').relative_to(ROOT)),
                    'config':str((run_dir(DL+suffix)/'config.json').relative_to(ROOT)),
                    'split':str((run_dir(DL+suffix)/'manifest.csv').relative_to(ROOT))})
    pd.DataFrame(index).to_csv(dst/'model_files.csv',index=False)
    if redraw:
        from revision.plot_train_test import render
        render(RUN)
    source=Path(figure_source) if figure_source else ROOT/'SHOW/figures_revised'/RUN/'train_test_style'
    figs=dst/'figures';figs.mkdir(exist_ok=True)
    for trait in TRAITS:
        for ext in ['png','svg']:
            name=trait+'_RGB_ablation.'+ext
            if (source/name).exists():shutil.copy2(source/name,figs/name)
            else:assert (figs/name).exists(), 'Use --redraw to generate missing figures'
    (dst/'README.md').write_text('''# RES颜色空间融合消融

对比：A RGB-only ResNet18；B RGB+HSV+Lab加权融合。
六项指标：chl_a、chl_b、total_chl、carot、total_carot、SPAD。
RGB-only使用已训练的RGB分支；融合使用同一RGB分支加HSV、Lab分支。
本次只整理已有结果，没有重新训练或重新选择参数。

## 文件
- metrics.csv：训练集、固定测试集R²/RMSE/MAE。
- outer_cv_summary.csv：同一外层五折的均值±标准差。
- outer_cv_folds.csv：每折成绩。
- training_oof_metrics.csv：训练集折外预测汇总，区别于训练拟合成绩。
- all_predictions.csv、outer_cv_predictions.csv：逐样本预测。
- RGB_ablation.csv、paired_test_comparisons.csv：固定测试集融合减RGB差值及95%区间。
- outer_cv_paired_differences.csv：五折配对差值，折间标准差不是置信区间。
- figures/：六张A/B消融图，PNG和SVG；浅色训练集、深色测试集。
- model_files.csv：项目内部模型、融合权重、参数与划分文件位置。

## 代码（在项目根目录）
revision/resnet_pipeline.py：分支训练和内部选择。
revision/resnet_refit.py：使用已选轮次和融合权重重拟合。
revision/summarize.py：消融指标及配对区间计算。
revision/nested_cv.py：相同外层五折评价。
revision/plot_train_test.py：训练/测试统一绘图。
revision/export_ablation.py：本目录结果导出与验证。

无需训练地重新整理结果和绘图：
PYTHONNOUSERSITE=1 /data2/yx/tuxiang/.revision_env/bin/python -m revision.export_ablation --redraw

模型权重已在RES目录保存，不重复复制权重或图片数据。
置信区间是固定模型下的匿名分组bootstrap区间，不代表不同训练随机种子的波动。
这是增强图片条目随机划分实验，不是独立原图泛化验证。
''',encoding='utf-8')
    print('PASS ablation export: 6 traits, 2 models, train/test and 5 folds, 12 PNG/SVG files; no training',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--redraw',action='store_true');p.add_argument('--figure-source');a=p.parse_args()
    export(a.redraw,a.figure_source)
