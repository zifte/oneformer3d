# 简单说，这个文件是用来 “告诉程序怎么训练一个 3D 全景分割模型” 的说明书，主要包含以下几部分核心信息：
# 1. 用什么模型？（model部分）
# 模型叫S3DISOneFormer3D，专门处理 S3DIS 数据集（一个室内场景点云数据集）。
# 模型分两大部分：
# ** backbone（ backbone）**：用SpConvUNet，负责从点云中提取基础特征（类似 “看清楚点云里的细节”）。
# ** decoder（解码器）**：用QueryDecoder，负责根据特征预测 “哪些点属于同一物体（实例）” 和 “每个点是什么类别（语义）”。
# 损失函数：用来判断模型预测得好不好，同时关注实例分割和语义分割的误差。
# 2. 用什么数据？（dataset部分）
# 用的是S3DISSegDataset_这个类来加载数据，数据存在data/s3dis/目录下，包含点云坐标、颜色、语义标签（每个点是什么类别）、实例标签（每个点属于哪个物体）。
# 训练用的是 S3DIS 数据集中的 Area 1、2、3、4、6，测试用 Area 5（这是 S3DIS 数据集的标准划分方式）。
# 3. 数据怎么处理？（pipeline部分）
# 训练时：
# 加载点云和标签 → 随机选 18 万个点（控制数据量） → 随机翻转、缩放、平移（增加数据多样性，让模型更鲁棒） → 颜色归一化（方便模型学习）。
# 测试时：
# 加载点云和标签 → 只做颜色归一化（不做翻转等增强，保证结果稳定）。
# 4. 怎么训练？（训练相关配置）
# 每次训练用 2 个样本（batch_size=2），用 3 个线程加载数据。
# 优化器用AdamW，学习率 0.0001，训练 512 个 epoch（完整过一遍所有数据算 1 个 epoch）。
# 每 16 个 epoch 保存一次模型，只保留最好的 1 个模型。
# 5. 怎么评估？（evaluator部分）
# 用UnifiedSegMetric评估，同时看语义分割（mIoU）和实例分割（AP）的效果。
# 区分 “背景类”（如天花板、墙等，stuff_class_inds）和 “物体类”（如桌子、椅子等，thing_class_inds），分别计算指标。
# 总结：这个文件就像给模型的 “训练计划表”，规定了用什么模型、什么数据、怎么处理数据、怎么训练和评估，最终目标是让模型能准确识别室内场景中点云的语义类别和实例归属。

_base_ = [
    'mmdet3d::_base_/default_runtime.py', # 定义文件：mmdet3d/_base_/default_runtime.py（MMDet3D 库内置）。继承 MMDet3D 的默认运行时配置（包含日志、钩子、分布式训练等基础设置）
]
custom_imports = dict(imports=['oneformer3d']) # 自定义模块导入，确保oneformer3d目录下的类（如模型、变换等）能被注册器识别。

# model settings
num_channels = 64
num_instance_classes = 13
num_semantic_classes = 13

model = dict(
    type='S3DISOneFormer3D', # 定义于 oneformer3d/oneformer3d.py，指定模型主类，实现 S3DIS 数据集上的 3D 全景分割逻辑。
    data_preprocessor=dict(type='Det3DDataPreprocessor'), # 定义文件：mmdet3d/models/data_preprocessors/det3d_data_preprocessor.py.作用：3D 检测数据预处理器，负责归一化、设备转换等。
    in_channels=6,
    num_channels=num_channels,
    voxel_size=0.05,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    backbone=dict(
        type='SpConvUNet', # 定义于 oneformer3d/backbones/spconv_unet.py（或类似 backbone 目录），基于 SpConv 的 3D U-Net backbone，提取点云特征。
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict(
        type='QueryDecoder', # 定义于 oneformer3d/decoders/query_decoder.py，Transformer 查询解码器，生成实例和语义预测结果。
        num_layers=3,
        num_classes=num_instance_classes,
        num_instance_queries=400,
        num_semantic_queries=num_semantic_classes,
        num_instance_classes=num_instance_classes,
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=True),
    criterion=dict(
        type='S3DISUnifiedCriterion', # S3DISUnifiedCriterion 定义于 oneformer3d/criterions/unified_criterion.py，整合语义和实例损失的统一损失函数。
        # 子损失如InstanceCriterion 定义于 oneformer3d/criterions/instance_criterion.py，包含匹配器（HungarianMatcher）和损失计算逻辑。
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type='S3DISSemanticCriterion', # 语义损失计算。定义文件：oneformer3d/criterions/semantic_criterion.py。
            loss_weight=5.0),
        inst_criterion=dict(
            type='InstanceCriterion', # 实例损失计算（含匈牙利匹配）。定义文件：oneformer3d/criterions/instance_criterion.py。
            matcher=dict(
                type='HungarianMatcher', # 实例匹配器，用于预测与 GT 的 bipartite 匹配。定义文件：oneformer3d/criterions/matcher.py。
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0)]),
            loss_weight=[0.5, 1.0, 1.0, 0.5],
            num_classes=num_instance_classes,
            non_object_weight=0.05,
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    train_cfg=dict(),
    test_cfg=dict( # 测试阶段后处理参数（如 NMS 阈值、分数筛选等）。相关逻辑：在模型S3DISOneFormer3D的forward_test方法中使用，定义于oneformer3d/oneformer3d.py。
        topk_insts=450,
        inst_score_thr=0.0,
        pan_score_thr=0.4,
        npoint_thr=300,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.15,
        nms=True,
        matrix_nms_kernel='linear',
        num_sem_cls=num_semantic_classes,
        stuff_cls=[0, 1, 2, 3, 4, 5, 6, 12],
        thing_cls=[7, 8, 9, 10, 11]))

# dataset settings
dataset_type = 'S3DISSegDataset_'作用：S3DIS 数据集类，加载点云、语义掩码、实例掩码等数据。定义文件：oneformer3d/datasets/s3dis_dataset.py。
data_root = 'data/s3dis/'
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask')

train_area = [1, 2, 3, 4, 6]
test_area = 5

train_pipeline = [
    dict(
        type='LoadPointsFromFile', # 加载点云文件（坐标 + 颜色）。定义文件：oneformer3d/transforms/loading.py。
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D', # 加载 3D 标注（语义掩码、实例掩码）。定义文件：oneformer3d/transforms/loading.py。
        with_label_3d=False,
        with_bbox_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(
        type='PointSample_', # 点云采样，固定训练点数量。定义文件：oneformer3d/transforms/transforms_3d.py。
        num_points=180000),
    dict(type='PointInstClassMapping_', # 实例类别映射，统一类别索引。定义文件：oneformer3d/transforms/transforms_3d.py。
        num_classes=num_instance_classes),
    dict(
        type='RandomFlip3D', # 3D 点云随机翻转增强。定义文件：mmdet3d/transforms/transforms_3d.py（MMDet3D 内置）。
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans', # 全局旋转、缩放、平移增强。定义文件：mmdet3d/transforms/transforms_3d.py（MMDet3D 内置）。
        rot_range=[0.0, 0.0],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(
        type='NormalizePointsColor_', # 点云颜色归一化。定义文件：oneformer3d/transforms/transforms_3d.py。
        color_mean=[127.5, 127.5, 127.5]),
    dict(
        type='Pack3DDetInputs_', # 将数据打包为模型输入格式（转换为张量）。定义文件：oneformer3d/transforms/formatting.py。
        keys=[
            'points', 'gt_labels_3d',
            'pts_semantic_mask', 'pts_instance_mask'
        ])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(
        type='MultiScaleFlipAug3D', # 测试时多尺度 / 翻转增强。定义文件：mmdet3d/transforms/transforms_3d.py（MMDet3D 内置）。
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='NormalizePointsColor_',
                color_mean=[127.5, 127.5, 127.5])]),
    dict(type='Pack3DDetInputs_', keys=['points'])
]

# run settings
train_dataloader = dict(
    batch_size=2,
    num_workers=3,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True), # 作用：数据采样器，控制训练时打乱、验证时顺序采样。定义文件：mmengine/dataset/samplers/default_sampler.py（MMEngine 内置）。
    dataset=dict(
            type='ConcatDataset', # 拼接多个 Area 的数据集（如 Area 1-4、6）。定义文件：mmengine/dataset/dataset_wrapper.py（MMEngine 内置）。
            datasets=([
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file=f's3dis_infos_Area_{i}.pkl',
                    pipeline=train_pipeline,
                    filter_empty_gt=True,
                    data_prefix=data_prefix,
                    box_type_3d='Depth',
                    backend_args=None) for i in train_area])))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=f's3dis_infos_Area_{test_area}.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        data_prefix=data_prefix,
        box_type_3d='Depth',
        backend_args=None))
test_dataloader = val_dataloader

class_names = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
    'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter', 'unlabeled']
label2cat = {i: name for i, name in enumerate(class_names)}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[num_semantic_classes],
    classes=class_names,
    dataset_name='S3DIS')
sem_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

val_evaluator = dict(
    type='UnifiedSegMetric', # 作用：全景分割评估指标（语义 mIoU、实例 AP 等）。定义文件：oneformer3d/evaluation/metrics/unified_seg_metric.py。
    stuff_class_inds=[0, 1, 2, 3, 4, 5, 6, 12],
    thing_class_inds=[7, 8, 9, 10, 11],
    min_num_points=1,
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=sem_mapping,
    submission_prefix_semantic=None,
    submission_prefix_instance=None,
    metric_meta=metric_meta)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper', # 作用：优化器包装器，管理 AdamW 优化器。定义文件：mmengine/optim/optim_wrapper.py（MMEngine 内置）。
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))
param_scheduler = dict(type='PolyLR', begin=0, end=512, power=0.9) # 作用：Poly 学习率调度器，按多项式衰减学习率。定义文件：mmengine/optim/scheduler.py（MMEngine 内置）。

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)] # 钩子. 作用：训练迭代后清空 GPU 缓存，减少内存占用。定义文件：mmengine/hooks/empty_cache_hook.py（MMEngine 内置）。
default_hooks = dict(
    checkpoint=dict( # 作用：模型保存钩子，控制保存间隔、最佳模型判定。定义文件：mmengine/hooks/checkpoint_hook.py（MMEngine 内置）。
        interval=16,
        max_keep_ckpts=1,
        save_best=['all_ap_50%', 'miou'],
        rule='greater'))

load_from = 'work_dirs/tmp/instance-only-oneformer3d_1xb2_scannet-and-structured3d.pth'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=512, val_interval=16) # 基于 epoch 的训练循环。定义文件：均位于mmengine/runner/loops.py（MMEngine 内置）。
val_cfg = dict(type='ValLoop') # 验证循环。定义文件：均位于mmengine/runner/loops.py（MMEngine 内置）。
test_cfg = dict(type='TestLoop') # 测试循环。定义文件：均位于mmengine/runner/loops.py（MMEngine 内置）。
