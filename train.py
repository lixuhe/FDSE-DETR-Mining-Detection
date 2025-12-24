import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR(r'ultralytics/cfg/models/rt-detr/rtdetr-Faster-DCN-Slimneck.yaml')
    local_weights = r'D:\\L_RTDETR-main\\rtdetr-l.pt'
    model.train(data=r'D:/foreign-data/foreign-data.yaml',
                cache=True,
                imgsz=640,  # 如果目标较小，可以尝试 imgsz=800 提升检测精度
                epochs=200,
                batch=8,    # 根据显存调整，可尝试4或12
                workers=0,
                device='0',
                #resume='D:/L_RTDETR-main/runs/train/rtdetr-Faster-DCN-Slimneck-emasvfl/weights/last.pt',
                
                # ===== 针对暗光环境的数据增强策略 =====
                mosaic=0.0,       # 关闭mosaic（避免背景图干扰）
                mixup=0.0,        # 关闭mixup
                copy_paste=0.0,   # 关闭copy_paste
                
                # 暗光场景专用：针对预处理后的图像调整
                # 注意：由于数据集已经过aggressive增强，这里减小亮度增强幅度
                hsv_h=0.01,       # 减小色调变化（矿山环境色调相对稳定）
                hsv_s=0.4,        # 适度饱和度变化
                hsv_v=0.4,        # 降低亮度变化（预处理后已提亮，这里适度即可）
                
                # 几何增强：适合细长目标（锚杆、木棒）
                degrees=10.0,     # 增加旋转角度，模拟物体不同摆放方向
                translate=0.15,   # 增加平移，提升位置鲁棒性
                scale=0.5,        # 增加缩放范围，适应不同距离
                shear=3.0,        # 添加剪切变换，模拟视角变化
                perspective=0.0001,  # 轻微透视变换
                fliplr=0.5,       # 保持水平翻转
                flipud=0.0,       # 不使用垂直翻转（皮带方向固定）
                
                # ===== 针对小目标的优化 =====
                conf=0.15,        # 降低置信度阈值，提高召回率（减少漏检）
                iou=0.5,          # 降低IoU阈值，对细长目标更友好
                
                # 学习率和优化器配置
                amp=True,
                augment=True,
                optimizer='AdamW',
                lr0=0.00015,      # 稍微降低初始学习率（更稳定）
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3.0,   # 减少warmup
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                
                # 损失函数权重调整（针对小目标）
                box=8.5,          # 提高box loss权重，强化定位精度
                cls=0.5,
                cos_lr=True,
                patience=50,
                close_mosaic=10,
                project='runs/train',
                name='rtdetr-Faster-DCN-Slimneck-emasvfl',  # 新版本命名
                )