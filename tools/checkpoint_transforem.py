import torch
from torch.cuda import device

from lib.config import cfg
from lib.models import get_net

# 假设这是您加载和创建模型的方式
# from models.yolo import Model

# 1. 定义旧模型和新模型的配置文件路径 (或者直接使用您提供的列表)

# 2. 加载旧的预训练权重

# 通常权重保存在 'model' 这个键下，具体看您的保存方式
checkpoint = torch.load(cfg.MODEL.PRETRAINED)
old_state_dict = checkpoint['state_dict']

# 3. 根据新的配置创建模型，此时模型权重是随机初始化的
# new_model = Model(new_cfg)
# 假设 new_model 是您新结构的模型对象

device = 'cuda:0'

print("load model to device")
model = get_net(cfg).to(device)
new_state_dict = model.state_dict()

print("成功加载旧权重和创建新模型。")

# 创建一个新的 state_dict 用于存放迁移后的权重
migrated_state_dict = new_state_dict.copy()

# 用于统计迁移和跳过的层
migrated_keys = []
skipped_keys = []

# 定义旧索引到新索引的映射规则
# 格式: {old_index: new_index}
index_mapping = {
    0: 0,  # ELANNet
    1: 1,  # PaFPNELAN
    2: 3,  # YOLOXHead
    3: 4,  # FPN_C3
    4: 5,  # FPN_C4
    # DA Seg Head (从旧索引5开始，迁移到新索引7开始)
    5: 7, 6: 8, 7: 9, 8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17,
    16: 18,  # seg_head
    17: 19,  # FPN_C2...
    18: 21  # bts
}

print("开始进行权重迁移...")

for new_key in new_state_dict.keys():
    # PyTorch模型的层名通常是 'model.索引.模块名...' 的格式
    # 我们通过分割字符串来获取索引号
    try:
        new_idx = int(new_key.split('.')[1])
    except (IndexError, ValueError):
        # 如果key不符合 'model.X. ...' 格式，直接跳过 (例如 'stride' 等非权重参数)
        continue

    # 寻找当前新索引对应的旧索引
    old_idx = None
    for k, v in index_mapping.items():
        if v == new_idx:
            old_idx = k
            break

    if old_idx is not None:
        # 如果找到了映射关系，构建旧的层名
        key_parts = new_key.split('.')
        key_parts[1] = str(old_idx)
        old_key = '.'.join(key_parts)

        # 确认旧的层名存在于旧的checkpoint中
        if old_key in old_state_dict:
            # 确认权重张量的形状一致
            if new_state_dict[new_key].shape == old_state_dict[old_key].shape:
                migrated_state_dict[new_key] = old_state_dict[old_key]
                migrated_keys.append(new_key)
            else:
                print(
                    f"警告: 形状不匹配，跳过 {new_key} (新: {new_state_dict[new_key].shape}, 旧: {old_state_dict[old_key].shape})")
                skipped_keys.append(new_key)
        else:
            print(f"警告: 在旧checkpoint中找不到key {old_key}，跳过 {new_key}")
            skipped_keys.append(new_key)
    else:
        # 如果没有找到映射关系，说明这是一个新层
        # 我们保留新模型自身的随机初始化权重
        print(f"信息: {new_key} 是新层，将使用随机初始化权重。")
        skipped_keys.append(new_key)

checkpoint['state_dict'] = migrated_state_dict
torch.save(checkpoint, 'D:\Multi-task\Git\Multi-Task\\tools\\runs\\BddDataset\\migrated\\migrated_checkpoint.pth')
print(f"\n迁移完成！")
print(f"成功迁移 {len(migrated_keys)} 个权重张量。")
print(f"跳过/保留初始化 {len(skipped_keys)} 个权重张量。")