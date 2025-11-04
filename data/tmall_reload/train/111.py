import numpy as np

# 首先看看原始文件的内容
original_samples = np.load('D:/桌面/causal/data/tianmao_reload/train/0.npy')
print("原始采样的形状:", original_samples.shape)
print("前几个样本:", original_samples[:5])
import numpy as np

# 1. 加载数据
train_dict, train_dict_count = np.load('D:/桌面/causal/data_tianmao/output/train.npy', allow_pickle=True)
original_samples = np.load('D:/桌面/causal/data/tianmao_reload/train/0.npy')

# 2. 从原始采样文件中提取用户-物品对的顺序
unique_user_item_pairs = []
for i in range(len(original_samples)):
    user_id, item_i, _ = original_samples[i]
    pair = (int(user_id), int(item_i))
    if pair not in unique_user_item_pairs:
        unique_user_item_pairs.append(pair)

# 3. 设置参数
num_ng = 5
item_num = 597977
np.random.seed(2022)

# 4. 按原始用户-物品对顺序生成采样
features_fill = []
for user_id, item_i in unique_user_item_pairs:
    positive_list = [int(item) for item in train_dict[user_id]]
    for t in range(num_ng):
        item_j = np.random.randint(item_num)
        while item_j in positive_list:
            item_j = np.random.randint(item_num)
        features_fill.append([user_id, item_i, item_j])

features_fill = np.array(features_fill, dtype=np.int64)

# 5. 打印结果和对比
print("新生成采样的形状:", features_fill.shape)
print("前几个样本:", features_fill[:5])
print("\n原始采样的前几个样本:", original_samples[:5])
is_same = np.array_equal(features_fill, original_samples)
print("\n是否完全相同:", is_same)

if not is_same:
    # 找出第一个不匹配的样本
    for i in range(min(len(features_fill), len(original_samples))):
        if not np.array_equal(features_fill[i], original_samples[i]):
            print(f"\n第一个不匹配的位置 {i}:")
            print(f"原始: {original_samples[i]}")
            print(f"新生成: {features_fill[i]}")
            print(f"随机种子状态:", np.random.get_state()[1][:5])
            break