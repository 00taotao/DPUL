import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
# 固定随机种子
from models.seed import setup_seed
#各个客户端的数据独立同分布
def mnist_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)#每个客户端的数据集数量（总数不变）
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]#dict_users是指每个客户端的数据集，all_idx是指所有图片的序号
    for i in range(num_users):#对每个客户端进行分配
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))#每个客户端随机取相同数量的图片，图片不重复
        all_idxs = list(set(all_idxs) - dict_users[i])#取出后剩余数据集
    return dict_users #返回各客户端数据集


    #独立非同分布
def mnist_noniid(dataset, num_users):
    setup_seed(0)
    num_shards, num_imgs = 200, 300 #将数据集分成200份，每份300张图片
    idx_shard = [i for i in range(num_shards)] #每份的序号
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}#分配每个客户端key
    idxs = np.arange(num_shards*num_imgs)# 给每个图片编号
    labels = dataset.train_labels.numpy() #将minist标签转成np类型

    # 分配标签
    idxs_labels = np.vstack((idxs, labels))# 将图片序号和标签堆成两排
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]# 先将标签排排序，并返回其原始序号，再将图片序号按照标签序号排序
    idxs = idxs_labels[0,:]# 数据集图片序号
    proportions = np.random.dirichlet(np.ones(num_users))
    rand_set = np.random.choice(idx_shard, size=num_shards, replace=False)
    # 分配数据
    for i in range(num_users):
        # 随机分配给num_users个客户端，服从Dirichlet分布
        num_shards_per_user = int(proportions[i] * num_shards)
        selected_shards = rand_set[:num_shards_per_user]
        rand_set = rand_set[num_shards_per_user:]
        for rand in selected_shards:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# 独立非同分布
# 独立非同分布
def cifar_noniid(dataset, num_users):
    setup_seed(0)
    num_shards, num_imgs = 200, 250  # 将数据集分成200份，每份300张图片
    idx_shard = [i for i in range(num_shards)]  # 每份的序号
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  # 分配每个客户端key
    idxs = np.arange(num_shards * num_imgs)  # 给每个图片编号
    labels = np.array(dataset.targets)  # 将CIFAR-10标签转换为NumPy数组

    # 分配标签
    idxs_labels = np.vstack((idxs, labels))  # 将图片序号和标签堆成两排
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 先将标签排序，并返回其原始序号，再将图片序号按照标签序号排序
    idxs = idxs_labels[0, :]  # 数据集图片序号
    proportions = np.random.dirichlet(np.ones(num_users))
    rand_set = np.random.choice(idx_shard, size=num_shards, replace=False)

    # 分配数据
    for i in range(num_users):
        # 随机分配给num_users个客户端，服从Dirichlet分布
        num_shards_per_user = int(proportions[i] * num_shards)
        selected_shards = rand_set[:num_shards_per_user]
        rand_set = rand_set[num_shards_per_user:]
        for rand in selected_shards:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def fmnist_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def fmnist_noniid(dataset, num_users):
    setup_seed(0)
    num_shards, num_imgs = 200, 300  # 将数据集分成200份，每份300张图片
    idx_shard = [i for i in range(num_shards)]  # 每份的序号
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  # 分配每个客户端key
    idxs = np.arange(num_shards * num_imgs)  # 给每个图片编号
    labels = np.array(dataset.targets)  # 将CIFAR-10标签转换为NumPy数组

    # 分配标签
    idxs_labels = np.vstack((idxs, labels))  # 将图片序号和标签堆成两排
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 先将标签排序，并返回其原始序号，再将图片序号按照标签序号排序
    idxs = idxs_labels[0, :]  # 数据集图片序号
    proportions = np.random.dirichlet(np.ones(num_users))
    rand_set = np.random.choice(idx_shard, size=num_shards, replace=False)

    # 分配数据
    for i in range(num_users):
        # 随机分配给num_users个客户端，服从Dirichlet分布
        num_shards_per_user = int(proportions[i] * num_shards)
        # selected_shards = rand_set[:num_shards_per_user]
        # rand_set = rand_set[num_shards_per_user:]
        # 随机取num_shards_per_user个shard
        selected_shards = np.random.choice(rand_set, num_shards_per_user, replace=False)
        rand_set = list(set(rand_set) - set(selected_shards))
        for rand in selected_shards:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

# 独立非同分布
def cifar_noniid(dataset, num_users):
    setup_seed(0)
    proportions = np.random.dirichlet(np.ones(num_users))
    # 先把数据集打乱
    idxs = np.arange(len(dataset))
    np.random.shuffle(idxs)
    labels = np.array(dataset.targets)[idxs]
    # 每个客户端的数据集数量
    num_shards, num_imgs = 100, 500  # 将数据集分成200份，每份250张图片
    idx_shard = [i for i in range(num_shards)]  # 每份的序号
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  # 分配每个客户端key

    # 分配标签
    rand_set = np.random.choice(idx_shard, size=num_shards, replace=False)

    # 分配数据
    for i in range(num_users):
        # 随机分配给num_users个客户端，服从Dirichlet分布
        num_shards_per_user = int(proportions[i] * num_shards-1)+1
        selected_shards = rand_set[:num_shards_per_user]
        rand_set = rand_set[num_shards_per_user:]
        # selected_shards = np.random.choice(rand_set, num_shards_per_user, replace=False)
        # rand_set = list(set(rand_set) - set(selected_shards))
        for rand in selected_shards:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users
def imagenet_tiny_noniid(dataset, num_users):
    setup_seed(0)
    num_shards, num_imgs = 500, 200  # 将数据集分成500份，每份200张图片
    idx_shard = [i for i in range(num_shards)]  # 每份的序号
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  # 分配每个客户端key
    idxs = np.arange(num_shards * num_imgs)  # 给每个图片编号
    labels = np.array(dataset.targets[:num_shards * num_imgs])  # 将CIFAR-100训练集标签转换为NumPy数组

    # 分配标签
    idxs_labels = np.vstack((idxs, labels))  # 将图片序号和标签堆成两排
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 先将标签排序，并返回其原始序号，再将图片序号按照标签序号排序
    idxs = idxs_labels[0, :]  # 数据集图片序号
    proportions = np.random.dirichlet(np.ones(num_users))
    rand_set = np.random.choice(idx_shard, size=num_shards, replace=False)

    # 分配数据
    for i in range(num_users):
        # 随机分配给num_users个客户端，服从Dirichlet分布
        num_shards_per_user = int(proportions[i] * num_shards-1)+1
        selected_shards = rand_set[:num_shards_per_user]
        rand_set = rand_set[num_shards_per_user:]
        # selected_shards = np.random.choice(rand_set, num_shards_per_user, replace=False)
        # rand_set = list(set(rand_set) - set(selected_shards))
        for rand in selected_shards:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

def cinic10_noniid(dataset, num_users):
    setup_seed(0)
    num_shards, num_imgs = 450, 200  # 将数据集分成500份，每份200张图片
    idx_shard = [i for i in range(num_shards)]  # 每份的序号
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  # 分配每个客户端key
    idxs = np.arange(num_shards * num_imgs)  # 给每个图片编号
    labels = np.array(dataset.targets[:num_shards * num_imgs])  # 将CIFAR-100训练集标签转换为NumPy数组

    # 分配标签
    idxs_labels = np.vstack((idxs, labels))  # 将图片序号和标签堆成两排
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 先将标签排序，并返回其原始序号，再将图片序号按照标签序号排序
    idxs = idxs_labels[0, :]  # 数据集图片序号
    proportions = np.random.dirichlet(np.ones(num_users))
    rand_set = np.random.choice(idx_shard, size=num_shards, replace=False)

    # 分配数据
    for i in range(num_users):
        # 随机分配给num_users个客户端，服从Dirichlet分布
        num_shards_per_user = int(proportions[i] * num_shards-1)+1
        selected_shards = rand_set[:num_shards_per_user]
        rand_set = rand_set[num_shards_per_user:]
        # selected_shards = np.random.choice(rand_set, num_shards_per_user, replace=False)
        # rand_set = list(set(rand_set) - set(selected_shards))
        for rand in selected_shards:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users
