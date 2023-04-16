import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from urllib.parse import urlparse

import allrank.models.losses as losses
import numpy as np
import os
import torch
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, create_data_loaders
from allrank.models.model import *
from allrank.models.model_utils import get_torch_device, CustomDataParallel
from allrank.training.train_utils import fit
from allrank.utils.command_executor import execute_command
from allrank.utils.experiments import dump_experiment_result, assert_expected_metrics
from allrank.utils.file_utils import create_output_dirs, PathsContainer, copy_local_to_gs
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import dummy_context_mgr
from argparse import ArgumentParser, Namespace
from attr import asdict
from functools import partial
from pprint import pformat
from torch import optim
from sklearn.metrics import ndcg_score
from allrank.data.dataset_loading import *
from allrank.training.train_utils import *
import time
'''
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i],hidden_sizes[i+1]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
    def forward(self, x):
        x = nn.functional.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))
        x = self.output_layer(x)
        return x
'''
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.activation = instantiate_class("torch.nn.modules.activation", "ReLU")
        self.output_size = output_size
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, output_size)

    def forward(self, x):
        out1 = self.activation(self.layer1(x).squeeze(dim=2))
        out2 = self.activation(self.layer2(out1).squeeze(dim=2))
        out3 = self.activation(self.layer3(out2).squeeze(dim=2))
        return out3


class newLTRModel(nn.Module):
    """
    This class represents a full neural Learning to Rank model with a given encoder model.
    """
    def __init__(self, input_layer, encoder, fc2, output_layer):
        """
        :param input_layer: the input block (e.g. FCModel)
        :param encoder: the encoding block (e.g. transformer.Encoder)
        :param output_layer: the output block (e.g. OutputLayer)
        """
        super(newLTRModel, self).__init__()
        self.input_layer = input_layer if input_layer else nn.Identity()
        self.encoder = encoder if encoder else first_arg_id
        #self.n_features = self.input_layer.n_features
        self.fc2 = fc2
        self.output_layer = output_layer

    def prepare_for_output(self, x, mask, indices):
        """
        Forward pass through the input layer and encoder.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: encoder output of shape [batch_size, slate_length, encoder_output_dim]
        """
        return self.encoder(self.input_layer(x), mask, indices)

    def forward(self, x, mask, indices):
        """
        Forward pass through the whole LTRModel.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: model output of shape [batch_size, slate_length, output_dim]
        """

        #x1 = x[:, :, :self.n_features]
        #x2 = x[:, :, self.n_features:]
        #x1 = self.prepare_for_output(x1, mask, indices)
        #x2 = self.prepare_for_output(x2, mask, indices)
        after_transformer_x = self.prepare_for_output(x, mask, indices)
        out = self.output_layer(self.fc2(self.prepare_for_output(x, mask, indices)))

        return after_transformer_x, out

    def score(self, x, mask, indices):
        """
        Forward pass through the whole LTRModel and item scoring.

        Used when evaluating listwise metrics in the training loop.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: scores of shape [batch_size, slate_length]
        """
        
        #if x.size()[-1] == self.n_features:
        #    return self.output_layer.score(self.fc2(self.prepare_for_output(x, mask, indices)))
        #else:
        #    rx2 = x[:, :, self.n_features:]
        #    return self.output_layer.score(self.fc2(self.prepare_for_output(rx2, mask, indices)))
        return self.output_layer.score(self.prepare_for_output(x, mask, indices))


def parse_args() -> Namespace:
    parser = ArgumentParser("allRank")
    parser.add_argument("--job-dir", help="Base output path for all experiments", required=True)
    parser.add_argument("--run-id", help="Name of this run to be recorded (must be unique within output dir)",
                        required=True)
    parser.add_argument("--config-file-name", required=True, type=str, help="Name of json file with config")

    return parser.parse_args()
#解析命令行参数，包括 job-dir、run-id 和 config-file-name

def run():
# reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    args = parse_args()

    paths = PathsContainer.from_args(args.job_dir, args.run_id, args.config_file_name)

    create_output_dirs(paths.output_dir) #该函数用于创建存储模型训练输出结果的目录，如果目录不存在则创建，如果已存在则什么也不做

    logger = init_logger(paths.output_dir) #该函数用于初始化 logger 对象，用于在训练过程中输出日志信息。它会返回一个 logger 对象，用于记录运行时的信息。参数 paths.output_dir 指定了存储日志文件的路径。
    logger.info(f"created paths container {paths}")

    # read config
    config = Config.from_json(paths.config_path) #该函数用于从配置文件中加载模型训练所需的配置。它从路径 paths.config_path 中加载 JSON 格式的配置文件，并返回一个 Config 对象。Config 对象是 allrank 库中定义的一个数据类，用于存储模型训练的各种配置。
    logger.info("Config:\n {}".format(pformat(vars(config), width=1)))

    output_config_path = os.path.join(paths.output_dir, "used_config.json") #该语句用于拼接生成输出配置文件的路径。
    execute_command("cp {} {}".format(paths.config_path, output_config_path)) #该函数用于将配置文件从 paths.config_path 复制到 output_config_path
    #这个函数设置随机种子以获得可重复结果，并根据命令行参数创建输出路径和记录器。然后，它读取配置文件并将其复制到输出路径中以备份。

    # train_ds, val_ds
    train_ds, val_ds = load_libsvm_dataset(
        input_path=config.data.path,
        slate_length=config.data.slate_length,
        validation_ds_role=config.data.validation_ds_role,
    ) #该函数用于加载训练和验证数据集，其中包括读取数据文件、将数据集分割为训练集和验证集等步骤。load_libsvm_dataset 函数是 allrank 库中定义的一个函数，它从指定的路径中加载 LIBSVM 格式的数据集，并将其分为训练集和验证集，返回两个 Dataset 对象，分别表示训练集和验证集。

    '''
    splitpoint = train_ds_272.X_by_qid[0].shape[-1] // 2 #split_point=136

    result1 = []
    result2 = []
    for arr in train_ds_272.X_by_qid:
        result1.append(arr[:, :splitpoint])
        result2.append(arr[:, splitpoint:])

    train_ds_true = train_ds_272
    train_ds_noise = train_ds_272

    train_ds_true.X_by_qid = result1 
    train_ds_noise.X_by_qid = result2
    '''

    # train_ds_true: 真实特征组成的数据集
    # train_ds_noise：含噪特征组成的数据集
    
    # val_ds.X_by_qid:
    # [
    # [ [1, 2, ..., 136], [1, 2, ..., 136], [1, 2, ..., 136], [1, 2, ..., 136], [1, 2, ..., 136], ...]
    # [ [1, 2, ..., 136], [1, 2, ..., 136], [1, 2, ..., 136], [1, 2, ..., 136], [1, 2, ..., 136], ...]
    #                                                                                                 ]
    # val_ds.X_by_qid: 三维：(qid个数，每个qid对应的webpage数，136维)
    # val_ds.y_by_qid: [[第一个qid对应的各个网页的lable], [第二个qid对应的各个网页的label], ... ]
    # val_ds.longest_query_length: vali.txt中各个qid对应的网页的最大数量
    # val_ds.transform: dataset_loading.FixLength + dataset_loading.ToTensor的组合

    n_features = train_ds.shape[-1] #该语句用于获取数据集中每个样本的特征数。在 allrank 库中，数据集通常被表示为一个二维数组，其中每一行表示一个样本，每一列表示一个特征。train_ds.shape[-1] 表示训练集中每个样本的最后一个维度，即特征数。
    #n_features = 272

    # train_dl, val_dl
    train_dl, val_dl = create_data_loaders(
        train_ds, val_ds, num_workers=config.data.num_workers, batch_size=config.data.batch_size)
    #这个函数加载训练和验证数据集，并创建对应的数据加载器。



    # gpu support
    dev = get_torch_device() #获取PyTorch的设备，如果有GPU，则选择GPU，否则使用CPU。
    logger.info("Model training will execute on {}".format(dev.type))


    
    fc_model = FCModel(**config.model.fc_model, n_features=136)
    d_model = fc_model.output_size
    #print(d_model)
    transformer = make_transformer(n_features=d_model, **asdict(config.model.transformer, recurse=False))
    #输入：d_model, 输出：d_model
    fc_model2 = MLP(d_model, d_model)

    model = newLTRModel(fc_model, transformer, fc_model2, OutputLayer(d_model, **config.model.post_model))
    
    '''
    fc_model = FCModel(**config.model.fc_model, n_features=136)
    d_model = fc_model.output_size
    #print(d_model)
    transformer = make_transformer(n_features=d_model, **asdict(config.model.transformer, recurse=False))
    #输入：d_model, 输出：d_model
    model = LTRModel(fc_model, transformer, OutputLayer(d_model, **config.model.post_model))
    '''

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    #model输入维度：136，输出维度：d_model(144...)

    if torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
        logger.info("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
    model.to(dev)


    optimizer = getattr(optim, config.optimizer.name)(params=model.parameters(), **config.optimizer.args) #使用getattr函数获取torch.optim模块中的优化器，该优化器的类型由config.optimizer.name指定。参数params指定了要优化的模型参数，**config.optimizer.args用于传递优化器的额外参数。
    loss_func = partial(getattr(losses, config.loss.name), **config.loss.args) #获取 losses 模块中的损失函数，该函数的类型由 config.loss.name 指定。 
    if config.lr_scheduler.name:
        scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.name)(optimizer, **config.lr_scheduler.args)
    else:
        scheduler = None

    '''
    for xb, yb, indices in train_dl:
        print(xb.shape)
        i=1
        for x in xb:
            if i==1:
                np.savetxt("xb.txt", np.array(x))
                i+=1
            else :
                np.savetxt("xb2.txt", np.array(x))
        print(yb.shape)
        np.savetxt("yb.txt", np.array(yb))
        print(indices.shape)
        np.savetxt("indices.txt", np.array(indices))

        assert False
    '''

    #xb shape: 每个batch中 (独特的qid数量，slate_length, 272), 272维特征和数据集中的数相同
    #yb shape: (独特的qid数量，slate_length)，第二维slate_length个数中，每个数代表xb中相应行的网页特征和qid之间的label
    #indices shape: (独特的qid数量，slate_length)， 第二维slate_length个数中，每个数是下标，每行都从0开始


    with torch.autograd.detect_anomaly() if config.detect_anomaly else dummy_context_mgr():  # type: ignore 如果config.detect_anomaly为真，则启用PyTorch中的异常检测机制，以便在计算图中发生异常时及时发现和解决问题。如果config.detect_anomaly为假，则不会进行异常检测，以提高代码的运行效率。
        # run training
        result = fit(
            model=model, #新的model
            loss_func=loss_func, #pointwise_rmse
            optimizer=optimizer, #Adam
            scheduler=scheduler, #StepLR
            train_dl=train_dl, 
            valid_dl=val_dl,
            config=config, #json参数文件中的参数
            device=dev, #CPU/GPU
            output_dir=paths.output_dir,
            tensorboard_output_path=paths.tensorboard_output_path,
            **asdict(config.training)
        )


if __name__ == "__main__":
    run()






'''
assert False

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i],hidden_sizes[i+1]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
    def forward(self, x):
        x = nn.functional.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))
        x = self.output_layer(x)
        return x


# 定义一个带有额外层的Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TransformerModel, self).__init__()
        # Transformer编码器层
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=6)
        # MLP层
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, data):
        # Transformer编码器
        transformer_output = self.transformer_encoder(data)
        # MLP层
        mlp_output = self.mlp(transformer_output)
        return mlp_output

# 定义损失函数
class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

    def forward(self, pred, target, ltr_weight=1.0, l2_weight=0.1):
        # LTR损失
        ltr_loss = torch.mean(torch.norm(pred - target, p=2, dim=1))
        # L2限制损失
        l2_loss = torch.mean(torch.norm(pred, p=2, dim=1))
        # 总损失
        total_loss = ltr_weight * ltr_loss + l2_weight * l2_loss
        return total_loss

# 创建训练和测试数据
input_dim = 100
batch_size = 32
train_data = np.random.randn(1000, input_dim).astype(np.float32)
noise_data = np.random.randn(50, input_dim).astype(np.float32)  # 添加高斯噪声
noise_mask = np.random.choice(1000, 50, replace=False)
train_data[noise_mask] += 0.5 * noise_data
train_labels = np.zeros((1000, 2))
train_labels[:950,0] = 1
train_labels[950:,1] = 1

train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 创建模型和损失函数
model = TransformerModel(input_dim, 128, 2)
criterion = LossFunction()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 开始模型训练
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 清空梯度
        optimizer.zero_grad()
        # 向前运行模型
        output = model(data)
        # 计算损失
        loss = criterion(output, target)
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 记录损失值
        if (batch_idx+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, batch_idx+1, len(train_loader), loss.item()))
'''