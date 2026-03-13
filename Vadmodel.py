import torch.nn as nn
import torch


class CRNN_VAD(nn.Module):
    """
    CRNN_VAD模型实现：用于语音活动检测（VAD）任务
    输入：[batch size, channels=1, T, n_fft] -> 频谱图（实部和虚部）
    输出：[batch size, T, 1] -> 每一帧的语音活动标签（0或1）
    """

    def __init__(self, feat_size):
        super(CRNN_VAD, self).__init__()

        # 第一卷积层：输入通道2（频谱图实部和虚部），输出通道4，卷积核大小为 (1, 3)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2,  # 输入为2个通道（实部和虚部）
                out_channels=4,  # 输出为4个通道
                kernel_size=(1, 3),  # 卷积核大小为 (1, 3)，即频率维度为3
                stride=(1, 2),  # 步幅为 (1, 2)，表示在频谱图上水平跳跃2个位置
                padding=(0, 1),  # 填充 (0, 1)，使得输出在频率维度保持一致
            ),
            nn.BatchNorm2d(4),  # 批量归一化
            nn.LeakyReLU(negative_slope=0.3),  # 激活函数，LeakyReLU，负斜率为0.3，避免死神经元问题
        )

        # 第二卷积层：输入通道4，输出通道8，卷积核大小为 (1, 3)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,  # 输入为4个通道
                out_channels=8,  # 输出为8个通道
                kernel_size=(1, 3),  # 卷积核大小为 (1, 3)
                stride=(1, 2),  # 步幅为 (1, 2)
                padding=(0, 1),  # 填充 (0, 1)
            ),
            nn.BatchNorm2d(8),  # 批量归一化
            nn.LeakyReLU(negative_slope=0.3),  # 激活函数，LeakyReLU，负斜率为0.3
        )

        # 第三卷积层：输入通道8，输出通道16，卷积核大小为 (1, 3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2)  # 步幅为 (1, 2)
            ),
            nn.BatchNorm2d(16),  # 批量归一化
            nn.LeakyReLU(negative_slope=0.3),  # 激活函数，LeakyReLU，负斜率为0.3
        )

        # GRU层：处理卷积层输出的特征，捕捉时序信息
        # input_size = feat_size // 8 * 16，GRU输入的维度大小由卷积层输出特征图大小决定
        self.GRU = nn.GRU(
            input_size=feat_size // 8 * 16,  # 特征图的宽度经过卷积和池化后减小
            hidden_size=128,  # GRU的隐藏层维度，128维
            num_layers=1,  # GRU的层数，1层
            batch_first=True,  # 输入数据的维度顺序为 [batch_size, seq_len, feature_dim]
        )

        # 输出层：全连接层，将GRU的输出映射为单个输出（0或1），表示每一帧的语音活动
        self.output_dense = nn.Linear(128, 1)  # 将GRU输出的128维特征映射到1个输出

    def forward(self, x):
        """
        前向传播函数：
        处理输入x，通过卷积层提取特征，并通过GRU捕捉时序依赖，最终输出语音活动标签
        """
        # (B, 2, T, F) -> (batch_size, channels=2, time_steps=T, freq_bins=F)
        x1 = self.conv1(x)  # 经过第1卷积层
        x2 = self.conv2(x1)  # 经过第2卷积层
        x3 = self.conv3(x2)  # 经过第3卷积层

        # GRU输入需要交换维度：从 [batch_size, channels, time_steps, freq_bins] 转换为 [batch_size, time_steps, feature_dim]
        mid_in = x3
        mid_GRU_in = mid_in.permute(0, 2, 1, 3)  # 将维度调整为 [batch_size, time_steps, channels * freq_bins]
        mid_GRU_in = mid_GRU_in.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], -1)  # 将频率维度展开为特征维度

        # 传入GRU进行时序建模
        gru_out, _ = self.GRU(mid_GRU_in)

        # 输出层，通过sigmoid将GRU的输出映射为[0, 1]之间的概率，表示语音活动的概率
        sig_out = torch.sigmoid(self.output_dense(gru_out))  # 输出是[batch_size, time_steps, 1]

        return sig_out


class BCEFocalLoss(torch.nn.Module):
    """
    自定义的损失函数：结合了二元交叉熵损失（BCE）和焦点损失（Focal Loss）
    适用于类别不平衡的情况，重点关注难以分类的样本
    """

    def __init__(self, gamma=2, alpha=0.6, reduction="elementwise_mean"):
        super().__init__()
        self.gamma = gamma  # 焦点损失的gamma参数，控制难分类样本的权重
        self.alpha = alpha  # alpha值，平衡正负样本的权重
        self.reduction = reduction  # 损失计算方式，'elementwise_mean'表示取均值，'sum'表示求和

    def forward(self, target, pt):
        """
        计算焦点损失：
        target：真实标签，0或1
        pt：模型预测的概率值，经过sigmoid函数后的值
        """
        # alpha：正样本的权重
        loss = -self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        # 根据reduction方式来计算损失
        if self.reduction == "elementwise_mean":
            loss = torch.mean(loss)  # 按元素求均值
        elif self.reduction == "sum":
            loss = torch.sum(loss)  # 按元素求和

        return loss
