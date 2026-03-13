# 导入必要的库
from torch.utils.data import DataLoader, Dataset  # PyTorch 数据加载和数据集类
import torch  # PyTorch 库
import os  # 文件操作库
from scipy.io import wavfile  # 用于读取 WAV 文件
import glob  # 用于文件路径匹配


# 定义自定义数据集类
class MyData(Dataset):
    def __init__(self, wavList, frame_size=512, hope_size=256):
        # 初始化方法，传入音频文件路径列表、帧大小和希望大小
        super().__init__()
        self.wavList = wavList  # 音频文件路径列表
        self.frame_size = frame_size  # 每帧的大小，单位为样本数
        self.hope_size = hope_size  # 希望（步长）大小，单位为样本数

    def __len__(self):
        # 返回数据集的大小，即音频文件的数量
        return len(self.wavList)

    def STFT2Channel(self, signal):
        """
        将音频信号转换为双通道（实部和虚部）STFT表示
        参数:
        - signal: 输入音频信号，Tensor 类型

        返回:
        - 双通道 STFT 结果，形状为 [2, n_frames, n_freq_bins]
        """
        window = torch.hamming_window(self.frame_size)  # Hamming 窗函数
        stftResult = torch.stft(signal,
                                n_fft=self.frame_size,  # FFT 长度
                                hop_length=self.hope_size,  # 步长（重叠大小）
                                win_length=self.frame_size,  # 窗口长度
                                center=False,  # 不进行信号中心化
                                window=window,  # 窗口函数
                                return_complex=True)  # 返回复数形式的结果

        # 分离实部和虚部
        realPart = stftResult.real
        imagPart = stftResult.imag
        # 将实部和虚部堆叠成 2 通道形式
        doubleChannel = torch.stack((realPart, imagPart), dim=0)
        return doubleChannel.permute(0, 2, 1)  # 转换形状为 [2, n_freq_bins, n_frames]

    def VadFrameLabel(self, vad):
        """
        根据 VAD 数据生成对应的标签（活动/静音）
        参数:
        - vad: VAD 数据，Tensor 类型

        返回:
        - labelData: 每帧的标签数据，1 表示活动，0 表示静音
        """
        dataLength = len(vad)
        # 计算帧的数量
        numFrames = (dataLength - self.frame_size) // self.hope_size + 1
        labelData = torch.ones(numFrames)  # 默认所有帧为活动（1）

        for i in range(numFrames):
            start = i * self.hope_size  # 当前帧的起始位置
            end = start + self.frame_size  # 当前帧的结束位置
            activation = torch.sum(vad[start:end])  # 当前帧内 VAD 激活的总和
            # 如果当前帧内的激活小于帧大小的一半，则认为是静音（0）
            if activation < self.frame_size // 2:
                labelData[i] = 0

        return labelData

    def __getitem__(self, index):
        """
        获取指定索引的音频数据和对应的标签
        参数:
        - index: 数据的索引

        返回:
        - inputData: 处理后的输入数据（STFT 双通道）
        - labelVad: 对应的 VAD 标签
        """
        fs, data = wavfile.read(self.wavList[index])  # 读取 WAV 文件
        # 对音频信号进行 STFT 转换
        inputData = self.STFT2Channel(torch.Tensor(data[:, 0]))  # 假设音频数据的第一个通道为信号
        # 生成对应的 VAD 标签
        labelVad = self.VadFrameLabel(torch.Tensor(data[:, 1]))  # 假设音频数据的第二个通道为 VAD 数据

        # 确保输入数据和标签的形状一致
        assert inputData.shape[1] == labelVad.shape[0], "The shape of input data is different from that of label"
        return inputData, labelVad

# # 假设音频文件在这个路径下
# wavPath = r"D:\BaiduNetdiskDownload\216220120苏灿(保存)"
# frameSize = 512  # 每帧的大小
# hopeSize = 256  # 步长
# # 获取所有的 WAV 文件
# allWav = glob.glob(os.path.join(wavPath, "*.wav"))
#
# # 创建数据集实例
# trainData = MyData(allWav, frameSize, hopeSize)
# # 创建数据加载器实例，指定批大小为 32，并打乱数据
# trainLoader = DataLoader(trainData, batch_size=32, shuffle=True)
#
# # 选择使用 GPU 或 CPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
#
# # 示例：遍历数据加载器，输出数据的形状
# for data, labels in trainLoader:
#     print(data.shape, labels.shape)
