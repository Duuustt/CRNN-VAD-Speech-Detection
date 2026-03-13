# 导入需要的库
import os  # 用于文件和目录操作
from scipy.io import wavfile  # 用于读取和写入 WAV 文件
import numpy as np  # 用于数组操作
import glob  # 用于查找符合特定规则的文件路径名
import random  # 用于生成随机数
import matplotlib.pyplot as plt
# 配置文件路径
wavDir = r"D:\pycharm projects\pythonProject\深度学习课设\数据集"  # 输入的语音文件夹路径
savePath = r"pycharm projects\pythonProject\深度学习课设"  # 保存处理后文件的路径
noisePath = r"D:\pycharm projects\pythonProject\深度学习课设\NOISE92X_16K\NOISE92X_16K"  # 噪声文件路径

# 配置参数
segmentLength = 3  # 每个片段的时长（秒）
snr = 5  # 信噪比（Signal-to-Noise Ratio），表示信号功率和噪声功率的比值，单位为dB

# 创建目录的函数
def MkDir(name):
    if not os.path.exists(name):  # 如果目录不存在，则创建
        os.makedirs(name)

# 创建保存结果的目录
MkDir(savePath)

# 获取噪声文件夹中的所有 WAV 文件
noiseWav = glob.glob(os.path.join(noisePath, "*.wav"))

# 遍历原始音频文件夹中的文件
for file in os.listdir(wavDir):
    if file[-3:] == "wav":  # 如果是 WAV 文件
        wavPath = os.path.join(wavDir, file)  # 获取音频文件的完整路径
        fs, data = wavfile.read(wavPath)  # 读取音频文件，fs是采样率，data是音频数据
        dataLength = len(data[:, 0])  # 获取音频数据的长度

        # 将音频数据分割成多个片段（根据 segmentLength 参数来分割）
        for i, v in enumerate(range(0, dataLength - segmentLength * fs, segmentLength * fs)):
            tmpData = data[v:v + segmentLength * fs, :]  # 获取当前片段的数据
            vadData = np.where(tmpData[:, 1] > 0.6, 1, 0)  # 使用 VAD（语音活动检测）标记语音部分
            tmpData[:, 1] = vadData  # 将 VAD 标记写入音频数据中

            # 随机选择一个噪声文件
            noiseFile = random.choice(noiseWav)
            sm, noiseData = wavfile.read(noiseFile)  # 读取噪声文件
            noiseLen = len(noiseData)  # 获取噪声数据的长度
            start = random.randint(1, noiseLen - segmentLength * fs - 1)  # 随机选择噪声片段的起始位置
            addNoise = noiseData[start:start + segmentLength * fs]  # 获取对应长度的噪声片段

            # 计算信号和噪声的功率，根据信噪比调整噪声的强度
            signal_power = abs(np.sum(tmpData[:, 0] ** 2))  # 计算信号的功率
            noise_power = abs(np.sum(addNoise ** 2))  # 计算噪声的功率
            rate = signal_power / (noise_power * 10 ** (snr / 10))  # 根据信噪比调整噪声功率

            # 根据信噪比计算噪声的缩放因子
            scale_factor = np.sqrt(rate)

            # 将噪声添加到原始信号中，并限制其值在有效范围内
            mixed_np = tmpData[:, 0] + addNoise * scale_factor
            tmpData[:, 0] = np.clip(mixed_np, -2 ** 15, 2 ** 15 - 1).astype(np.int16)

            # 构建保存文件的名称
            saveName = file[:-3] + "_" + str(i) + ".wav"
            saveWav = os.path.join(savePath, saveName)  # 获取保存路径

            # 保存包含噪声的音频片段
            wavfile.write(saveWav, fs, tmpData)
# 检查 tmpData[:, 1] 的最大值和最小值
print(f"Min value: {np.min(tmpData[:, 1])}")
print(f"Max value: {np.max(tmpData[:, 1])}")
# 假设 tmpData[:, 1] 是归一化后的音频数据
plt.hist(tmpData[:, 1], bins=50, range=(0, 1))
plt.title("Histogram of Audio Data")
plt.xlabel("Amplitude")
plt.ylabel("Frequency")
plt.show()
