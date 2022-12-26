import pyaudio
import numpy as np

RATE = 44100
CHUNK = 2048 * 32
p = pyaudio.PyAudio()
# 打开声卡，设置 采样深度为16位、声道数为2、采样率为16、输入、采样点缓存数量为2048
# stream = pa.open(format=pyaudio.paInt16, channels=2, rate=16000, input=True, frames_per_buffer=2048)
player = p.open(format=pyaudio.paInt16, channels=2, rate=RATE, output=True,frames_per_buffer=CHUNK)
stream = p.open(format=pyaudio.paInt16, channels=2, rate=RATE, input=True, frames_per_buffer=CHUNK)
for i in range(int(20*RATE/CHUNK)): #do this for 10 seconds
    player.write(np.fromstring(stream.read(CHUNK),dtype=np.int16))

stream.stop_stream()
stream.close()
p.terminate()