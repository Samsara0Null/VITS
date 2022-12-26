import pyaudio
import wave
import time
import numpy as np

# 非阻塞式录音，省略import语句
path = "record.wav"
data_list = []  # 录制用list会好一点，因为bytes是常量，+操作会一直开辟新存储空间，时间开销大

def callback(in_data, frame_count, time_info, status):
    data_list.append(in_data)
    # output=False时数据可以直接给b""，但是状态位还是要保持paContinue，如果是paComplete一样会停止录制
    return b"", pyaudio.paContinue

record_seconds = 180  # 录制时长/秒
pformat = pyaudio.paInt16
channels = 1
rate = 16000  # 采样率/Hz

audio = pyaudio.PyAudio()
stream = audio.open(format=pformat,
                    channels=channels,
                    rate=rate,
                    input=True,
                    stream_callback=callback)

stream.start_stream()

t1 = time.time()
# 录制在stop_stream之前应该都是is_active()的，所以这里不能靠它来判断录制是否结束
while time.time() - t1 < record_seconds:
    time.sleep(0.1)

wav_data = b"".join(data_list)
with wave.open("tmp.wav", "wb") as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(pyaudio.get_sample_size(pformat))
    wf.setframerate(rate)
    wf.writeframes(wav_data)

stream.stop_stream()
stream.close()
audio.terminate()
