# 项目名称：
# 程序内容：
# 作   者：MBJC
# 开发时间：2022/9/5 0:12

import pyaudio

import wave

import time



# instantiate PyAudio (1)

# define callback (2)

# open stream using callback (3)



CHUNK = 1024

FORMAT = pyaudio.paInt16

CHANNELS = 2

RATE = 44100

p = pyaudio.PyAudio()

################################################3

wf = wave.open("output.wav", 'wb')

wf.setnchannels(CHANNELS)

wf.setsampwidth(p.get_sample_size(FORMAT))

wf.setframerate(RATE)



def callback(in_data, frame_count, time_info, status):

    wf.writeframes(in_data)

    return (in_data, pyaudio.paContinue)

##################################################打开文件

stream = p.open(format=FORMAT,

        channels=CHANNELS,

        rate=RATE,

        input=True,

        frames_per_buffer=CHUNK,

        stream_callback=callback)



# start the stream (4)

stream.start_stream()



# wait for stream to finish (5)

for _ in range(500):

    if stream.is_active():

        time.sleep(0.1)   #休眠，不影响录音



# stop stream (6)

stream.stop_stream()   #直到运行此句录音终止

stream.close()

wf.close()



# close PyAudio (7)

p.terminate()