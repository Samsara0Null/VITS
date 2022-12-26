import logging
import os
import threading
import time

import sounddevice as sd

from sovits import infer_tool

logging.getLogger('numba').setLevel(logging.WARNING)
infer_tool.del_temp_wav("./record/")


def play(data, fs):
    global play_status
    sd.play(data, fs)
    sd.wait()
    play_status = [0]
    return 0


# 这个不改
file_name = 1
record_pth = "./wav_temp/record"
infer_tool.mkdir(["./wav_temp", record_pth])

# 说话人序号、升降半音、模型、配置json
spk_id = 4
tran = 6
model_name = "152_epochs.pth"  # 模型名称（pth文件夹下）
config_name = "nyarumul.json"  # 模型配置（config文件夹下）

# 加载sovits模型、参数
net_g_ms, hubert_soft, feature_input, hps_ms = infer_tool.load_model(f"pth/{model_name}", f"configs/{config_name}")
target_sample = hps_ms.data.sampling_rate
print("start")
threading.Thread(target=os.system, args=("python sclient_split_recorder.py",)).start()
while True:
    a = time.time()
    if os.path.exists(f"{record_pth}/{file_name}.wav"):
        record_name = str(file_name)
    else:
        continue
    try:
        source_path = f"{record_pth}/{record_name}.wav"
        out_audio, out_sr = infer_tool.infer(source_path, spk_id, tran, net_g_ms, hubert_soft, feature_input)
        jump = False
    except Exception as e:
        print(e)
        jump = True
    finally:
        os.remove(source_path)
    if jump:
        file_name += 1
        continue
    threading.Thread(target=play, args=(out_audio, target_sample,)).start()
    play_status = [1]
    print("time taken: " + str(time.time() - a))
    print("playing", file_name)
    while play_status[0] == 1:
        pass
    print("play end", file_name)
    file_name += 1
