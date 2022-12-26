import gradio as gr
import os
os.system('cd monotonic_align && python setup.py build_ext --inplace && cd ..')

import logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
import librosa
import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import numpy as np
import soundfile as sf
from preprocess_wave import FeatureInput

def resize2d(x, target_len):
    source = np.array(x)
    source[source<0.001] = np.nan
    target = np.interp(np.arange(0, len(source)*target_len, len(source))/ target_len, np.arange(0, len(source)), source)
    res = np.nan_to_num(target)
    return res

def transcribe(path, length, transform):
    featur_pit = featureInput.compute_f0(path)
    featur_pit = featur_pit * 2**(transform/12)
    featur_pit = resize2d(featur_pit, length)
    coarse_pit = featureInput.coarse_f0(featur_pit)
    return coarse_pit

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    print(text_norm.shape)
    return text_norm

convert_cnt = [0]

hps_ms = utils.get_hparams_from_file("configs/nyarumul.json")
net_g_ms = SynthesizerTrn(
    len(symbols),
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps_ms.data.hop_length,
    n_speakers=hps_ms.data.n_speakers,
    **hps_ms.model)

featureInput = FeatureInput(hps_ms.data.sampling_rate, hps_ms.data.hop_length)


hubert = torch.hub.load("bshall/hubert:main", "hubert_soft")

_ = utils.load_checkpoint("nyarumodel.pth", net_g_ms, None)

def vc_fn(sid,random1, input_audio,vc_transform):
    if input_audio is None:
        return "You need to upload an audio", None
    sampling_rate, audio = input_audio
    # print(audio.shape,sampling_rate)
    duration = audio.shape[0] / sampling_rate
    if duration > 450:
        return "请上传小于450s的音频，需要转换长音频请使用colab", None
    audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)

    source = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)
    print(source.shape)
    with torch.inference_mode():
        units = hubert.units(source)
        soft = units.squeeze(0).numpy()
    audio22050 = librosa.resample(audio, orig_sr=16000, target_sr=22050)
    sf.write("temp.wav", audio22050, 22050)
    pitch = transcribe("temp.wav", soft.shape[0], vc_transform)
    pitch = torch.LongTensor(pitch).unsqueeze(0)
    sid = torch.LongTensor([0]) if sid == "猫雷" else torch.LongTensor([1])
    stn_tst = torch.FloatTensor(soft)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, pitch=pitch,sid=sid, noise_scale=float(random1),
                               noise_scale_w=0.1, length_scale=1)[0][0, 0].data.float().numpy()
    convert_cnt[0] += 1
    print(convert_cnt[0])
    return "Success", (hps_ms.data.sampling_rate, audio)


app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("Basic"):
            gr.Markdown(value="""**new!!!!!** 更新了一个训练更多epoch的版本:链接[在这里](https://huggingface.co/spaces/innnky/nyaru-svc2.0-advanced)
            ，增加了3个音色，增加了f0音高曲线等方便查看转换后是否跑调
            
            old: 本模型相比与前一个模型，音质和音准方面有一定的提升，但是低音音域目前存在较大问题。

            目前猫雷模型能够唱的最低音为#G3(207hz) 低于该音会当场爆炸（之前的模型只是会跑调），
            
            因此请不要让这个模型唱男声的音高，请使用变调功能将音域移动至207hz以上。
            
            该模型的 [github仓库链接](https://github.com/innnky/so-vits-svc)
            
            如果想自己制作并训练模型可以访问这个 [github仓库](https://github.com/IceKyrin/sovits_guide)
            
            ps: 更新了一下模型，现在和视频中不是一个同一个模型，b站视频中的模型在git历史中（因为之前数据集中似乎混入了一些杂项导致音色有些偏离猫雷音色）

            """)
            sid = gr.Dropdown(label="音色",choices=['猫雷'], value="猫雷")
            vc_input3 = gr.Audio(label="上传音频（长度小于45秒）")
            vc_transform = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是12）",value=0)
            random1 = gr.Number(label="随机化程度，似乎会影响音质，建议保持默认",value=0.4)
            vc_submit = gr.Button("转换", variant="primary")
            vc_output1 = gr.Textbox(label="Output Message")
            vc_output2 = gr.Audio(label="Output Audio")
        vc_submit.click(vc_fn, [sid,random1,  vc_input3, vc_transform], [vc_output1, vc_output2])

    app.launch()