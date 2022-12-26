import requests
from playsound import playsound

# 先get一下 http://127.0.0.1:8000/model?model_name=xxx&config_name=xxx 加载模型
test_file = open("./raw/十年.wav", "rb")
# speaker是音色序号，tran是升降半音数s
test_url = "http://127.0.0.1:8000/svc?speaker=4&tran=0"
test_response = requests.get(test_url, files={"audio": test_file})

with open("./wav_temp/http/cli_out.wav", "wb") as f:
    f.write(test_response.content)
playsound("./wav_temp/http/cli_out.wav")
