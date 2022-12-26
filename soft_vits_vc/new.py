import gradio as gr

def generate_mutimodal(title, context, img):
    return f"Title:{title}\nContext:{context}\n...{img}"

server = gr.Interface(
    fn=generate_mutimodal, 
    inputs=[
        gr.Textbox(lines=1, placeholder="请输入标题"),
        gr.Textbox(lines=2, placeholder="请输入正文"),
        gr.Image(shape=(200, 200),  label="请上传图片(可选)")
    ], 
    outputs="text"
)

server.launch()