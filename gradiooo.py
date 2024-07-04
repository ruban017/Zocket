import gradio as gr
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import replicate
import replicate
import requests
import cv2
import numpy as np
import os
import json
from groq import Groq
from dotenv import load_dotenv


def prompt_generation(prompt):
    load_dotenv()
    mod = "mixtral-8x7b-32768"

    client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f""" Generate a prompt for generating an ad-creatives that are visually appealing and for photoshoots of product with appealing products : {json.dumps(prompt)}. keep it very simple and short but the model should generate good images based on this prompt.
                    Here is an example for your reference Create a series of visually stunning and high-quality advertisements for an Adidas sports shoe. The images should capture the essence of the shoes performance, style, and innovative design. 
                    Each photo should showcase the shoe from multiple angles, including close-ups of key features like the logo, sole, and materials. The background should enhance the products appeal with a sleek, modern aesthetic. 
                    Consider dynamic and engaging settings such as a high-energy urban street scene, a professional athletic field, or a sleek, minimalist studio environment with creative lighting and shadows. 
                    Emphasize the shoe's sporty and trendy attributes with vibrant colors, bold compositions, and energetic visuals. The overall look should reflect Adidas's brand identity of high performance and cutting-edge style, appealing to both athletes and fashion-conscious consumers.
                    you need to create similar prompts like this.  """
                }
            ],
            model=mod,
        )
        prompt_content = chat_completion.choices[0].message.content
        return prompt_content
    
    except Exception as e:
        return f"Error generating prompt: {str(e)}"

def image_segmentation(image):
    model = YOLO('best.pt')

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    results = model(image)
    result = results[0]
    masks = result.masks.data.cpu().numpy()
    colored_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)
    
    for i, mask in enumerate(masks):
        resized_mask = cv2.resize(mask.astype(float), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        resized_mask = (resized_mask > 0.5).astype(bool) 
        colored_mask[resized_mask] = colors[i]
    alpha = 0.5
    blended = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    # Print additional information
    # print(f"Number of detected objects: {len(masks)}")
    # for i, mask in enumerate(masks):
    #     print(f"Object {i+1}: Area = {np.sum(mask)} pixels")
    
    return blended

def image_generation(prompt):
    load_dotenv()
    prompt = prompt_generation(prompt)
    replicate_api_key = os.getenv("REPLICATE_API_TOKEN")
    output = replicate.run(
    "konieshadow/fooocus-api:fda927242b1db6affa1ece4f54c37f19b964666bf23b0d06ae2439067cd344a4",
    input={
        "prompt": prompt,
        "cn_type1": "ImagePrompt",
        "cn_type2": "ImagePrompt",
        "cn_type3": "ImagePrompt",
        "cn_type4": "ImagePrompt",
        "sharpness": 2,
        "image_seed": 50403806253646856,
        "uov_method": "Disabled",
        "image_number": 1,
        "guidance_scale": 4,
        "refiner_switch": 0.5,
        "negative_prompt": "",
        "style_selections": "Fooocus V2,Fooocus Enhance,Fooocus Sharp",
        "uov_upscale_value": 0,
        "outpaint_selections": "",
        "outpaint_distance_top": 0,
        "performance_selection": "Speed",
        "outpaint_distance_left": 0,
        "aspect_ratios_selection": "1152*896",
        "outpaint_distance_right": 0,
        "outpaint_distance_bottom": 0,
        "inpaint_additional_prompt": ""
    }
)
    print(output)
    url = output[0]
    response = requests.get(url)
    response.raise_for_status()
    image_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return  image

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("## AD - CREATION")
            with gr.Tabs():
                # with gr.TabItem("Prompt Generation"):
                #     gr.Markdown("## Prompt Generation")
                #     prompt_input = gr.Textbox(label="Enter text")
                #     prompt_output = gr.Textbox(label="Generated Prompt")
                #     prompt_button = gr.Button("Generate")
                #     prompt_button.click(prompt_generation, inputs=prompt_input, outputs=prompt_output)
                with gr.TabItem("Image Generation"):
                    gr.Markdown("## Image Generation")
                    gen_prompt_input = gr.Textbox(label="Enter Prompt")
                    gen_image_output = gr.Image(type="numpy", label="Generated Image")
                    gen_image_button = gr.Button("Generate Image")
                    gen_image_button.click(image_generation, inputs=gen_prompt_input, outputs=gen_image_output)
                with gr.TabItem("Image Segmentation"):
                    gr.Markdown("## Image Segmentation")
                    image_input = gr.Image(type="numpy", label="Upload Image")
                    image_output = gr.Image(type="numpy", label="Segmented Image")
                    image_button = gr.Button("Segment")
                    image_button.click(image_segmentation, inputs=image_input, outputs=image_output)
                

demo.launch(share=True, inline=False, server_name="0.0.0.0", server_port=7860)