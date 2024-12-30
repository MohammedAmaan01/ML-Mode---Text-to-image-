import tkinter as tk 
import customtkinter as ctk
from PIL import ImageTk
from authtoken import auth_token

import torch
from torch.amp import autocast
from diffusers import StableDiffusionPipeline

# Create the app window
app = ctk.CTk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

# Input prompt
prompt = ctk.CTkEntry(master=app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

# Display area
lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

# Load Stable Diffusion model
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token= "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" )
pipe.to(device)

# Generate image
def generate():
    with autocast("cuda"):
        result = pipe(prompt.get(), guidance_scale=8.5)
        
        # Ensure you are accessing the correct key
        image = result["images"][0]  # This is usually where the image is stored
        
    image.save('generatedimage.png')    
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)
    lmain.image = img  
    
# Button to trigger image generation
trigger = ctk.CTkButton(master=app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate") 
trigger.place(x=200, y=60)

trigger.pack()
app.mainloop()
