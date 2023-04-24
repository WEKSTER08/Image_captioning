import os
import urllib.request
from flask import Flask, request, jsonify, render_template
# from transformers import AutoTokenizer, AutoModelForCausalLM


from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds


# predict_step(['./Image1.png'])

app = Flask(__name__)

# # Load the Hugging Face model and tokenizer
# model_name = "microsoft/image-captioning-english-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded file and save it to disk
        file = request.files["file"]
        filename = file.filename
        filepath = os.path.join("uploads", filename)
        file.save(filepath)
        
        # # Load the image from disk and convert it to bytes
        # with open(filepath, "rb") as f:
        #     image_bytes = f.read()

        # # Generate captions using the Hugging Face model
        # inputs = tokenizer(image_bytes, return_tensors="pt", padding=True)
        outputs = predict_step(filepath)
        captions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # Return the captions as JSON
        return jsonify(captions=captions)

    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

