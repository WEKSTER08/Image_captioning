import os
import urllib.request
from flask import Flask, request, jsonify, render_template
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

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
      # Get the uploaded file and save it to disk
      file = request.files["file"]
      filename = file.filename
      filepath = os.path.join("uploads", filename)
      file.save(filepath)        
      i_image = Image.open(filepath)
      if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

      # images.append(i_image)

      pixel_values = feature_extractor(images=i_image, return_tensors="pt").pixel_values
      pixel_values = pixel_values.to(device)
      num_captions = int(request.form.get("num_captions", 1))
      output_ids = model.generate(pixel_values, **gen_kwargs,num_return_sequences=num_captions)

      preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
      preds = [pred.strip() for pred in preds]
      print(preds)
      return jsonify(captions=preds)

    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

