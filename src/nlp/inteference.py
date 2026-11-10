import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor

def generate_caption(image_path):
    model_dir = "models/scaleelapse_checkpoint"
    
    print("Loading fine-tuned model and processors...")
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    image_processor = AutoImageProcessor.from_pretrained(model_dir)
    
    model.eval()

    # Load and preprocess test image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

    print("Generating marketing caption and hashtags...")
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=128,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

    # Decode tokens back into text
    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print("\n--- ScaleeLapse Generation Result ---")
    print(result)
    print("---------------------------------------")

if __name__ == "__main__":
    # Point this to any test image path in your dataset or computer
    test_image = "data/raw/brandimages/Khaadi/khaadi_001_ivory_terracotta_floral.png" 
    generate_caption(test_image)