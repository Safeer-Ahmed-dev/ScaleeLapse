from datasets import load_dataset
import pandas as pd
import os

print("Downloading clean fashion product dataset from Hugging Face...")
ds = load_dataset("ashraq/fashion-product-images-small", split="train[:2000]")

image_dir = "data/raw/images"
os.makedirs(image_dir, exist_ok=True)

metadata_records = []

print("Processing images, generating captions, and building metadata...")
for idx, item in enumerate(ds):
    image = item["image"]
    product_name = item.get("productDisplayName", "Modern Apparel Collection")
    sub_category = item.get("subCategory", "Apparel")
    
    # Ensure image is RGB format
    if image.mode != "RGB":
        image = image.convert( "RGB")
        
    filename = f"img_{idx:04d}.jpg"
    img_path = os.path.join(image_dir, filename)
    
    # Save physical image locally
    image.save(img_path)
    
    # Construct a natural marketing caption + automated tags from the product data
    marketing_caption = f"Check out this stunning {product_name}. Perfect for your everyday {sub_category.lower()} style! #{sub_category.lower()} #fashionstyle #ootd #brandnew"
    
    metadata_records.append({
        "image_path": img_path,
        "caption": marketing_caption
    })

# Save the master metadata CSV locally
df = pd.DataFrame(metadata_records)
csv_path = "data/raw/metadata.csv"
df.to_csv(csv_path, index=False)

print(f"Success! Saved {len(metadata_records)} commercial images and generated metadata.csv with captions & hashtags.")