import os
import torch
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from dataset import ScaleeLapseDataset

def default_data_collator_multimodal(features):
    """
    Custom data collator to correctly stack pixel values and pad label tensors 
    for the Vision-Encoder-Decoder multimodal architecture.
    """
    pixel_values = torch.stack([torch.tensor(f["pixel_values"]) for f in features])
    labels = torch.stack([torch.tensor(f["labels"]) for f in features])
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }

def train_model():
    # 1. Configuration & Paths
    decoder_checkpoint = "facebook/bart-base"
    encoder_checkpoint = "google/vit-base-patch16-224-in21k"
    csv_path = "data/raw/metadata.csv"
    output_dir = "models/scaleelapse_checkpoint"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing tokenizers and image processors for BART...")
    tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
    image_processor = AutoImageProcessor.from_pretrained(encoder_checkpoint)
    
    # 2. Load or initialize Vision-Encoder-Decoder model with BART decoder
    print("Loading Vision-Encoder-Decoder model...")
    try:
        model = VisionEncoderDecoderModel.from_pretrained(output_dir)
    except:
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_checkpoint, decoder_checkpoint
        )
        
    # Set special token configurations for BART generation
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # 3. Initialize Dataset
    print("Loading local dataset...")
    train_dataset = ScaleeLapseDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        feature_extractor=image_processor,
        max_target_length=128
    )

    # 4. Training Arguments optimized for local execution
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        save_steps=100,
        logging_steps=10,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    # 5. Initialize Trainer with custom data collator
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator_multimodal,
    )

    print("Starting model fine-tuning pipeline...")
    trainer.train()

    # Save final fine-tuned model weights
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    image_processor.save_pretrained(output_dir)
    print(f"Training complete! Model saved successfully to {output_dir}/")

if __name__ == "__main__":
    train_model()