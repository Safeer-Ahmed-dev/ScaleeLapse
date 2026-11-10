import torch
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor

def initialize_multimodal_model(decoder_name="facebook/bart-base", encoder_name="google/vit-base-patch16-224-in21k"):
    """
    Initializes a Vision-Encoder-Decoder model pairing a Vision Transformer 
    with a BART text decoder for image-to-text (caption & hashtag) generation.
    """
    print(f"Initializing Vision-Encoder-Decoder model using {encoder_name} and {decoder_name}...")
    
    # Build the model from pre-trained components
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_name, decoder_name
    )
    
    # Load corresponding tokenizer and image processor
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    image_processor = AutoImageProcessor.from_pretrained(encoder_name)
    
    # Set special token configurations for generation tasks
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # Hyperparameters for generation
    model.config.max_length = 128
    model.config.min_length = 10
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    return model, tokenizer, image_processor

if __name__ == "__main__":
    model, tokenizer, image_processor = initialize_multimodal_model()
    print("Model initialized successfully!")