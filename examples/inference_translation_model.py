"""
Inference script for English-Chinese translation.
Loads trained model and allows interactive translation.
"""

import torch
import os
from demo_transformer import Transformer, TransformerInference


def tokenize_sentence(sentence, vocab, is_chinese=False):
    """Tokenize sentence using vocabulary."""
    if is_chinese:
        tokens = list(sentence.replace(" ", ""))
    else:
        tokens = sentence.lower().split()
    
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]


def load_model_and_vocabs(save_dir="./translation_model", use_best=True):
    """Load trained model and vocabularies."""
    print(f"Loading model from {save_dir}...")
    
    if not os.path.exists(save_dir):
        print(f"Error: Model directory {save_dir} not found!")
        print("Please run 'uv run python examples/train_translation_model.py' first.")
        return None, None, None, None
    
    # Try to load best model first
    best_model_dir = os.path.join(save_dir, "best_model")
    if use_best and os.path.exists(best_model_dir):
        print("Loading best model...")
        model_dir = best_model_dir
    else:
        print("Loading latest model...")
        model_dir = save_dir
    
    # Load complete model file
    model_path = os.path.join(model_dir, 'model.pt')
    print(f"Full model path: {os.path.abspath(model_path)}")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return None, None, None, None
    
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Extract everything from the checkpoint
        config = checkpoint['config']
        en_vocab = checkpoint['en_vocab']
        zh_vocab = checkpoint['zh_vocab']
        
        # Load model
        model = Transformer(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        
        print("Model loaded successfully!")
        print(f"Model type: {'Best' if use_best and os.path.exists(best_model_dir) else 'Latest'}")
        print(f"English vocabulary size: {len(en_vocab):,}")
        print(f"Chinese vocabulary size: {len(zh_vocab):,}")
        print(f"Using device: {device}")
        
        # Display model stats
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        return model, config, en_vocab, zh_vocab
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None


def translate_text(model, config, en_vocab, zh_vocab, text, max_output_len=50):
    """Translate English text to Chinese."""
    # Create reverse vocabulary for decoding
    zh_id_to_token = {v: k for k, v in zh_vocab.items()}
    
    # Initialize inference
    inference = TransformerInference(model)
    device = next(model.parameters()).device
    
    # Tokenize input
    tokens = tokenize_sentence(text, en_vocab, is_chinese=False)
    print(f"Tokenized '{text}' -> {tokens}")
    if not tokens:
        return "<empty>"
    
    # Handle unknown tokens
    unk_count = sum(1 for token in tokens if token == en_vocab["<unk>"])
    if unk_count > len(tokens) * 0.5:  # If more than 50% unknown tokens
        return f"<too many unknown words: {unk_count}/{len(tokens)}>"
    
    src_ids = torch.tensor([tokens], device=device)
    # Create proper padding mask - None for no padding
    src_padding_mask = None
    
    print(f"Source input to encoder: {src_ids}")
    print(f"Decoder will start with SOS token: {config.sos_token_id}")
    
    try:
        # Greedy decoding
        with torch.no_grad():
            print(f"Starting greedy decode with SOS={config.sos_token_id}, EOS={config.eos_token_id}")
            output = inference.greedy_decode(
                src_ids, src_padding_mask, max_output_len=max_output_len
            )
        
        # Decode output
        if isinstance(output, torch.Tensor):
            output_tokens = output[0].cpu().tolist()
        else:
            output_tokens = output  # Already a list from greedy_decode
        
        print(f"Generated tokens: {output_tokens}")
        decoded_tokens = []
        
        for token_id in output_tokens:
            if token_id == zh_vocab["<eos>"]:
                break
            if token_id not in [zh_vocab["<sos>"], zh_vocab["<pad>"]]:
                token = zh_id_to_token.get(token_id, "<unk>")
                if token != "<unk>":
                    decoded_tokens.append(token)
        
        print(f"Decoded tokens: {decoded_tokens}")
        translation = "".join(decoded_tokens) if decoded_tokens else "<empty>"
        print(f"Final translation: {translation}")
        return translation
        
    except Exception as e:
        return f"<translation error: {e}>"


def run_batch_translation(model, config, en_vocab, zh_vocab, test_sentences):
    """Run translation on a batch of test sentences."""
    print("\nBatch Translation Results:")
    print("-" * 50)
    
    for i, sentence in enumerate(test_sentences, 1):
        translation = translate_text(model, config, en_vocab, zh_vocab, sentence)
        print(f"{i:2d}. EN: {sentence}")
        print(f"    ZH: {translation}")
        print()


def interactive_translation(model, config, en_vocab, zh_vocab):
    """Interactive translation mode."""
    print("\n" + "=" * 50)
    print("Interactive Translation Mode")
    print("Type English text to translate to Chinese")
    print("Commands: 'quit', 'exit', 'q' to exit")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\nEnter English text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', '']:
                break
                
            if user_input:
                print("Translating...")
                translation = translate_text(model, config, en_vocab, zh_vocab, user_input)
                print(f"Chinese: {translation}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")


def main():
    """Main inference function."""
    print("English-Chinese Translation Model Inference")
    print("=" * 50)
    
    # Ask which model to load
    print("\nWhich model would you like to use?")
    print("1. Best model (recommended)")
    print("2. Latest model")
    choice = input("Enter choice (1 or 2, default=1): ").strip()
    use_best = choice != '2'
    
    # Load model
    model, config, en_vocab, zh_vocab = load_model_and_vocabs(use_best=use_best)
    
    if model is None:
        return
    
    # Test sentences for batch translation
    test_sentences = [
        "Hello",
        "How are you",
        "Thank you",
        "Good morning",
        "I love you",
        "What is your name",
        "Nice to meet you",
        "Where are you from",
        "I am fine",
        "The weather is nice today",
        "I want to learn Chinese",
        "This is a beautiful day",
        "Can you help me",
        "I don't understand",
        "Please speak slowly"
    ]
    
    # Run batch translation
    run_batch_translation(model, config, en_vocab, zh_vocab, test_sentences)
    
    # Interactive mode
    interactive_translation(model, config, en_vocab, zh_vocab)


if __name__ == "__main__":
    main()