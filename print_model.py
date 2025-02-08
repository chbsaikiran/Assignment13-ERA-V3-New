from model import build_llama_for_training

if __name__ == "__main__":

    # Initialize the trainer (this also builds the model)
    model = build_llama_for_training()

    # Print the model's summary or structure
    print(model)
