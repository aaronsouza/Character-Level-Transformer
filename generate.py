import torch
from model import TransformerModel
from data_processing import TextProcessor

def generate_text(model, processor, start_text="T", length=100):
    model.eval()
    chars = [processor.char2idx[c] for c in start_text]
    
    with torch.no_grad():
        for _ in range(length):
            input_seq = torch.tensor(chars[-processor.seq_length:]).unsqueeze(1)
            output = model(input_seq)
            next_char = torch.argmax(output[-1]).item()
            chars.append(next_char)
    
    return ''.join([processor.idx2char[i] for i in chars])

if __name__ == "__main__":
    with open("../data/sample.txt", "r") as f:
        text = f.read()
    
    processor = TextProcessor(text)
    model = TransformerModel(vocab_size=processor.vocab_size)
    model.load_state_dict(torch.load("model.pth"))
    
    print(generate_text(model, processor, start_text="Hello"))
