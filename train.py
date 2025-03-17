import torch
import torch.optim as optim
import torch.nn as nn
from model import TransformerModel
from data_processing import TextProcessor

def train_model(model, dataloader, epochs=10, lr=0.001):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.T, targets.T
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.view(-1, model.fc_out.out_features), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

if __name__ == "__main__":
    with open("../data/sample.txt", "r") as f:
        text = f.read()
    
    processor = TextProcessor(text)
    dataloader = processor.get_batches()
    
    model = TransformerModel(vocab_size=processor.vocab_size)
    train_model(model, dataloader)
