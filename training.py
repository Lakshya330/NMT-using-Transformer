from tqdm.notebook import tqdm
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


def translate_sentence(model, sentence, german, english, device, max_length=30):
    model.eval()

    # Tokenize the sentence
    tokenized = [token.text.lower() for token in spacy_german.tokenizer(sentence)]
    tokenized = [german.init_token] + tokenized + [german.eos_token]

    # Convert tokens to indices
    token_indices = [german.vocab.stoi[token] for token in tokenized]

    # Convert to tensor and move to device
    sentence_tensor = torch.LongTensor(token_indices).unsqueeze(0).to(device)

    # Pass through encoder
    with torch.no_grad():
        encoder_out = model.encoder(sentence_tensor)

    # Initialize target sequence with <sos> token
    translated_tokens = [english.vocab.stoi[english.init_token]]

    for _ in range(max_length):
        # Convert last token to tensor
        target_tensor = torch.LongTensor(translated_tokens).unsqueeze(0).to(device)

        # Generate mask for decoder
        trg_mask = torch.tril(torch.ones((len(translated_tokens), len(translated_tokens))).unsqueeze(0).unsqueeze(0).to(device))

        # Decode
        with torch.no_grad():
            output = model.decoder(target_tensor, encoder_out, trg_mask)

        # Get the last predicted token
        next_token_logits = output[:, -1, :]
        next_token = next_token_logits.argmax(1).item()

        # Append the predicted token to the sequence
        translated_tokens.append(next_token)

        # Break if <eos> token is predicted
        if next_token == english.vocab.stoi[english.eos_token]:
            break

    # Convert indices back to tokens
    translated_sentence = [english.vocab.itos[token] for token in translated_tokens]
    return translated_sentence

# Define loss function and optimizer
loss_fn = CrossEntropyLoss(ignore_index=german.vocab.stoi[german.pad_token])
optimizer = Adam(params=transformer.parameters())

# Example sentence to translate
sentence = "Ein Mann in einem blauen Hemd steht auf einer Leiter und reinigt ein Fenster"

num_epochs = 100

for epoch in range(num_epochs):
    train_loss = 0.0
    dev_loss = 0.0

    translated = translate_sentence(transformer, sentence, german, english, device)
    print(f"Translated Sentence after epoch {epoch}:")
    print(translated)

    for batch in tqdm(train_iterator, desc=f"Epoch {epoch + 1} Train Iterator"):
        src = batch.src.to(device)
        trg = batch.trg.to(device)

        trg_input = trg[:, :-1]  # Ignore <eos> token
        trg_output = trg[:, 1:]  # Shift to right

        # Generate target mask
        seq_len = trg_input.shape[1]
        trg_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0).unsqueeze(0)

        # Forward pass
        output = transformer(src, trg_input, trg_mask)

        # Reshape for loss calculation
        output = output.reshape(-1, output.shape[-1])
        trg_output = trg_output.reshape(-1)

        # Calculate loss
        loss = loss_fn(output, trg_output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    writer.add_scalar("Training Loss", train_loss, epoch)

    for batch in tqdm(dev_iterator, desc = f"Epoch {epoch + 1} Dev Iterator"):
        src = batch.src.to(device)
        trg = batch.trg.to(device)

        trg_input = trg[:, :-1]
        trg_output = trg[:, 1:].reshape(-1)

        seq_len = trg_input.shape[1]

        trg_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = transformer(src, trg_input, trg_mask)

        output = output.reshape(-1, output.shape[2])
        trg_output = trg_output

        loss = loss_fn(output, trg_output)
        dev_loss += loss.item()

    writer.add_scalar("Developement Loss", dev_loss, epoch)
    print(f"Epoch {epoch + 1} Train Loss: {train_loss / len(train_iterator)} Dev Loss: {dev_loss / len(dev_iterator)}", end = "\n\n")

path = "/content/Transformer/state_dict.pth"
torch.save(transformer.state_dict(), path)
