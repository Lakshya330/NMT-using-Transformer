
def translate(model, data, german, english, device, max_length=30):
    model.eval()

    translated = []

    ## If the input data is a single sentence
    with torch.no_grad():
        if isinstance(data, str):
            tokenized = [token.text.lower() for token in spacy_german.tokenizer(data)]
            tokenized = [german.init_token] + tokenized + [german.eos_token]
            input_ids = [german.vocab.stoi[token] for token in tokenized]
            input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)

            with torch.no_grad():
                encoder_output = model.encoder(input_tensor)

            decoded = [english.vocab.stoi[english.init_token]]

            for _ in range(max_length):
                decoder_input = torch.LongTensor(decoded).unsqueeze(0).to(device)
                trg_mask = torch.tril(torch.ones(len(decoded), len(decoded))).unsqueeze(0).unsqueeze(0).to(device)

                output = model.decoder(decoder_input, encoder_output, trg_mask)

                next_token_logits = output[:, -1, :]

                next_token = next_token_logits.argmax(1).item()

                decoded.append(next_token)

                translated.append(english.vocab.itos[next_token])

                if next_token == english.vocab.stoi[english.eos_token]:
                    break



    ## If the input data is multiple sentences
    with torch.no_grad():
        if isinstance(data, list):
            batch_size = len(data)
            tokenized = [[token.text for token in spacy_german.tokenizer(sentence)] for sentence in data]
            tokenized = [[german.init_token] + seq + [german.eos_token] for seq in tokenized]

            input_ids = [[german.vocab.stoi[token] for token in tokens] for tokens in tokenized]
            input_tensor = torch.LongTensor(input_ids).to(device)

            with torch.no_grad():
                encoder_output = model.encoder(input_tensor)

            translated_sentences = [[english.vocab.stoi[english.init_token]] for _ in range(batch_size)]

            for batch_idx, sample in enumerate(translated_sentences):

                trg_mask = torch.tril(torch.ones(len(translated_sentences[batch_idx]), len(translated_sentences[batch_idx]))).unsqueeze(0).unsqueeze(0).to(device)

                for _ in range(max_length):

                    decoder_input = torch.LongTensor(translated_sentences[batch_idx]).unsqueeze(0).to(device)
                    output = model.decoder(decoder_input, encoder_output[batch_idx, :, :].unsqueeze(0), trg_mask)

                    next_token_logits = output[:, -1, :]
                    next_token = next_token_logits.argmax(1).item()

                    translated_sentences[batch_idx].append(next_token)

                    if next_token == english.vocab.stoi[english.eos_token]:
                        break

            for translations in translated_sentences:
                translated.append([english.vocab.itos[index] for index in translations])

    return translated


sentence = "ein Mann, der auf einer Straße läuft"


output = translate(transformer, sentence, german, english, device)
print(output)
