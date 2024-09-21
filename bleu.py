import sacrebleu
from tqdm.notebook import tqdm

# Reference translations

references = []
candidates = []
translated_sentences = []
correct_sentences = []

for batch in tqdm(test_iterator):
    src = batch.src.to(device)
    trg = batch.trg.to(device)

    trg_input = trg[:, :-1]  # Ignore <eos> token
    trg_output = trg[:, 1:]  # Shift to right

    # Generate target mask
    seq_len = trg_input.shape[1]
    trg_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        output = transformer(src, trg_input, trg_mask)

    prediction = torch.argmax(output, dim = -1)

    for correct_sentence, predicted_sentence in zip(trg_output, prediction):
        decoded_predicted_sentence = []
        decoded_correct_sentence = []

        for correct_token, predicted_token in zip(correct_sentence, predicted_sentence):
            decoded_correct_sentence.append(english.vocab.itos[correct_token])
            decoded_predicted_sentence.append(english.vocab.itos[predicted_token])

        correct_sentences.append(decoded_correct_sentence)
        translated_sentences.append(decoded_predicted_sentence)


for correct, translated in zip(correct_sentences, translated_sentences):
    references.append(" ".join(correct))
    candidates.append(" ".join(translated))

# Calculate BLEU score
bleu = sacrebleu.corpus_bleu(candidates, [references])
print(f"BLEU score: {bleu.score:.4f}")
