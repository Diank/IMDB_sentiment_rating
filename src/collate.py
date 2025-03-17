import torch

# выравниваем отзывы до макс длины в баче
def collate_with_padding(input_batch, pad_id):

    reviews = [item['review'] for item in input_batch]
    sentiment = [item['sentiment'] for item in input_batch]
    rating = [item['rating'] for item in input_batch]

    max_seq_len = max([len(seq) for seq in reviews])

    padded_reviews = []
    for seq in reviews:
        padded_seq = seq.tolist() + [pad_id] * (max_seq_len - len(seq))
        padded_reviews.append(padded_seq)

    new_batch = {
        'review': torch.tensor(padded_reviews, dtype=torch.long),
        'sentiment': torch.tensor(sentiment, dtype=torch.long),
        'rating': torch.tensor(rating, dtype=torch.float)
    }

    return new_batch
