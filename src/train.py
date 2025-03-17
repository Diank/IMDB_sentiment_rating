import torch
from tqdm import tqdm

from sklearn.metrics import accuracy_score, mean_absolute_error


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, mode='min', verbose=True, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode  # 'min' for loss/MAE, 'max' for accuracy
        self.verbose = verbose
        self.path = path  # куда сохранять модель

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model)
        elif self._is_improvement(current_score):
            self.best_score = current_score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, current_score):
        if self.mode == 'min':
            return current_score < self.best_score - self.min_delta
        else:
            return current_score > self.best_score + self.min_delta

    def save_checkpoint(self, model):
        if self.verbose:
            print(f"Saving model with score: {self.best_score:.4f}")
        torch.save(model.state_dict(), self.path)


# Масштабирует вероятность класса 1 в рейтинг [1, 4] для отрицательного отзыва,
# [7, 10] для положительного отзыва 
def map_probability_to_rating(probs):

    predicted_class = torch.argmax(probs, dim=1)  # [batch]

    rating_pred = torch.zeros_like(predicted_class, dtype=torch.float)  # [batch]

    mask_pos = (predicted_class == 1)  # для класса 1 (положительный отзыв)
    rating_pred[mask_pos] = 7 + (10 - 7) * probs[mask_pos, 1]

    mask_neg = (predicted_class == 0) # для класса 0 (отрицательный отзыв)
    rating_pred[mask_neg] = 1 + (4 - 1) * probs[mask_neg, 0]

    return rating_pred


def train(model, dataloader, optimizer, criterion_sentiment, criterion_rating, 
          alpha, beta, epoch, device):

    model.train()

    total_loss = 0
    total_rating_loss = 0
    total_sentiment_loss = 0

    for batch in tqdm(dataloader, desc=f'Training epoch: {epoch+1}'):

        review = batch['review'].to(device)
        sentiment = batch['sentiment'].to(device)
        rating = batch['rating'].to(device)

        optimizer.zero_grad()

        sentiment_logits = model(review)

        probs = torch.softmax(sentiment_logits, dim=1)
        rating_pred = map_probability_to_rating(probs)

        loss_sentiment = criterion_sentiment(sentiment_logits, sentiment)
        loss_rating = criterion_rating(rating_pred, rating)

        loss = alpha * loss_sentiment + beta * loss_rating

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_rating_loss += loss_rating.item()
        total_sentiment_loss += loss_sentiment.item()

    print(f"Loss sentiment: {loss_sentiment.item():.4f}, Loss rating: {loss_rating.item():.4f}, Total: {loss.item():.4f}")



def evaluate(model, dataloader, criterion_sentiment, criterion_rating, 
             alpha, beta, device):
  
    model.eval()

    total_loss = 0
    all_sentiment_preds = []
    all_sentiment_labels = []
    all_rating_preds = []
    all_rating_labels = []

    with torch.no_grad():
        for batch in dataloader:
            
            review = batch['review'].to(device)
            sentiments = batch['sentiment'].to(device)
            ratings = batch['rating'].to(device)

            sentiment_logits = model(review)

            probs = torch.softmax(sentiment_logits, dim=1)
            rating_pred = map_probability_to_rating(probs)

            loss_sentiment = criterion_sentiment(sentiment_logits, sentiments)
            loss_rating = criterion_rating(rating_pred, ratings)
            loss = alpha * loss_sentiment + beta * loss_rating
            total_loss += loss.item()

            preds_sentiment = torch.argmax(sentiment_logits, dim=1)
            all_sentiment_preds.extend(preds_sentiment.cpu().numpy())
            all_sentiment_labels.extend(sentiments.cpu().numpy())

            all_rating_preds.extend(rating_pred.cpu().numpy())
            all_rating_labels.extend(ratings.cpu().numpy())


    accuracy = accuracy_score(all_sentiment_labels, all_sentiment_preds)
    mae = mean_absolute_error(all_rating_labels, all_rating_preds)

    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy, mae
