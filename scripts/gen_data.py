import pandas as pd
import random

pos_df = pd.read_csv('data/sentiment_feedback.csv')
neu_df = pd.read_csv('data/neutral_feedback.csv')
neg_df = pd.read_csv('data/negative_feedback.csv')

tar_df = {
    'text': [],
    'sentiment': []
}
vars = [0, 1, 2]
for i in range((len(pos_df))):
    print(f'ITER : {i}')
    choice = random.choice(vars)
    if choice == 0:
        tar_df['text'].append(pos_df['text'][i])
        tar_df['sentiment'].append(pos_df['sentiment'][i])
        n_choice = [1, 2]
        choice2 = random.choice(n_choice)
        if n_choice == 1:
            tar_df['text'].append(neu_df['text'][i])
            tar_df['sentiment'].append(neu_df['sentiment'][i])
            tar_df['text'].append(neg_df['text'][i])
            tar_df['sentiment'].append(neg_df['sentiment'][i])
        else:
            tar_df['text'].append(neg_df['text'][i])
            tar_df['sentiment'].append(neg_df['sentiment'][i])
            tar_df['text'].append(neu_df['text'][i])
            tar_df['sentiment'].append(neu_df['sentiment'][i])
            
    elif choice == 1:
        tar_df['text'].append(neu_df['text'][i])
        tar_df['sentiment'].append(neu_df['sentiment'][i])
        n_choice = [0, 2]
        choice2 = random.choice(n_choice)
        if n_choice == 0:
            tar_df['text'].append(pos_df['text'][i])
            tar_df['sentiment'].append(pos_df['sentiment'][i])
            tar_df['text'].append(neg_df['text'][i])
            tar_df['sentiment'].append(neg_df['sentiment'][i])
        else:
            tar_df['text'].append(neg_df['text'][i])
            tar_df['sentiment'].append(neg_df['sentiment'][i])
            tar_df['text'].append(pos_df['text'][i])
            tar_df['sentiment'].append(pos_df['sentiment'][i])       
            
    elif choice == 2:
        tar_df['text'].append(neg_df['text'][i])
        tar_df['sentiment'].append(neg_df['sentiment'][i])
        n_choice = [0, 1]
        choice2 = random.choice(n_choice)
        if n_choice == 0:
            tar_df['text'].append(pos_df['text'][i])
            tar_df['sentiment'].append(pos_df['sentiment'][i])
            tar_df['text'].append(neu_df['text'][i])
            tar_df['sentiment'].append(neu_df['sentiment'][i])
        else:
            tar_df['text'].append(neu_df['text'][i])
            tar_df['sentiment'].append(neu_df['sentiment'][i])
            tar_df['text'].append(pos_df['text'][i])
            tar_df['sentiment'].append(pos_df['sentiment'][i])    
            
df = pd.DataFrame.from_dict(tar_df)

df.to_csv('full_dataset.csv')        