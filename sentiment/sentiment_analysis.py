import pandas as pd
import jieba

lexicon_path = r'D:\download\spider-topic\qingan\Sentiment vocabulary labeling dataset.csv'    # Sentiment vocabulary labeling dataset CSV
data_path    = r'D:\download\spider-topic\qingan\final_data1.csv'           # test CSV 
output_path  = r'D:\download\spider-topic\qingan\weibo_with_highscores.csv'    # output

# Read the sentiment dictionary and specify the encoding
lex_df = pd.read_csv(lexicon_path, encoding='gb18030')
# The first column is the words, and the third is the intensity
lex_dict = dict(zip(lex_df.iloc[:, 0], lex_df.iloc[:, 2]))

# 2. Read data to be tested

df = pd.read_csv(data_path, encoding='utf-8-sig', parse_dates=['publish_time'])

# 3. Define a normalized scoring function
def score_sentiment(text: str) -> float:
    tokens = jieba.lcut(str(text))
    total, emo_cnt = 0, 0
    for w in tokens:
        intensity = lex_dict.get(w)
        if intensity is not None:
            total += intensity
            emo_cnt += 1
    return total / emo_cnt if emo_cnt > 0 else 0.0

# 4. Apply a scoring function
df['sentiment_intensity'] = df['content'].apply(score_sentiment)

# 5. save result
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"the result is in {output_path}")