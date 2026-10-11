from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations
import Levenshtein

project_root = Path(__file__).resolve().parent.parent
raw_data = project_root / 'data' / 'raw' / 'TMDB_movie_dataset_v11.csv'
cleaned_data = project_root / 'data' / 'cleaned' / 'cleaned_data.csv'

def load_data():
    print('Loading raw data...')
    df = pd.read_csv(raw_data)
    print(f'Loaded {len(df)} rows.')
    return df

def clean_values(dataframe):
    print('Cleaning columns revenue, runtime, budget, vote_count and vote_average...')
    dataframe.loc[dataframe['revenue'] <= 0, 'revenue'] = np.nan
    dataframe.loc[dataframe['runtime'] <= 0, 'runtime'] = np.nan
    dataframe.loc[dataframe['budget'] <= 0, 'budget'] = np.nan
    dataframe.loc[dataframe['vote_count'] == 0, 'vote_average'] = np.nan
    print('Cleaned invalid values for columns.')
    return dataframe

def deduplicate_id(dataframe):
    print('Deleting real duplicates and ID duplicates...')
    rows_before = len(dataframe)
    dataframe = dataframe.drop_duplicates().copy()
    dataframe['completeness_score'] = dataframe.notna().sum(axis=1)
    dataframe = dataframe.sort_values(by='completeness_score', ascending=False,
    kind='stable')
    dataframe = dataframe.drop_duplicates(subset='id', keep='first')
    rows_after = len(dataframe)
    dataframe = dataframe.drop(columns='completeness_score')
    print('Deleted duplicates and ID duplicates.')
    print(f'Rows removed after ID deduplication: {rows_before - rows_after}.')
    return dataframe

def deduplicate_imdb(dataframe):
    print('Deleting IMDB duplicates...')
    imdb_duplicates = dataframe[dataframe['imdb_id'].notna()
        & dataframe['imdb_id'].duplicated(keep=False)].copy()
    pair_results = []
    for imdb_id, group in imdb_duplicates.groupby('imdb_id'):
        for idx1, idx2 in combinations(group.index, 2):
            title1 = group.loc[idx1, 'original_title']
            title2 = group.loc[idx2, 'original_title']
            if pd.isna(title1) or pd.isna(title2):
                continue
            similarity = Levenshtein.ratio(title1.casefold(),title2.casefold())
            year1 = pd.to_datetime(group.loc[idx1, 'release_date'], errors='coerce').year
            year2 = pd.to_datetime(group.loc[idx2, 'release_date'], errors='coerce').year
            year_difference = abs(year1 - year2)
            pair_results.append({
                'imdb_id': imdb_id,
                'idx1': idx1,
                'idx2': idx2,
                'year_difference': year_difference,
                'similarity': similarity})
    pair_df = pd.DataFrame(pair_results)
    matches = pair_df[(pair_df['similarity'] >= 0.80)& ((pair_df['year_difference'] <= 1)
        | pair_df['year_difference'].isna())]
    matching_pairs = {frozenset((row.idx1, row.idx2))
                      for row in matches.itertuples()}
    completeness_scores = dataframe.notna().sum(axis=1)
    rows_to_remove = []
    for imdb_id, group in imdb_duplicates.groupby('imdb_id'):
        ranked_indices = completeness_scores.loc[group.index].sort_values(
            ascending=False,
            kind='stable').index
        kept_indices = []
        for idx in ranked_indices:
            is_duplicate = any(
                frozenset((idx, kept_idx)) in matching_pairs
                for kept_idx in kept_indices)
            if is_duplicate:
                rows_to_remove.append(idx)
            else:
                kept_indices.append(idx)
    dataframe = dataframe.drop(index=rows_to_remove).copy()
    print('Deleted IMDB duplicates.')
    print(f'Rows remaining after IMDb deduplication: {len(dataframe)}')       
    return dataframe

def clean_dates(dataframe):
    print('Cleaning column release_date...')
    dataframe['release_date'] = pd.to_datetime(dataframe['release_date'],
                                               errors='coerce')
    print('Cleaned dates.')
    return dataframe
    
def save_data(dataframe):
    print('Saving cleaned data...')
    cleaned_data.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(cleaned_data, index=False)
    print(f'Saved {len(dataframe)} rows to {cleaned_data}.')
    
def main():
    df = load_data()
    df = clean_values(df)
    df = deduplicate_id(df)
    df = deduplicate_imdb(df)
    df = clean_dates(df)
    print(f'Final row count: {len(df)}')
    save_data(df)

if __name__ == '__main__':
    main()