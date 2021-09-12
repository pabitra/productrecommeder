import pandas as pd
import pickle


def recommend_it(num_recommendations=20, ruserId='02dakota'):
    predictions_df = pickle.load(open('./model/preds_df.pkl', 'rb'))
    itm_df = pickle.load(open('./model/items_df.pkl', 'rb'))
    original_ratings_df = pickle.load(open('./model/df1.pkl', 'rb'))

    # Get and sort the user's predictions
    sorted_user_predictions = predictions_df.loc[ruserId].sort_values(
        ascending=False)

    # Get the user's data and merge in the item information.
    user_data = original_ratings_df[original_ratings_df.reviews_username == ruserId]
    
    user_full = (user_data.merge(itm_df, how='left', left_on='name',
                                 right_on='name').sort_values(['reviews_rating'], ascending=False))
    
    print('User {0} has already purchased {1} items.'.format(
        ruserId, user_full.shape[0]))
    print('Recommending the highest {0} predicted  items not already purchased.'.format(
        num_recommendations))

    # Recommend the highest predicted rating items that the user hasn't bought yet.
    recommendations = (itm_df[~itm_df['name'].isin(user_full['name'])].merge(pd.DataFrame(sorted_user_predictions).reset_index(
    ), how='left', left_on='name', right_on='name').rename(columns={ruserId: 'Predictions'}).sort_values('Predictions', ascending=False).iloc[:num_recommendations, :-1])
    topk = recommendations.merge(
        original_ratings_df, right_on='name', left_on='name').drop_duplicates(['name'])[['name', 'cleaned_text']]
    return topk


print(recommend_it(10))
