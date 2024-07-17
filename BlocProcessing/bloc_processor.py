import json
import csv
import gzip
from bloc.generator import add_bloc_sequences
from bloc.util import conv_tf_matrix_to_json_compliant
from bloc.util import get_bloc_variant_tf_matrix
from bloc.util import get_default_symbols
from bloc.util import getDictFromJson
from bloc.util import get_bloc_params

bot_dataset_files = [
    {'src': 'astroturf', 'classes': ['political_Bot']},
    {'src': 'kevin_feedback', 'classes': ['human', 'bot']},
    {'src': 'botwiki', 'classes': ['bot']},
    {'src': 'zoher-organization', 'classes': ['human', 'organization']},
    {'src': 'cresci-17', 'classes': ['human', 'bot-socialspam', 'bot-traditionspam', 'bot-fakefollower']},
    {'src': 'rtbust', 'classes': ['human', 'bot']},
    {'src': 'stock', 'classes': ['human', 'bot']},
    {'src': 'gilani-17', 'classes': ['human', 'bot']},
    {'src': 'midterm-2018', 'classes': ['human']},
    {'src': 'josh_political', 'classes': ['bot']},
    {'src': 'pronbots', 'classes': ['bot']},
    {'src': 'varol-icwsm', 'classes': ['bot', 'human']},
    {'src': 'gregory_purchased', 'classes': ['bot']},
    {'src': 'verified', 'classes': ['human']}
]


# Read UserId.txt file, and add it to a map
def get_user_id_class_map(file_path):
    user_id_class_map = {}
    all_classes = set()

    with open(file_path, 'r') as fd:
        for line in fd:
            parts = line.strip().split()
            if len(parts) >= 2:
                user_id, user_class = parts[0], parts[1]
                user_id_class_map[user_id] = user_class
                all_classes.add(user_class)

    return user_id_class_map, all_classes


# Return each twitter account into a bloc doc
def get_bloc_doc(u_bloc, bloc_model, user_id_class):
    doc = [u_bloc['bloc'][dim] for dim in bloc_model['bloc_alphabets'] if dim in u_bloc['bloc']]
    doc = ''.join(doc)
    doc = doc.strip()
    return {
        'text': doc,
        'user_id': u_bloc['user_id'],
        'screen_name': u_bloc['screen_name'],
        'src': 'IU',
        'class': user_id_class
    }


all_bloc_symbols = get_default_symbols()
gen_bloc_params, gen_bloc_args = get_bloc_params([], '', sort_action_words=True,
                                                 keep_bloc_segments=True,
                                                 tweet_order='noop')
bloc_model = {
    'name': 'm1: bigram',
    'ngram': 2,
    'token_pattern': '[^ |()*]',
    'tf_matrix_norm': 'l1',
    'keep_tf_matrix': True,
    'set_top_ngrams': True,
    'top_ngrams_add_all_docs': True,
    'bloc_variant': None,
    'bloc_alphabets': ['action', 'content_syntactic']
}

bloc_doc_lst = []
dataset_path = 'retraining_data'
for file in bot_dataset_files:
    tweets_file_path = dataset_path + file['src'] + '/tweets.jsons.gz'
    userid_file_path = dataset_path + file['src'] + '/userIds.txt'
    user_id_class_map, all_classes = get_user_id_class_map(userid_file_path)
    with gzip.open(tweets_file_path, 'rt', encoding='windows-1252') as infile:
        print(f"Processing Twitter file: {tweets_file_path}")
        for line in infile:
            line = line.split('\t')
            if user_id_class_map.get(line[0], '') in file['classes']:
                tweets = getDictFromJson(line[1])
                u_bloc = add_bloc_sequences(tweets, all_bloc_symbols=all_bloc_symbols, **gen_bloc_params)
                user_class = user_id_class_map.get(line[0], '')
                if file['src'] == 'astroturf':
                    user_class = 'bot'
                elif file['src'] == 'cresci-17' and (
                        user_class == 'bot-socialspam' or user_class == 'bot-traditionspam' or user_class == 'bot-fakefollower'):
                    user_class = 'bot'
                elif file['src'] == 'zoher-organization':
                    user_class = 'human'
                bloc_doc_lst.append(get_bloc_doc(u_bloc, bloc_model, user_class))
        print(f"Finished processing Twitter file: {tweets_file_path}")

tf_matrix = get_bloc_variant_tf_matrix(bloc_doc_lst,
                                       tf_matrix_norm=bloc_model['tf_matrix_norm'],
                                       keep_tf_matrix=bloc_model['keep_tf_matrix'],
                                       min_df=2,
                                       ngram=bloc_model['ngram'],
                                       token_pattern=bloc_model['token_pattern'],
                                       bloc_variant=bloc_model['bloc_variant'],
                                       set_top_ngrams=bloc_model.get('set_top_ngrams', False),
                                       top_ngrams_add_all_docs=bloc_model.get('top_ngrams_add_all_docs', False))

with gzip.open('tf_idf_mat.json.gz', 'wt', encoding='utf-8') as json_gz_file:
    json.dump(conv_tf_matrix_to_json_compliant(tf_matrix), json_gz_file, indent=2)

csv_file_path = 'features-bloc.csv'

with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    header = ['user_id', 'class'] + [f'{i}' for i in tf_matrix['vocab']]
    writer.writerow(header)

    for entry in tf_matrix['tf_matrix']:
        row = [entry['user_id'], entry['class']] + entry['tf_vector']
        writer.writerow(row)
