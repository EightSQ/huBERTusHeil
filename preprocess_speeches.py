import pandas as pd
import re
from tqdm import tqdm

to_replace = [
    'AFD',
    'AfD',
    'Alternative für Deutschland',
    'Bündnis 90',
    'Bündnis 90/Die Grünen',
    'CDU',
    'CDU/CSU',
    'CSU',
    'Christdemokraten',
    'Christlich-Demokratische Union',
    'Christlich-Demokratischen Union',
    'Christlich-Soziale Union'
    'Christlich-Sozielen Union'
    'Christsoziale',
    'Die Grünen',
    'Die Linke',
    'Die Linken',
    'FDP',
    'Freie Demokraten',
    'Freie Demokratische Partei',
    'Freien Demokraten',
    'Groko',
    'Grüne',
    'Grünen',
    'Junge Alternative',
    'Jungen Alternative',
    'Liberale',
    'Liberale',
    'Liberalen',
    'Liberalen',
    'Linke',
    'Linken',
    'Linkspartei',
    'Mauerpartei'
    'Regierungskoalition',
    'SED',
    'SPD',
    'SPÖ',
    'Sozialdemokrat',
    'Sozialdemokraten',
    'Sozialdemokratie',
    'Sozialdemokratische Partei Deutschlands',
    'Sozialdemokratische Partei',
    'Union und SPD',
    'Union',
    'Unionsfraktion',
    'Wir als Alternative',
    'blau',
    'christlich-sozial',
    'den Grünen',
    'den Linken',
    'der Linkspartei',
    'die Grünen',
    'die Linken',
    'die Linkspartei',
    'freie Demokraten',
    'freien Demokraten',
    'gelb',
    'große Koalition',
    'grün',
    'grüne Partei',
    'grüne',
    'grünen',
    'linke Partei',
    'rot',
    'schwarz',
    'sozialdemokratisch',
    'wir als Alternative'
]

labels = ["AfD", "B90", "Union", "Linke", "FDP", "SPD", "fraktionslos"]

def main():
    e = re.compile(f'({"|".join(to_replace)})')
    df = pd.read_csv('rede_fraktion.csv')
    df = df.dropna()
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            row['text'] = e.sub('[UNK]', row['text'])
        except:
            print(row['text'])
        
    df = df[df['fraktion'] != 'Bremen']
    m = {
        'DIE LINKE': 'Linke',
        'BÜNDNIS\xa090/DIE GRÜNEN' : 'B90',
        'CDU/CSU': 'Union',
        'BÜNDNIS 90/DIE GRÜNEN': 'B90',
        'Fraktionslos': 'fraktionslos',
        'Bündnis 90/Die Grünen': 'B90'
    }
    for k in m.keys():
        df.loc[df['fraktion'] == k, 'fraktion'] = m[k]

    df['top'] = df['top'].apply(lambda x: re.sub(r'[\x00-\x1f]|[\x7f-\xa0]', r' ', x))

    df.to_csv('rede_fraktion_preprocessed.csv', index=False)

if __name__ == "__main__":
    main()
