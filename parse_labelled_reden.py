from pathlib import Path
from lxml import etree
from tqdm import tqdm
import pandas as pd

def parse_rede(reden, top):
    for rede in reden:
        try:
            rede_fraktion = rede.xpath("p[@klasse = 'redner']/redner/name/fraktion/text()")[0]
            redner_vorname = rede.xpath("p[@klasse = 'redner']/redner/name/vorname/text()")[0]
            redner_nachname = rede.xpath("p[@klasse = 'redner']/redner/name/nachname/text()")[0]
            texts = []
            for child in rede.xpath("p[@klasse = 'J_1' or @klasse = 'J' or @klasse = 'O']"):
                if child.text:
                    texts.append(child.text)
            rede_text = " ".join(texts)
            yield top, rede_fraktion, redner_nachname, redner_vorname, rede_text
        except:
            pass

def parse_top(top):
    text = top.xpath("p[@klasse = 'T_fett']/text()")
    if len(text) == 0:
        text = top.xpath("p[@klasse = 'T_NaS']/text()")
    name = " ".join(text)

    reden = top.xpath("./rede")
    return parse_rede(reden, name)

parser = etree.XMLParser()
def parse_sitzung(file, output_file):

    #:wprint(f"Now working on Sitzung ... {file}")
    tree = etree.parse(str(file), parser)
    tops = tree.xpath("//tagesordnungspunkt")
    res = []
    for top in tops:
        res += parse_top(top)
    return res


with open("muchtext.txt", "w") as output_file:
    rows = []
    for sitzung in tqdm(list(Path(".").rglob("*.xml"))):
        rows += parse_sitzung(sitzung, output_file)
    df = pd.DataFrame(rows, columns=["top", "fraktion", "nachname", "vorname", "text"])
    df.to_csv("rede_fraktion.csv", index=False)

