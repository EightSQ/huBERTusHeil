from pathlib import Path
from lxml import etree
from tqdm import tqdm

def parse_rede(rede):
    texts = []
    for child in rede.xpath("p[@klasse = 'J_1' or @klasse = 'J' or @klasse = 'O']"):
        if child.text:
            texts.append(child.text)
    return " ".join(texts) + "\n"


parser = etree.XMLParser()
def parse_sitzung(file, output_file):
    #:wprint(f"Now working on Sitzung ... {file}")
    tree = etree.parse(str(file), parser)
    reden = tree.xpath("//rede")
    for rede in reden:
        output_file.write(parse_rede(rede))


with open("muchtext.txt", "w") as output_file:
    for sitzung in tqdm(list(Path(".").rglob("*.xml"))):
        parse_sitzung(sitzung, output_file)

