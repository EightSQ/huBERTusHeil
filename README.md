# huBERTusHeil - Project for HPI-NLP ST21

<img src="https://i.imgur.com/jYL3fX3.jpg" width="200" height="330"><img src="https://upload.wikimedia.org/wikipedia/commons/d/de/2020-07-02_Bundesminister_Hubertus_Heil_by_OlafKosinsky_MG_1922.jpg" width="330" height="330">

> [Hubertus Heil](https://en.wikipedia.org/wiki/Hubertus_Heil) is a german politician and currently serving as Federal Minister of Labour and Social Affairs.

**Leon Schiller**, *Hasso-Plattner-Institute, University of Potsdam*
**Georg Ortwin Otto Kißig**, *Hasso-Plattner-Institute, University of Potsdam*


Please find this document (which is more readable in the browser) and our source code [here](https://github.com/EightSQ/huBERTusHeil).

## Project Implementation

### a) Goals

We want to classify to which political party a speech from the german parliament (Bundestag) matches best. Based on all speeches from the Bundestag from the past election period, we trained a classifier that takes a raw speech as its input and predicts the corresponding party. For this we want to use a state-of-the-art deep neural network.

The objective of this classifier is to transform its input into a representation in a latent feature space which encodes the corresponding party. Speeches from the same or a similar party should then result in similar representations. We want to analyze and visiualize the learned representations of all speeches. Ideally, this would allow us to assess which parties have similar political positions or give similar speeches. We imagine that speeches of similar parties form overlapping clusters in the latent space whereas speeches from distinct parties from disjoint clusters.

When restraining inputs of the model to certain topics, such as e.g. immigration, environmental protection, tax policies, ..., we might even find out which parties have similar opinions on these topics by again analyzing the representations of these speeches and their overlaps in the latent space. 

Thus, our goals are:
- Train a precise classifier mapping a text to a party that would agree with this position
- Find out how well the learned representations of this classifier represent similarities between the political parties 

### b) Data Preparation

The federal german pariliament (Bundestag) [publishes](https://www.bundestag.de/services/opendata) protocols of its sessions as part of an open data initiative. Beginning with the current term, these are also offered in a [machine-readable XML format](https://www.bundestag.de/resource/blob/577234/f9159cee3e045cbc37dcd6de6322fcdd/dbtplenarprotokoll_kommentiert-data.pdf).

An excerpt from such an XML file:
```xml=
<tagesordnungspunkt top-id="Tagesordnungspunkt 17">
  <p klasse="T_NaS">a) Beratung der Unterrichtung durch die Bundesregierung</p>
  <p klasse="T_fett">Die Hightech-Strategie 2025 – Forschung und Innovation für die Menschen</p>
  <p klasse="T_Drs">Drucksache 19/4100</p>
  <rede id="ID197800100">
    <p klasse="redner">
      <redner id="11004323">
        <name>
          <vorname>Anja</vorname>
          <nachname>Karliczek</nachname>
          <rolle><rolle_lang>Bundesministerin für Bildung und Forschung</rolle_lang><rolle_kurz>Bundesministerin BMBF</rolle_kurz></rolle>
        </name>
      </redner>
      Anja Karliczek, Bundesministerin für Bildung und Forschung:
    </p>
    <p klasse="J_1">Sehr geehrter Herr Bundestagspräsident! Liebe Kolleginnen und Kollegen! Meine sehr geehrten Damen und Herren! Jeder zweite von uns erkrankt im Laufe seines Lebens an Krebs. Krebs ist die zweithäufigste Todesursache. Wenn man fragt: „Vor welcher Krankheit fürchten Sie sich am meisten?“, dann antwortet die Mehrheit in Deutschland: Krebs. – Deswegen haben wir in dieser Woche die Nationale Dekade gegen Krebs gestartet. Zehn Jahre lang mobilisieren wir alle Kräfte. Wir wollen Krebs besser verstehen, wir wollen Krebs verhindern, wir wollen Krebs heilen. Die Nationale Dekade gegen Krebs ist ein zentrales Thema der Hightech-Strategie. An ihr möchte ich heute zeigen, wie wir mit der Hightech-Strategie Probleme lösen und was wir meinen, wenn wir sagen: Die Menschen stehen im Mittelpunkt unserer Innovationspolitik.</p>
    <p klasse="J">Drei Beispiele dafür:</p>
    <p klasse="J">Erstens ein Transferthema. Neue Therapien müssen schneller raus aus dem Labor, ran ans Krankenbett kommen. Je näher Forschung und Patienten beieinander sind, desto schneller gelingt uns das, so wie im Nationalen Centrum für Tumorerkrankungen in Heidelberg und in Dresden. Im Rahmen der Nationalen Dekade gegen Krebs bauen wir weitere solcher Standorte auf. Die Ängste und Sorgen der Patienten sind wichtig. Nicht nur in der Therapie, auch in der Forschung werden sie in Zukunft eine wichtige Rolle spielen. Jeder Krankheitsverlauf ist individuell. Persönlich zugeschnittene Therapien helfen individuell. Das ist eine Chance, wie wir sie noch nie hatten, und diese Chance wollen wir nutzen.</p>
    <!-- Paragraphs of the speech -->
    <kommentar>
      (Beifall bei der CDU/CSU sowie bei Abgeordneten der SPD)
    </kommentar>
    <name>Präsident Dr. Wolfgang Schäuble:</name>
    <p klasse="J_1">Dr. Götz Frömming, AfD, ist der nächste Redner.</p>
    <kommentar>(Beifall bei der AfD)</kommentar>
  </rede>
  <!-- Following "rede" (speech) documents -->
 </tagesordnungspunkt>
```

To be able to build a model leveraging the available data source, we first needed to parse a few hundred of these XML files.

The XML documents contain a lot of information for a session, like lists of attending members, formal ID's ("Drucksachen") of the matters discussed, of course speeches and also comments ("kommentar" elements) that mark where applause was given and by whom.

For the purpose of our project, we restricted ourselves to each speech's content, the speaker, his party affiliation and the corresponding "discussion item" (Tagesordnungspunkt).

We did this in the python script `parse_labelled_reden.py`.

The speeches understandably arrive containing many obvious hints on the speakers' affiliation, e.g., "We as Democrats want to...". In order to force our model to actually look at speech's contens besides very obvious keywords, we first replace party names and corresponding synonyms with an `[UNK]` token as a preprocessing step. The following excerpt from the full preprocessing script `preprocess_speeches.py` illustrates this process.

```python=
# list of many synonyms, abbreviations, common phrases for the parties
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
    ...
]

import re
# Create a compiled regular expression of replacable phrases
e = re.compile(f'({"|".join(to_replace)})')

# Read in CSV containing the parsed speeches
import pandas as pd
df = pd.read_csv('rede_fraktion.csv')

for _, row in df.iterrows():
    row['text'] = e.sub('[UNK]', row['text'])
    
# (further cleansing, removing unicode escape sequences, etc.)

df.to_csv('rede_fraktion_preprocessed.csv', index=False)
```

In order to map speeches to a certain political topic, we use their corresponding "Tagesordnungspunkte" (TOPs) which are a (sometimes pretty complicated) description of the debate topic. We used a pretrained Sentence Encoder to map these descriptions to representation vectors that allowed us to find siutable TOPs and thus speeches given a keyword inputted by the user.

### c) System

We construct our classifier by fine-tuning a BERT-Language Model pretrained on the german Wikipedia. We use the pretrained model included in Huggingface from [here](https://huggingface.co/bert-base-german-cased). This model also contains a tokenizer based on the WordPiece Algorithm.

We embedded the model supplied by Huggingface into our own PyTorch Module `BertClassifier` (to be found in the notebook `huBERTusHeil.ipynb`), which additionally contains a classifier stage consisting of two Linear layers and an ReLU activation and BatchNorm inbetween them. When classifying, the first linear layer takes the last hidden state of the BERT model and translates it into an 25-dimensional (50-dimensional) vector, which is intended to represent features of the speech like topic, opinion, tone and other features. The second linear layer then translates it into an 7-dimensional vector, which is the final output corresponding to the 7 different labels we try to predict (*AfD*, *B90/Die Grünen*, *FDP*, *Die Linke*, *SPD*, *CDU/CSU* and *no affiliation*). The following figure illustrates our model architecture.

![](https://i.imgur.com/vSDKfcT.png)


To train the classifier, split our dataset into a 90% training set and 10% test set. As this is a multi-label classification problem, we used [Cross Entropy Loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) as loss function. As parties in the parliament have varying numbers of seats, they also get varying "speaking time" and therefore the dataset contains varying numbers of speeches per party. To address this, we set the class weights going into the Cross Entropy Loss according to

$$
weight(c) = min\left(10, \frac{max_i |S_i|}{|S_c|}\right),
$$

where $c$ is a label and $S_c$ is the set of samples with the label $c$ in the dataset. The surrounding $min$ sets maximum weight, since the very underrepresented class label **fraktionslos** ("no affiliation") would otherwise get assigned an unthinkably high weight.


For thematically grouping speeches, we used the Multilingual Universal Sentence Encoder (MUSE), which is available in TensorflowHub [here](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3). We fed all our TOPs into this model and obtained thus a vector representation for each of them. When a user inputs a phrase corresponding to a political topic, e.g. "immigration law", we can feed this phrase into MUSE as well and then find the k closest TOP-representations using cosine similarity. All speeches corresponding to these TOPs are then considered semantically similar to the topic inputted by the user.



## Evaluation

### a) Empirical Evaluation

We use a confusion matrix as well as per class precision, recall and f1-score for evaluating the classification performance. These are our results after 3 epochs of training with a latent space dimension of 25. We used a learning rate of $5\cdot 10^{-5}$ and a value of epsilon of $10^{-8}$. 

```python
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
cm = confusion_matrix(y_true, y_pred)
class_labels = ["AfD", "B90", "FDP", "Linke", "SPD", "Union", "fraktionslos"]
ConfusionMatrixDisplay(cm, display_labels=class_labels).plot()
```
![confusion matrix on the test set](https://i.imgur.com/oFNZ3lF.png)

```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
```
```
              precision    recall  f1-score   support

         AfD       0.89      0.76      0.82       311
         B90       0.58      0.59      0.59       270
         FDP       0.59      0.63      0.61       285
       Linke       0.63      0.71      0.67       240
         SPD       0.56      0.68      0.61       394
       Union       0.74      0.62      0.67       533
fraktionslos       0.77      0.62      0.69        16

    accuracy                           0.66      2049
   macro avg       0.68      0.66      0.67      2049
weighted avg       0.67      0.66      0.66      2049
```

We can see that the model works best for AfD speeches. This is probabily because AfD is the most radical party in the parliament which also uses the most controversial language so it is easily distinguishable from the other parties. 

SPD and Union are commonly confused with each other. This is probably because they formed the governing coalition over the past election period and are thus often talking about similar things.



### b) Example Results

#### Analyzing the latent space (hidden layer of the classifier)

In the following figures, each dot represents one speech of the test dataset, its color the true label, i.e. affiliation.

![tSNE latent space of test set](https://i.imgur.com/1fmrL3H.png)

Here, we have broken down the 25-dimensional vectors that the hidden layer of the classifier produces down to 2 dimensions using [tSNE](https://lvdmaaten.github.io/tsne/) with a perplexity parameter of 300.

Content-wise, we can clearly see the horizontal border between governing coalition of SPD and Union (red and black) in the top and the opposition in the bottom. The governing parties are blending into each other, like one would expect for two parties in a coalition. This also illustrates the poor distinguishability of the two that we discussed earlier.

In the opposition part in the bottom, we can see that the AfD departs from the rest of the coalition, which also is in line with its controversial and radical nature we discussed earlier. Nevertheless, parts of the FDP's speeches diffuse into the AfD's territory, which is explainable with the similar policies regarding topics around *economy* both parties share. Very appealing to us is that the "party clusters" are layed out in the same way they are in the parliament. From left to right, *Die Linke*, *B90/Die Grünen*, *FDP* and *AfD* in the opposition as well as from left to right *SPD* and *Union* in the governing coalition.


We created the same plot for the entire dataset, i.e., including both training and test set.

![tSNE latent space on entire dataset](https://i.imgur.com/u8VeMOG.png)

Keep in mind that during the training, the model "has seen" most of the samples 3 times, so naturally the boundaries are much more clear than just on the test set.

This time we can see that both the extremes *Linke* and *AfD* in the parliament can be well classified and both share few similarities to other parties. Like progamatically, the *FDP* overlaps with *B90/Die Grünen*, the *SPD* and the *CDU*. CDU and SPD as governing coalition once again blend into each other.
And finally, the "political left-to-right spectrum" has once again been matched by the algorithm. 

#### Finding TOPs corresponding to a certain topic

We used the MUSE Sentence Encoder (from [here](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3)) to map the TOPs to a vector representation. Then we can find the $k$ nearest neighbors of them given a user input. This is done using cosine similarity. Here are some examples of which TOPs are retrieved for a given user input:

User Input: **Ecological Agriculture**
Raw german results (TOPs) for $k = 10$:
- Gesellschaftlichen Zusammenhalt stärken – Gutes Leben und Arbeiten auf dem Land gewährleisten Smart Farming – Flächendeckende Breitbandversorgung für eine innovative Landwirtschaft in Deutschland
- hier: Einzelplan 10 Bundesministerium für Ernährung und Landwirtschaft
- Einzelplan 10 Bundesministerium für Ernährung und Landwirtschaft
- Luftreinhaltung im Straßenverkehr – Ökonomisch, ökologisch und sozial
- Deutsche Landwirtschaft stärken – Bäuerliche Familienbetriebe in Deutschland nachhaltig schützen und erhalten Deutsche Landwirtschaft stärken – Herkunftskennzeichnung von Lebensmitteln, um Bürgern eine selbstbestimmte und transparente Kaufentscheidung zu ermöglichen Deutsche Landwirtschaft stärken – Versorgung mit frischem Obst und Gemüse gewährleisten Lebensmittelverschwendung in Deutschland nachhaltig reduzieren Landwirtschaft eine Zukunft geben – EU-Agrarpolitik neu ausrichten und ambitioniert umsetzen Grünland- und Klimaschutz verbessern, Ackerstatus bei dauernder Grünlandnutzung erhalten
- Produktivität, Klimaresilienz und Biodiversität steigern – Agroforstwirtschaft fördern Neuanlage von Hecken als Bestandteile von modernen Agroforstsystemen fördern Agroforstsysteme als ein nachhaltiges Anbausystem anerkennen und fördern Agroforstwirtschaft möglich machen Agrofortsysteme umfassend fördern
- Faire Bedingungen für Lebensmittel aus deutscher Landwirtschaft im EU-Wettbewerb Deutsche Landwirtschaft stärken – Bäuerliche Familienbetriebe in Deutschland nachhaltig schützen und erhalten
- Auswirkungen der Afrikanischen Schweinepest auf die Agrar- und Ernährungswirtschaft
- Nachhaltige Entwicklungsziele erreichen – Potenziale aus der Agrarökologie anerkennen und unterstützen
- Landwirtschaft eine Zukunft geben – EU-Agrarpolitik neu ausrichten und ambitioniert umsetzen Faire Bedingungen für Lebensmittel aus deutscher Landwirtschaft im EU-Wettbewerb Teilhabe von Frauen in der Landwirtschaft und den ländlichen Räumen

#### Analyzing similaries between parties with focus on particular topics

We have computed a PCA of our latent space when only considering speeches corresponding to a certain topic. In the best case, this shows us beween which parties we have the highest variance in the feature space, i.e., we should see which parties have the highest *disagreement* regarding this specific topic. Some examples with the corresponding user input below.

User Input: **Migration**

<img src="https://i.imgur.com/fycZQ6H.png" width="400" height="300">

Here we can see that the left and green parties (magenta and green) and also the AfD (blue) separate well from each other and from the other parties. This corresponds well to the fact that the AfD is the most restrictive party w.r.t. migration whereas the left and green parties are the least restrictive.

User Input: **Weapon export**

<img src="https://i.imgur.com/rdyJymO.png" width="400" height="300">

Here, "Die Linke" (magenta) is very well separated from the rest. This is because this party favours a strict ban of weapon exports in their program and often makes remarks about this in the parliament. 

User Input: **Family**

<img src="https://i.imgur.com/tvglxHf.png" width="400" height="300">

Here the government (black and red), the left party (magenta), AfD (blue) as well as a cluster consisting of FDP and the green party (yellow and green) separate well from each other. All the parties within one respective cluster do indeed have similar opinions on the broad topic "Family".

User Input: **Climate Protection**

<img src="https://i.imgur.com/T3Uv0iR.png" width="400" height="300">

We see that the green party and the AfD separate especially well from the Rest. This is backed by the fact that AfD denies climate change and does not favor climate protection whereas the green party has quite the opposite opinion and is very ambitious in this regard. 

<img src="https://i.imgur.com/jaTMaK2.png" width="200" height="200">

However, we can observe the rare case of *Die Linke* being near the governing parties in this case, which attributes to similar views and focus of these parties in that regard.

### c) Error Analysis

In our notebook `huBERTusHeil.ipynb`, we also built an interface for direct classification of an input.

A prime example for an error we have seen oftentimes when experimenting with that is the following.

Query: **"Wir sollten sozial Schwache künftig deutlich stärker besteuern."** ("We should tax the poor much more in the future.")
Answer:

![](https://i.imgur.com/CP0RALy.png)

This is certainly not a statement that the *Linke* would make, quite the opposite. A common theme found in the party's program is higher taxation of wealthy people in order to help the poor. The same holds for the *SPD*, in the distribution on the second place.

This error shows us, that the model not really learned to fully understand the content, but rather keywords and phrases typical for parties. In this particular example, "soziale Schwache" ("socially weak", "poor"), and "künftig deutlich stärker besteuern" ("tax significantly higher in the future"). Both seem to lead to the wrong decision.

Another example shows, that contradictory statements can lead to the exact same decision.

Query: **Wir brauchen ein generelles Tempolimit auf den Autobahnen.** ("We need a general speed limit on highways.")
Answer:

![](https://i.imgur.com/Uw0iCiY.png)

Query: **Wir brauchen kein generelles Tempolimit auf den Autobahnen.** ("We need a general speed limit on highways.")
Answer:

![](https://i.imgur.com/oDndfY9.png)


If we insert another word, however, that does not even change the meaning of the query, we can get another decision.


Query: **Wir brauchen kein generelles Tempolimit auf den deutschen Autobahnen.** ("We don't need a general speed limit on the german highways.")
Answer:

![](https://i.imgur.com/sgAW4hg.png)

Just by adding the adjective "german" to referring to the highways, we sway the model to decide that the input comes from the *AfD*, which arguably is often described as the most nationalistic party in the parliament.

This backs our claim that vocabulary could matter more than the actual meaning of the input for the task at hand, i.e., classifying speeches by party.


When doing this kind of analysis, we need to keep in mind that the samples on which model was trained on were typically 500 tokens long, much longer and certainly containing more content that can enable a more informed decision.


### Explainability with SHAP values

For the last query we have measured SHAP values of tokens using the [`shap` library](https://shap.readthedocs.io/en/latest/index.html).
To do this, we feed the measuring library the probability of a given class (here *AfD*) as observable score. Therefore, we can measure which tokens back the decision for the class and which ones counts against it.

![](https://i.imgur.com/Ll0Fukt.png)

As we would expect, "deutschen" (adjective "german") is by far the most significant token for the decision for *AfD*. Interestingly, the personal pronoun "Wir" ("we") in the beginning of the sentence counts heavily against *AfD*, indicating the fact that they typically use different vocabulary for these kinds of statements.

### Similarities of Parties

Sometimes, parties with similar opinions on a certain topic are not necessarily grouped into similar clusters in latent space. Our model just has the objective of separating political parties from each other. Although it  works quite well in separating parties with very *different* views on a specific topic, it does not recognize similarities very well. 

Sometimes our search for matching TOPs given a certain topic is also not working too well. It happens that unrelated TOPs or simply TOPs completely devoid of content pop up in certain searches. For example, we encountered the TOPs "Organ Donation" or simply "Letter a, b and c" when searching for "Speed Limit". We observed that this is especiallu the case for very short TOPs and therefore removed TOPs that consisted of 6 words or less. This did improve our search results.

## Discussion

### a) Discussion of Results

#### Classification Results

Our model was able to discriminate parties from each other quite well. Especially the "extremes" could be adequately distinguished from the rest, which we attribute to the fact that they use more memorable and exclusive vocabulary, which clearly sets them apart from other parties.

We could have further improved classification accuracy by, e.g., choosing a higher dimensionality of the latent space because that would allow the model to encode more information into the latent space. However, we did not do this since it would have made it more difficult to recognize relationships between the political parties. We have actually trained a model with a latent space of 50 dimensions, which did indeed not capture these releationships very well but had a higher classification accuracy. 

#### Latent Space Analysis

We could observe that the model does distinguish speeches from the government very well from those of the opposition. We did not look more deeply into this in particular, however, it is likely that goverment parties are commonly talking about what they have done (using phrases like, e.g. "we introduced this new law...") than opposition parties do, which makes it easier to tell these two party "types" apart.

Distinct parties do indeed form clusters in latent space that do in some sense represent the relationships between them. More extreme parties correspond to clusters far from the other parties whereas the moderate parties correspond slightly overlapping clusters.

When analyzing only speeches that concern a certain topic, we can see that the latent space represents which parties have the largest disagreements on this specific topic.

However, similarities between parties are not captured very well and the overall structure of the latent space remains largely the same for all topics. That is, governing parties are always percieved as very similar and the opposition and extreme parties are usually separated from the government. 

The model does not really follow the objective of mapping similar parties to similar representations in the latent space. It is rather trained with the objective of *distinguishing* parties instead of representing their relationships. Although we could observe that some relationships between parties are encoded in the latent space, there are certainly better ways of forcing the model to recognize them (See next section for some ideas on that). 

### b) Future Prospects

In the future, looking into the extraction of the "applause" features and using them as different kind of label for training like we mentioned earlier could be an effective way of improving the model's capture of party similarities. We would then train a multiclass-classifier on our data that aims to predict all parties that applaud a given speech. That way, it is explicitly forced that the model learns content-related relationships between parties and encodes them into the latent space.

Furthermore, we think that given enough time, one could find a better way to "cluster" TOPs surrounding a particular topic for the purpose of analyzing party similarities in that particular context. This could be achieved by manually assigning keywords to clusters pre-built by leveraging an automatic approach like the sentence embeddings that we already used.

We did try to form clusters in the latent space formed by the sentence embeddings of all TOPs. We did so by using the Agglomerative Clustering algorihm with cosine similarity and average linkage from sklearn (see e.g. [here](https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering_metrics.html)). However, these custers were not always coherent with a specific political topic. We observed that finding the neares neighbors given a user input worked better for our puroposes.

When having assigned thematical labels to speeches, we could also profit from letting the model classify the topics a speech corresponds to. Then, the embeddings in the latent space would contain not only features describing the party affiliation of the speech but also features describing what the speech is about.

Although the data source is great, we could only look at the last four years of parliament sessions since the Bundestag began just in 2017 to publish the protocols in the XML format. Certainly, it would be very interesting to see the same analysis for example for a legislature period in which different coalition forms the government. We would certainly like to see a similar "blending" of the governing parties and general distinguishability from the opposition, once again, to confirm our explanation of this observation. Looking at current voter surveys and given that the current governing coalition may not reach a quorum in September, it seems likely that we can repeat this analysis in a few years with a different governing coalition.
