import os
import json
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

docs =[]
dirs = ["biorxiv_medrxiv/pdf_json"]

for d in dirs:
    for file in os.listdir(f"{d}/"):
        file_path = f"{d}/{file}"
        j = json.load(open(file_path, "rb"))
        # print(j)
        title = j['metadata']['title']

        try:
            abstract = j['abstract'][0]
        except:
            abstract = ""
        full_text = ""
        for text in j['body_text']:
            # print(text['text'])
            full_text += text['text'] +'\n\n'

        # print(full_text)
        docs.append([title, abstract, full_text])
        # break
df = pd.DataFrame(docs, columns=['title', 'abstract', 'full_text'])
# print(df)

incubation = df[df['full_text'].str.contains('incubation')]
# print(incubation.head())

texts = incubation['full_text'].values
print(len(texts))

nums = []
incubation_times = []
for text in texts:
    for sentence in text.split(". "):
        if "incubation" in sentence:
            # print(sentence)
            single_day_flt = re.findall(r"( \d{1,2}(\.\d{1,2})? day[s]?)", sentence)
            print(single_day_flt)

            if len(single_day_flt) == 1:
                # print(single_day_flt[0])
                # print(single_day_flt[0][0])
            # single_day = re.findall(r"( \d{1,2} day)", sentence)
                n = single_day_flt[0][0].split(" ")
                print(n)

                incubation_times.append(float(n[1]))


print(len(incubation_times))
print(incubation_times)
avg = np.mean(incubation_times)
print('The mean projected incubation period is: '+ str(round(avg,2)) + ' days')

#plotting
plt.hist(incubation_times, bins=23)
plt.ylabel("bin counts")
plt.xlabel("incubation time (days)")
plt.show()
