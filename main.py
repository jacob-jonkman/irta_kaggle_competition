from stemming.porter2 import stem
import numpy as np
import csv
from nltk.corpus import stopwords

def stemquestion(question):
  stemmedquestion = []
  for word in question.split():
    if word not in stopwords.words('english'):
      stemmedquestion.append(stem(word.lower()))
    
  return stemmedquestion  

def main():
  with open("data/train.csv", "r") as f:
    truesims = []
    falsesims = []
    
    reader = csv.reader(f)
    i = 0
    for row in reader:
      q1 = stemquestion(row[3])
      q2 = stemquestion(row[4])
      if row[5] == "1":
        print(q1, q2, "\n\n")
      
      # Bekijk de similarity tussen de twee questions, zie wat een goede cut-off zou zijn #
      sim = 0
      for word in q1:
        if word in q2:
          sim += 1
      if row[5] == 1:
        truesims.append(sim/len(q1))
      else:
        falsesims.append(sim/len(q1))
                   
      
if __name__ == "__main__":
  main()
