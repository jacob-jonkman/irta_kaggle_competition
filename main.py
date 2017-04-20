from stemming.porter2 import stem
import numpy as np
import _thread as thr
import csv
from nltk.corpus import stopwords

class Hoi:

   def __init__(self, datafile):
      self.truesims = []
      self.falsesims = []
      self.datafile = datafile

   def stemquestion(self, question):
     stemmedquestion = []
     for word in question.split():
       if word not in stopwords.words('english'):
         stemmedquestion.append(stem(word.lower()))
       
     return stemmedquestion
     
   def similarityQuestions(self, row):
      q1 = self.stemquestion(row[3])
      q2 = self.stemquestion(row[4])
      print("row: {0}, q1: {1}, q2: {2}".format(row, q1, q2))
      #if row[5] == "1":
      #  print(q1, q2, "\n\n")
      
      # Bekijk de similarity tussen de twee questions, zie wat een goede cut-off zou zijn #
      sim = 0
      for word in q1:
        if word in q2:
          sim += 1
      if row[5] == 1:
        self.truesims.append(sim/len(q1))
      else:
        self.falsesims.append(sim/len(q1))


   def run(self):
     with open(self.datafile, "r") as f:
       
       reader = csv.reader(f)
       i = 0
       for row in reader:
         try:
            thr.start_new_thread(self.similarityQuestions, (row,))                  
            #self.similarityQuestions(row)
         except:
            print("Error: THE END IS COMING")
      
if __name__ == "__main__":
   datafile = "data/train.csv"
   hoi = Hoi(datafile)
   print(hoi.truesims)
   print(hoi.falsesims)
