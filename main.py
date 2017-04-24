from stemming.porter2 import stem #for stemming of words
import numpy as np
#import _thread as thr #to enable multithreading
import threading as thr #more hogh-level multithreading library
from queue import Queue #to limit the amount of threads
import csv #to read in csv files
from nltk.corpus import stopwords #stopwords to filter out
from difflib import SequenceMatcher as seqmatch #similarity measure between strings

class Hoi:

   def __init__(self, datafile, amountOfThreads):
      #self.truesims = []
      #self.falsesims = []
      self.datafile = datafile
      self.wordsFilter = stopwords.words('english') #to enable usage in threads
      self.queue = Queue()
      self.threads = []
      #self.lock = thr.allocate_lock() #with the _thread import
      self.lock = thr.Lock() #with the threading import
      self.aOT = amountOfThreads
      self.tp = self.fp = self.tn = self.fn = 0
      
      self.startWorkers()
      
   def startWorkers(self):
      for i in range(self.aOT):
         t = thr.Thread(target=self.threadWorker)
         t.start()
         self.threads.append(t)
         
   def threadWorker(self):
      while True:
         row = self.queue.get()
         if row is None:
            break
         self.similarityQuestions(row)
         self.queue.task_done()

   def stemquestion(self, question):
     stemmedquestion = []
     for word in str(question).split():
       if word not in self.wordsFilter:
         stemmedquestion.append(stem(word.lower()))
     return stemmedquestion
     
   def similarityQuestions(self, row):
      q1 = self.stemquestion(row[3])
      q2 = self.stemquestion(row[4])
      #print("row: {0}, q1: {1}, q2: {2}".format(row, q1, q2))
      #if row[5] == "1":
      #  print(q1, q2, "\n\n")
      
      # Bekijk de similarity tussen de twee questions, zie wat een goede cut-off zou zijn #
      sim = seqmatch(None, q1, q2).ratio()
      if sim > 0.6: #we guess they are duplicate questions
         if row[5] == "1": #true positive
            self.tp += 1
         else: #false positive
            self.fp += 1
      else: #we guess they are different questions
         if row[5] == "0": #true negative
            self.tn += 1
         else: #false negative
            self.fn += 1
#      with self.lock:
#         if row[5] == "1":
#           self.truesims.append(sim)
#         else:
#           self.falsesims.append(sim)


   def run(self):
     with open(self.datafile, "r") as f:
       #hasHeader = csv.Sniffer().has_header(f.read(1024))
       #f.seek(0) #restart reading
       reader = csv.reader(f)
       #if hasHeader:
       next(reader) #skip header row
       for row in reader:
         try:
            #thr.start_new_thread(self.similarityQuestions, (row,)) 
            self.queue.put(row)                 
            #self.similarityQuestions(row)
         except: #disadvantage: we need a lot of ctrl-c to stop it (or a kill command)
            pass
            #print("Error: THE END IS COMING!\nRow: {0}".format(row))
       self.queue.join() #block until all tasks are done
       #then stop the workers
       for i in range(self.aOT):
         self.queue.put(None)
       for t in self.threads:
         t.join()
      
if __name__ == "__main__":
   datafile = "data/train.csv"
   amountOfThreads = 8
   hoi = Hoi(datafile, amountOfThreads)
   hoi.run()
   print("""
   Precision: {0}
   Recall: {1}
   """.format((hoi.tp / (hoi.tp + hoi.tn)), (hoi.tp / (hoi.tp + hoi.fp))))
