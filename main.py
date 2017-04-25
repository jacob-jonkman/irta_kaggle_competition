from stemming.porter2 import stem #for stemming of words
import numpy as np #always useful. currently not in use
import threading as thr #more hogh-level multithreading library
from queue import Queue #to limit the amount of threads
import csv #to read in csv files
from nltk.corpus import stopwords #stopwords to filter out
from difflib import SequenceMatcher as seqmatch #similarity measure between strings
from PyDictionary import PyDictionary #to find synonyms
import re #yay regexes. I mean: to filter out nun-alphabetical characters

class Classifier:
   """
   Something about what this program is about.
   """
   def __init__(self, datafile, amountOfThreads):
      # The file the data is read out of.
      self.datafile = datafile

      # To enable usage in threads, the stopwords are cached here.
      # Otherwise, the nltk library would try to load multiple times at
      # the same time, which gives errors.
      self.wordsFilter = stopwords.words('english')
      
      # A queue to hold all the data and distribute it to the threads.
      self.queue = Queue()
      
      # A list of all present threads.
      self.threads = []
      
      # A lock to prevent multiple threads writing to the same variable at
      # the same time.
      # Not actually used now.
      self.lock = thr.Lock()
      
      # The amount of threads which should be used.
      # Is required on multiple occasions, and therefore made a class variable.
      self.aOT = amountOfThreads
      
      # Amount of, respectively, true and false positives, and true and false 
      # negatives. All initialised to 0.
      self.tp = self.fp = self.tn = self.fn = 0

      # Dictionary class, used for synonym matching #
      self.dictionary=PyDictionary()
   

   def startWorkers(self):
      """
      Start as many threads as ordered (provided by variable self.aOT).
      Also stores all threads in self.threads, so they can be stopped later as well.
      """
      for i in range(self.aOT):
         t = thr.Thread(target=self.threadWorker)
         t.start()
         self.threads.append(t)

   def threadWorker(self):
      """
      Basically the process of a thread. Constantly tries to work on a job,
      and ends when none are available anymore.
      The queue holds all data, and distributes it over the workers, which
      request the data to work on.
      """
      while True:
         row = self.queue.get() #get a row of data
         if row is None: #ending criterium
            break
         self.similarityQuestions(row) #the actual working function
         self.queue.task_done() #inform the queue one task is done

   def stemQuestion(self, question):
      """
      Stem all the words in a given question.
      Also filter all the words on a given filter: Filtered words are removed.
      """
      stemmedquestion = []
      for word in re.split("\W+", str(question)):
         w = re.sub('[^a-z]', '', word.lower())
         #print("w = {0}".format(w))
         if w and (w not in self.wordsFilter):
            stemmedquestion.append(stem(w))
      return stemmedquestion

   def computeSimilarity(self, q1, q2):
      """

      """
      matches = 0
      total = 0
      for word in q1:
         if word in q2:
            matches += 1
         else:
            try:
               synonyms = self.dictionary.synonym(word)
            except:
               pass
            if synonyms:               
               for i in range(len(synonyms)):
                  if (synonyms[i] not in q1) and (synonyms[i] in q2):
                     matches += 1
                     break
         total += 1
      if matches == 0:
         return 0
      else:
         return total/matches

   def similarityQuestions(self, row):
      """
      Function to determine the similarity between two given questions.
      For now uses the SequenceMatcher from difflib to do so.
      Might want to use a custom method in the future, as this gives a low precision.
      """
      q1 = self.stemQuestion(row[3])
      q2 = self.stemQuestion(row[4])
      
      # Compute similarity of the two questions#
      #sim = seqmatch(None, q1, q2).ratio()
      sim = self.computeSimilarity(q1, q2)
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

   def run(self):
      # Function which starts the threads.
      self.startWorkers()
      
      with open(self.datafile, "r") as f:
         # These two lines are for finding out if there is a header.
         # Can be left commented if the header being present is given.
         #hasHeader = csv.Sniffer().has_header(f.read(1024))
         #f.seek(0) #restart reading
         reader = csv.reader(f)
         #if hasHeader: #only used when 'sniffing' for a header
         next(reader) #skip header row
         # Read out all the rows and put them in the queue.
         for row in reader:
            try:
               self.queue.put(row)
            except:
               print("Cannot put {0} in the queue!".format(row))
         self.queue.join() #block until all tasks are done
         #then stop the workers
         for i in range(self.aOT):
            self.queue.put(None)
         for t in self.threads:
            t.join()
      return (self.tp, self.fp, self.tn, self.fn)

if __name__ == "__main__":
   datafile = "data/train.csv"
   amountOfThreads = 16
   classifier = Classifier(datafile, amountOfThreads)
   tp, fp, tn, fn = classifier.run()
   print("""
   Precision: {0}
   Recall: {1}
   """.format((tp / (tp + tn)), (tp / (tp + fp))))
