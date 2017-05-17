from stemming.porter2 import stem #for stemming of words
import numpy as np #always useful. currently not in use
import threading as thr #more hogh-level multithreading library
from queue import Queue #to limit the amount of threads
import csv #to read in csv files
from nltk.corpus import stopwords #stopwords to filter out
from difflib import SequenceMatcher as seqmatch #similarity measure between strings
from PyDictionary import PyDictionary #to find synonyms
import re #yay regexes. I mean: to filter out nun-alphabetical characters
import sys, os #for parsing command-line arguments
from getopt import getopt, GetoptError #also to parse command-line arguments
import subprocess #to call tail

class Classifier:
   """
   Something about what this program is about.
   """
   def __init__(self, datafile, amountOfThreads, 
                trainingSet, similarityTreshold, outputfile):
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

      # Dictionary class, used for synonym matching.
      self.dictionary=PyDictionary()
      
      # Boolean which depicts whether or not we are using a training set.
      self.trainingSet = trainingSet
      
      # File the testing predictions should be written to.
      self.outputfile = outputfile
      
      # Float to depict the amount of similarity two questions need to have to be considered equal.
      self.similarityTreshold = similarityTreshold
      
      # To keep track of where the progressbar is, so it won't go back if a slower thread finishes an iteration of lower number.
      # Only used there, but needs to be global.
      self.lastIteration = -1
      
      # The amount of IDs in the datafile.
      # Used for preallocation purposes and the progress bar.
      self.amountOfIDs = self._getAmountOfIDs()
      
      # Length of the progress bar.
      # Not really an argument for the user, change here when needed.
      self.barLength = 50
      
      if(not(trainingSet)):
         # Dictionary to contain the predictions per question pair.
         # Preallocated for efficiency.
         self.testDictionary = dict.fromkeys([str(i) for i in range(self.amountOfIDs + 1)])

   def _getAmountOfIDs(self):
      """
      Call the external 'tail' command to read out the last line of the datafile.
      Then return the id which is in the last line, so we know how many question
      pairs there are in total.
      """
      line = str(subprocess.check_output(['tail', '-1', self.datafile]))
      return int(line.split(',')[0][3:-1])

   def startWorkers(self):
      """
      Start as many threads as ordered (provided by variable self.aOT).
      Also stores all threads in self.threads, so they can be stopped later 
      as well.
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
         try:
            self.similarityQuestions(row) #the actual working function
         except:
            print("Error processing {0}".format(row))
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
      Computes the similarity between two given questions based on the
      inverse jaccard similarity (union divided by intersection).
      No, that's not (yet) a thing.
      Yes, it works.
      """

      sq1 = set(q1)
      sq2 = set(q2)

      ulen = len(set.union(sq1, sq2))
      ilen = len(set.intersection(sq1, sq2))

      if (ilen == 0):
         return 0

      return ulen / ilen

      '''matches = 0
      for word in q1:

         # The words actually match.
         if word in q2:
            matches += 1

         # Try synonyms.
         else:
            try:
               # Try to find a synonym that does not occur in q1 but is in q2.
               synonyms = set(self.dictionary.synonym(word)) - set(q1)
               if synonyms:
                  if set.intersection(synonyms, set(q2)):
                     matches += 1
            except:
               pass

      if matches == 0:
         return 0
      else:
         return len(q1) / matches'''
         
   def updateTrainingResults(self, sim, row):
      """
      Updates the true and false positives and negatives based on the result
      achieved and the result provided by the training set.
      ...therefore only works on training sets.
      """
      if sim > self.similarityTreshold: #we guess they are duplicate questions
         if row[5] == "1": #true positive
            self.tp += 1
         else: #false positive
            self.fp += 1
      else: #we guess they are different questions
         if row[5] == "0": #true negative
            self.tn += 1
         else: #false negative
            self.fn += 1

   def updateTestingResults(self, sim, pairID):
      """
      Checks whether two questions are the same based on the similarity.
      If so, update the value in the testing dictionary accordingly.
      """
      if sim > self.similarityTreshold:
         self.testDictionary[str(pairID)] = 1
      else:
         self.testDictionary[str(pairID)] = 0
   
   def similarityQuestions(self, row):
      """
      Function to determine the similarity between two given questions.
      For now uses the SequenceMatcher from difflib to do so.
      Might want to use a custom method in the future, as this gives a low
      precision.
      """
      indexone = 1
      indextwo = 2
      #indices of questions are different in test/trainingset
      if(self.trainingSet): 
         indexone = 3
         indextwo = 4
      q1 = self.stemQuestion(row[indexone])
      q2 = self.stemQuestion(row[indextwo])

      # Compute similarity of the two questions#
      #sim = seqmatch(None, q1, q2).ratio()
      sim = self.computeSimilarity(q1, q2)
      with self.lock:
         currentIteration = int(row[0])
         # Only print the bar if it will fill more.
         if(currentIteration > self.lastIteration):
            self.lastIteration = currentIteration
            self._printProgressBar(int(row[0]), self.amountOfIDs, prefix = "Progress: ")
         if(self.trainingSet):
            self.updateTrainingResults(sim, row)
         else:
            self.updateTestingResults(sim, row[0])
         
   def _printProgressBar(self, iteration, total, prefix='', suffix = '', 
                         decimals = 1):
      """
      Creates a progress bar. Call it after each computation.
      @params:
         iteration   - Required  : current iteration (int)
         total       - Required  : total amount of iterations (int)
         prefix        Optional  : prefix string (str)
         suffix        Optional  : suffix string (str)
         decimals      Optional  : positive number of decimals in percentage (int)
      """
      strFormat = "{0:." + str(decimals) + "f}"
      percents = strFormat.format(100 * (iteration / float(total)))
      filledLength = int(round(self.barLength * iteration / float(total)))
      bar = '█' * filledLength + '-' * (self.barLength - filledLength)

      sys.stdout.write("\r{0} |{1}| {2}{3} {4}".format(prefix, bar, percents, '%', suffix)),

      if iteration == total:
         sys.stdout.write(' Complete \n')
      sys.stdout.flush()
         
   def _writeOutputFile(self):
      """
      Write the testDictionary, containing the testing predictions, to a csv file.
      """
      #print(self.testDictionary)
      with open(self.outputfile, 'w') as of:
         for key, value in self.testDictionary.items():
            if(value == None): #why does this happen?
               value = 0
            of.write("{0},{1}\n".format(key, value))

   def run(self):
      """
      Do like everything this program has to offer.
      """
      # Function which starts the threads.
      self.startWorkers()
      #Show the progress bar
      self._printProgressBar(0, self.amountOfIDs, prefix = "Progress: ")
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
      if self.trainingSet:
         return (self.tp, self.fp, self.tn, self.fn)
      else:
         self._writeOutputFile()
         return (0, 0, 0, 0)
      
def main(argv):
   """
   First calls the command-line argument parser.
   Then calls the class and executes the program.
   """
   if not ((3,0,0) <= sys.version_info[:3]):
	   raise RuntimeError("At least Python version 3.0 required!")
	
   datafile, amountOfThreads, trainingSet, similarityTreshold, outputfile = getArguments(argv)
   classifier = Classifier(datafile, amountOfThreads, trainingSet, similarityTreshold, outputfile)
   tp, fp, tn, fn = classifier.run()
   if tp == fp == tn == fn == 0:
      print("See file {0} for the predictions!".format(outputfile))
      return 0
   print("""
   Precision: {0}
   Recall: {1}
   """.format((tp / (tp + tn)), (tp / (tp + fp))))
   
def usage():
   print("""
Usage: python main.py [options]
Options:
   -d <file>,  --datafile=<file>    The file which contains the questions
                                    Default is 'data/train.csv'.
   -t <int>,   --threads=<int>      The amount of threads to be used when 
                                    running the program. Default is 8.
   -s <float>, --similarity=<float> The treshold for which amount of similarity
                                    will be needed to have two questions be
                                    estimated as being the same question.
                                    Default is 0.6.
   -o <file>,  --outputfile=<file>  The file the testing prediction should be
                                    put out to, if existent. Else this
                                    argument has no effect.
                                    Default is 'data/output.csv'.
   -e,         --test               Use if the datafile contains a test set, or
                                    should be considered containing one.
                                    If not given, the program will assume the
                                    datafile contains a training set.
   -h,         --help               Display this help text and exit.
   
   Note that this program requires a python version of at least 3.0.0 to be
   ran with.
   """)

def getArguments(argv):
   """
   Parses the arguments given certain options.
   Currently supports reading in a specific datafile, 
   specifying the amount of threads used,
   giving the similarity equal questions should at least have,
   defining the file an eventual test prediction should be put out to,
   and whether the datafile contains a training or a test set.
   All other options call the usage function, which explains
   the user how to correctly call the program.
   Mind that the values given to the variables in this function
   are the default values as also written in the usage() function.
   """
   try:
      opts, args = getopt(argv, 'd:t:s:o:eh', ['datafile=', 'threads=', 'similarity=', 'outputfile=', 'test', 'help'])
   except GetoptError:
      usage()
      sys.exit(2)
   datafile = "data/train.csv"
   outputfile = "data/output.csv"
   amountOfThreads = 8
   trainingSet = True
   similarity = 0.6
   for opt, arg in opts:
      if opt in ('-d', '--datafile'):
         if os.path.isfile(arg): #if the file exists
            datafile = arg
      elif opt in ('-t', '--threads'):
         if int(arg) > 1: #we need to have at least 1 thread
            amountOfThreads = int(arg)
      elif opt in ('-s', '--similarity'):
         if 0.0 < float(arg) <= 1.0: #similarity should be between 0 and 1
            similarity = float(arg)
         else:
            print("Error: Similarity should be in between 0.0 and 1.0!\nUsing default value {0} instead.".format(similarity))
      elif opt in ('-o', '--outputfile'):
         outputfile = arg
      elif opt in ('-e', '--test'):
         trainingSet = False
      else: #includes --help
         usage()
         sys.exit(2)
   
   return datafile, amountOfThreads, trainingSet, similarity, outputfile
   
if __name__ == "__main__":
   main(sys.argv[1:])
