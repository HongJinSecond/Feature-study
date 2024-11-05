import csv # csv module for reading in comma-seperated files

class Category():
	""" 
	represents a category used to categorize commits.
	"""

	associatedWords = [] # all words associated w/ this category
	category_name = None # name of category

	def __init__(self, fileLocation, name):
		""" 
		constructor
		reads in all associated words w/ this category from specified 
		file location
		"""
		self.category_name = name
		self.associatedWords = [] # reset the instance so that class name is visible to self reference
		self.readInAssociatedWords(fileLocation)

	def readInAssociatedWords(self, fileLocation):
		""" 
		reads in all associated words w/ this category
		"""
		with open(fileLocation, 'rt',encoding="utf-8") as csvfile:
			wordreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			for row in wordreader:
				for word in row:
					self.associatedWords.append(word)
		#print(self.associatedWords)

	def belongs(self, commit_msg):
		"""
		checks if a commit belongs to this category by analyzing
		its commit message.
		@return boolean
		"""

		commit_msg = commit_msg.lower()  #.split(" ") # to array <---le no real need to separate the commit message in words. In any case, assoc_words in word would result in any of the keywords associated to a give type of commit being within a word of the commit, rather than matching this word exactly. And, in Chinese, there are frequently no spaces between words.

		# need to go beyond list contains i.e. fixed = fix
		#for word in commit_msg: <---le
		for assoc_word in self.associatedWords:
			assoc_word = assoc_word.split("...")  # <---le to enable keywords to have other words between them, e.g., 修改... 问题
			found_keyword = True
			for part_keyword in assoc_word: 
			#if assoc_word in word:
				if found_keyword and part_keyword in commit_msg:
					found_keyword = True
				else:
					found_keyword = False
			if found_keyword:
				return True

		# No associated words found!
		return False

	def getName(self):
		""" 
		returns the name of the category
		"""
		return self.category_name



