# TODO: add a static variable so that this function know if it's already been called.
load_libraries = function() {
  # Hide all the extra messages displayed when loading these libraries.
  suppressMessages({
    library(R.matlab)  # To load the Matlab data.
    library(ggplot2)   # For charts
    library(reshape2)  # For heatmaps
    library(e1071)     # For svm
    library(foreach)   # For multicore
    library(doMC)       # For multicore (not used at the moment)
    library(doParallel) # For multicore
    library(RhpcBLASctl) # Accurate physical core detection
    library(dplyr)     # For group_by
    library(tm)        # For text mining
    library(SnowballC) # For stemming
    library(NLP)       # For ngrams and sentence tokenization.
    library(openNLP)   # For sentence tokenization, part-of-speech tagging and named-entity recognition.
    library(stringi)   # For sentence feature engineering.
    library(stringr)   # For sentence feature engineering.
    library(caret)     # For stratified cross-validation folds.
    library(eqs2lavaan) #  For covariance heatmap.
    #library(qdap)      # Not sure if we actually need this one.
  })
}

# Raw data is a Reddit dataframe with metadata and the comment as the $body column
featurize = function(raw_data, downsample_pct = 1,
      sparsity_ngrams = c(0.99, 0.99, 0.99), principal_components = 0, two_way = F,
      prune_training_features=T, remove_stopwords = T, stem = T, binarize = F,
      include_bigrams = T, include_trigrams = T) {
  cat("Featurizing", prettyNum(nrow(raw_data), big.mark=","), "rows.\n")

  # Downsample the raw data if we want to speed up computation.
  if (downsample_pct != 1) {
    raw_data = raw_data[sample(nrow(raw_data), round(nrow(raw_data) * downsample_pct), replace=F), ]
    cat("Downsampled input data by rate:", downsample_pct, ". Revised row count:", prettyNum(nrow(raw_data), big.mark=","), "\n")
  }

  # Convert raw_data$body to a DocCorpus for further text processing.
  text_corpus = Corpus(DataframeSource(data.frame(raw_data$body)))
  length(text_corpus)

  # TODO: investigate tm::DCorpus for a distributed corpus class.

  # Process an input dataframe with raw reddit data.
  text_features = doc_feature_matrix(text_corpus, load_stopwords(), sparsity_ngrams=sparsity_ngrams,
              remove_stopwords=remove_stopwords, stem=stem, binarize=binarize, include_bigrams=include_bigrams, include_trigrams=include_trigrams)


  # Convert corpuses into feature matrices.
  text_features$dtm = as.matrix(text_features$dtm)
  if (include_bigrams) {
    text_features$bigrams = as.matrix(text_features$bigrams)
  }
  if (include_trigrams) {
    text_features$trigrams = as.matrix(text_features$trigrams)
  }

  # Add prefixes to all feature names to ensure no collisions.
  colnames(raw_data) = paste0("red_", colnames(raw_data))
  colnames(text_features$raw_features) = paste0("raw_", colnames(text_features$raw_features))
  colnames(text_features$dtm) = paste0("gm1_", colnames(text_features$dtm))
  if (include_bigrams) {
    colnames(text_features$bigrams) = paste0("gm2_", colnames(text_features$bigrams))
  }
  if (include_trigrams) {
    colnames(text_features$trigrams) = paste0("gm3_", colnames(text_features$trigrams))
  }

  if (binarize) {
    text_features$dtm = binarize_dtm_matrix(text_features$dtm)
    if (include_bigrams) {
      text_features$bigrams = binarize_dtm_matrix(text_features$bigrams)
    }
    if (include_trigrams) {
      text_features$trigrams = binarize_dtm_matrix(text_features$trigrams)
    }
  }

  # Eliminate grams that are zero in the test data, to reduce the potential for overfitting.
  if (prune_training_features) {
    # Skip this implementation for now.
  }

  # Integrate these features into a dataframe.
  if (include_bigrams && include_trigrams) {
    feature_data = with(text_features, data.frame(dtm, bigrams, trigrams, raw_features))
  } else if (include_bigrams) {
    feature_data = with(text_features, data.frame(dtm, bigrams, raw_features))
  } else if (include_trigrams) {
    feature_data = with(text_features, data.frame(dtm, trigrams, raw_features))
  } else {
    feature_data = with(text_features, data.frame(dtm, raw_features))
  }

  if (principal_components > 0) {

    # Restrict to the first X components.
    cat("Restricting to first", principal_components, "principal components.\n")

    # PCA transform the training data.
    # May want to set scale=F
    pca = prcomp(feature_data)

    feature_data = pca$x[, 1:principal_components]

    # Apply same transformation to the test data.
    # test_data = predict(pca, test_data)[, 1:principal_components]
  }

  if (two_way) {
    cat("Adding two-way interactions.\n")
    int = two_way_interactions(feature_data)
    feature_data = cbind(feature_data, int)
    cat("Total features:", ncol(feature_data), "\n")

    #int = two_way_interactions(test_data)
    #test_data = cbind(test_data, int)
  }


  # Add the new data to the original raw Reddit db columns and return.
  final_data = cbind(raw_data, feature_data)

  results = list(data = final_data)
  return(results)
}

# Convert a corpus of documents into a feature matrix.
# Derive a dictionary of words/ bigrams and total number of their appearances through out the whole dataset.
# If training_data is passed in, use that to restrict our feature set (for the test set).
doc_feature_matrix = function(doc_corpus, stopwords = c(), prior_training_data = NULL,
                              sparsity_ngrams = c(0.98, 0.99, 0.99), squared_raw = F, remove_stopwords = T,
                              stem=F, binarize=F, include_bigrams = T, include_trigrams = T) {
  if (is.null(prior_training_data)) {
    cat("N-gram sparsity:", paste0(sparsity_ngrams, collapse=", "), "\n")
    sparsity_unigram = sparsity_ngrams[1]
    if (include_bigrams) {
      sparsity_bigram = sparsity_ngrams[2]
    }
    if (include_trigrams) {
      sparsity_trigram = sparsity_ngrams[3]
    }
  }

  # Create features from the raw text.
  # This does not need to be restricted to training_data features, because no sparsity constraint
  # is used.
  # TODO: make this faster!
  raw_time = system.time({
    raw_features = create_raw_features(doc_corpus)
  })

  # Create squared terms for raw features.
  # Disabled by default, since it seems to reduce performance.
  if (squared_raw) {
    raw_features_sqr = apply(raw_features, MARGIN=2, FUN=function(x) {
      x^2
    })
    # We need to use colnames() here because raw_features_sqr is a matrix, not a df.
    colnames(raw_features_sqr) = paste0(names(raw_features), "_sqr")
    raw_features = cbind(raw_features, raw_features_sqr)
  }

  if (is.null(prior_training_data)) {
    # Remove raw features that are constant.
    removed_names = c()
    for (name in names(raw_features)) {
      if (length(unique(raw_features[, name])) == 1) {
        removed_names = c(removed_names, name)
      }
    }
    if (length(removed_names) > 0) {
      cat("Removing constant raw features:", paste0(removed_names, collapse=", "), "\n")
      raw_features = raw_features[, !names(raw_features) %in% removed_names]
    }

    cat("Raw features created:", prettyNum(ncol(raw_features), big.mark=","), "\n")
  } else {
    # Restrict to raw features that existed in the training set.
    raw_features = raw_features[, names(prior_training_data$raw_features)]
  }

  # Now use standard processing to handle the documents.
  book = tm_map(doc_corpus, content_transformer(tolower))

  #if (length(stopwords) > 0 & remove_stopwords) {
  #  book  = tm_map(book, removeWords, stopwords)
  #}

  # NOTE: we should double-check this removePunctuation code, because it may remove the punctation
  # without substituting spaces, which will mess up the words.
  book = tm_map(book, removePunctuation)

  # NOTE: are we sure that we want to remove numbers? May want to review.
  book = tm_map(book, removeNumbers)

  book = tm_map(book, stripWhitespace)

  #if (remove_stopwords) {
  #   book = tm_map(book, removeWords, stopwords("english"))
  #}

  if (stem)  {
    book = tm_map(book, stemDocument)
  }

  # Code TBD, but we need to do before we removeSparseTerms

  # If this is null, it means we are currently processing the training data.
  if (is.null(prior_training_data)) {

    # Remove stopwords from the unigrams, but not the bigrams and trigrams.
    dtm = DocumentTermMatrix(book, control=list(stopwords=T))

    # Is 0.99 sufficient? Should we go lower?
    dtm = removeSparseTerms(dtm, sparsity_unigram)
    cat("Unigrams created:", prettyNum(ncol(dtm), big.mark=","), "\n")

    # Create bigrams - should we do these before removing sparse terms and/or stopwords?
    if (include_bigrams) {
      dtm_bigrams = DocumentTermMatrix(book, control = list(tokenize = BigramTokenizer))
      # We have to removeSparseTerm before converting to a matrix because there are too many cells otherwise (> a billion).
      # This is a loose restriction - bigram must be used in at least 1% of documents.
      # TODO: may need to tweak this for prediction/verification data - don't want to remove training bigrams.
      dtm_bigrams = removeSparseTerms(dtm_bigrams, sparse = sparsity_bigram)
      cat("Bigrams created:", prettyNum(ncol(dtm_bigrams), big.mark=","), "\n")
    }

    # Create trigrams
    if (include_trigrams) {
      dtm_trigrams = DocumentTermMatrix(book, control = list(tokenize = TrigramTokenizer))
      dtm_trigrams = removeSparseTerms(dtm_trigrams, sparse = sparsity_trigram)
      cat("Trigrams created:", prettyNum(ncol(dtm_trigrams), big.mark=","), "\n")
    }

  } else {
    # Restrict dictionary to the terms used in the training data.
    dtm = DocumentTermMatrix(book, control = list(dictionary = Terms(prior_training_data$dtm)))
    if (include_bigrams) {
      dtm_bigrams = DocumentTermMatrix(book, control = list(tokenize = BigramTokenizer, dictionary=Terms(prior_training_data$bigrams)))
    }
    if (include_trigrams) {
      dtm_trigrams = DocumentTermMatrix(book, control = list(tokenize = TrigramTokenizer, dictionary=Terms(prior_training_data$trigrams)))
    }
  }

  if (include_bigrams && include_trigrams) {
    results = list(dtm = dtm, bigrams = dtm_bigrams, trigrams = dtm_trigrams, raw_features = raw_features)
  } else if (include_bigrams) {
    results = list(dtm = dtm, bigrams = dtm_bigrams, raw_features = raw_features)
  } else if (include_trigrams) {
   results = list(dtm = dtm, trigrams = dtm_trigrams, raw_features = raw_features)
  } else {
   results = list(dtm = dtm, raw_features = raw_features)
  }
  return(results)
}

binarize_dtm_matrix = function(dtm_matrix) {
  apply(dtm_matrix, MARGIN=2, FUN=function(col) {
    as.numeric(col > 0)
  })
}

# @param punc_feature_pct
#   Set to T to generate both sum and pct features for each punctuation mark.
create_raw_features = function(corpus, punct_feature_pct = F) {
  n = length(corpus)

  # Run this processing outside of the loop to make it faster.
  corpus = tm_map(corpus, content_transformer(tolower))

  # TODO: multicore, maybe via mcmapply? Trying it but it's not tested yet.
  # This loop is incredibly slow, need to also consider more vectorization.
  feature_matrix = sapply(corpus, FUN=function(book) {
    #feature_matrix = parSapply(cl=conf$cluster, corpus, FUN=function(book) {
    # Note: text is an array, where each element is a line in the original text.
    text = tolower(book$content)

    # Convert from an array of lines to a single collapsed string.
    # When analyzing words we may want a space between lines.
    text_blob_with_spaces = paste(text, collapse=" ")
    # But when analyzing characters we don't want to add extra characters.
    text_blob_no_spaces = paste(text, collapse="")

    text_no_punct = sapply(text, FUN=remove_punc)

    # Word-level features
    # TODO: Count and pct of unique words. I.e. words that only that document uses.
    word_features = c(words_with_chars_count=0, words_with_chars_pct=0)

    ########
    # Character-level features.

    # What percentage or count of the text is each of these individual characters.
    # We use a list to allow elements to also be lists of characters (groups).
    characters = list(question_mark="?", semicolon=";", dollar_sign="$", pound_sign="#",
                      exclamation_mark="!", parentheses=c("(", ")"), brackets=c("[", "]"),
                      ampersand="&", hyphen="-", comma=",", forward_slash = "/",
                      backward_slash = "\\", period = ".", at_sign="@", angle_bracket = c("<", ">"),
                      double_quotes = "\"", single_quotes="'", tilde="~", space=" ",
                      plus = "+", equals="=", underscore="_", asterisk="*")
    features_per_char = 1
    if (punct_feature_pct) {
      features_per_char = features_per_char + 1
    }
    character_features = rep(NA, length(characters)*features_per_char)
    char_names = rep(NA, length(character_features))
    # Number of non-whitespace characters.
    text_char_length = max(1, (sum(str_length(text)) - sum(str_count(text, "\\p{WHITE_SPACE}"))))
    for (i in 1:length(characters)) {
      char_str = characters[[i]]
      char_count = sum(stri_count(text_blob_no_spaces, fixed=char_str))
      char_count = ifelse(is.na(char_count), 0, char_count)
      # results = c(char_count, char_count / text_char_length)
      # Just keep the pct features.
      results = c(char_count / text_char_length, char_count)
      char_name = names(characters[i])
      new_names = c(paste0(char_name, "_sum"), paste0(char_name, "_pct"))
      # names(results) = new_names
      if (!punct_feature_pct) {
        # Restrict to just the count variable.
        results = results[1]
        new_names = new_names[1]
      }
      # Save a character vector of length two.
      index_range = ((i-1)*features_per_char+1):(i*features_per_char)
      character_features[index_range] = results
      char_names[index_range] = new_names
    }
    names(character_features) = char_names

    # Analyze linguistic features via openNLP
    linguistics = analyze_linguistics(text)

    pos_unique_tags = length(unique(linguistics$pos_tags))

    # From http://www.surdeanu.info/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html#Word
    pos_tag_options = c('CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB')

    # Calculate the distribution of Parts of Speech in the text.
    pos_dist = rep(0, length(pos_tag_options))
    for (i in 1:length(pos_tag_options)) {
      tag = pos_tag_options[i]
      pos_dist[i] = sum(linguistics$pos_tags == tag)
    }
    names(pos_dist) = paste0("pos_", tolower(pos_tag_options))
    pos_dist_sd = sd(pos_dist)

    sentences = linguistics$sentences

    # CK: Can we convert this to tm_map(sents, removePunctation)? TBD.
    sentences = lapply(sentences, remove_punc)

    # Number of sentences.
    num_sentences = max(1, length(sentences))

    words_per_sentence_dist = stri_count(sentences, regex="\\S+")
    total_raw_words = max(1, sum(words_per_sentence_dist))
    avg_words_per_sentence = total_raw_words / num_sentences
    sd_words_per_sentence = ifelse(num_sentences > 1, sd(words_per_sentence_dist), 0)

    num_lines = length(text)
    words_per_line = stri_count(text_no_punct, regex="\\S+")
    avg_words_per_line = mean(words_per_line, na.rm=T)
    sd_words_per_line = ifelse(num_lines > 1, sd(words_per_line, na.rm=T), 0)

    punct_per_line = stri_count(text, regex="[^a-zA-Z0-9 ]")
    avg_punct_per_line = mean(punct_per_line, na.rm=T)
    sd_punct_per_line = ifelse(num_lines > 1, sd(punct_per_line, na.rm=T), 0)

    text_chars_per_line = stri_count(text, regex="[-a-zA-Z0-9_]")
    avg_text_chars_per_line = mean(text_chars_per_line, na.rm=T)
    sd_text_chars_per_line = ifelse(num_lines > 1, sd(text_chars_per_line, na.rm=T), 0)


    # Number of 4-digit numbers (years).
    digits_range = 2:6
    digit_features = rep(NA, length(digits_range))
    for (i in 1:length(digits_range)) {
      # We need to sum() the result because text is an array.
      digit_count = sum(str_count(text, paste0("[^\\d]\\d{", digits_range[i], "}[^\\d]")))
      digit_features[i] = digit_count
    }
    names(digit_features) = paste0("digit_count_", digits_range)

    # Number of digits.
    # CK: actually, this is number of sentences that contain a digit.
    sentences_with_digits = sum(grepl("[[:digit:]]", sentences))

    sentence_features = c(num_sentences=num_sentences, total_raw_words=total_raw_words,
                          sentences_with_digits=sentences_with_digits,
                          avg_words_per_sentence=avg_words_per_sentence,
                          sd_words_per_sentence=sd_words_per_sentence,
                          avg_words_per_line=avg_words_per_line,
                          sd_words_per_line=sd_words_per_line,
                          avg_punct_per_line=avg_punct_per_line,
                          sd_punct_per_line=sd_punct_per_line,
                          avg_text_chars_per_line=avg_text_chars_per_line,
                          sd_text_chars_per_line=sd_text_chars_per_line
    )

    lines_start_with_subject = sum(grepl("^subject ?:", text))

    subject_line_only = as.numeric(num_lines == 1 & lines_start_with_subject == 1)

    misc_features = c(lines_start_with_subject=lines_start_with_subject,
                      subject_line_only=subject_line_only, num_lines=num_lines,
                      pos_unique_tags=pos_unique_tags, pos_dist_sd=pos_dist_sd)

    # TODO: add in word_features once those exist.
    combined_features = c(character_features, digit_features, sentence_features, pos_dist, misc_features)
    if (sum(is.na(combined_features)) > 0) {
      cat("Error: some features have NAs. ")
      print(combined_features)
      stopifnot(F)
    }

    combined_features
  }) # for the previous sapply version.
  #}) # for the mclapply version.

  # TODO: generate 2-way interactions.

  # We need to transpose because sapply gives us each result as a column.
  #results = data.frame(t(feature_matrix))
  results = data.frame(t(feature_matrix))

  # Confirm that we have 1 result for each document in our corpus.
  stopifnot(nrow(results) == n)

  return(results)
}

# Tokenize raw message text into sentences.
analyze_linguistics = function(text) {
  # Check if the string is only whitespace.
  if (text %in% c("", " ") || str_count(text, "\\s") == str_length(text)) {
    return(c())
  }

  # Create the sentence token annotator as a static variable.
  # Hack via http://rsnippets.blogspot.com/2012/05/emulating-local-static-variables-in-r.html
  # We do this to prevent a "too many open files" error with openNLP.
  annotators = attr(analyze_linguistics, "annotator")
  if (is.null(annotators)) {
    annotators = list()
    # Create annotators for the first time.
    annotators$sentence = Maxent_Sent_Token_Annotator()
    annotators$word = Maxent_Word_Token_Annotator()
    annotators$pos = Maxent_POS_Tag_Annotator()

    # Save it as a static variable.
    attr(analyze_linguistics, "annotator") <<- annotators
  }

  # Otherwise process into sentences.
  text = as.String(text)
  # Need to specify NLP package, because ggplot2 also has an annotate function.
  text_sents_words = NLP::annotate(text, list(annotators$sentence, annotators$word))
  # Text_pos now contains sentence, word, and POS annotations.
  text_pos = annotate(text, annotators$pos, text_sents_words)
  sentences = text[subset(text_pos, type == "sentence")]
  words = text[subset(text_pos, type == "word")]
  # Restrict the sentence annotation to the words.
  text_pos_words = subset(text_pos, type == "word")
  pos_tags = sapply(text_pos_words$features, `[[`, "POS")
  pos_tags
  results = list(text_annotated=text_pos, sentences=sentences, words=words, pos_tags=pos_tags)
  return(results)
}

# Clean punctuation from raw text of messages.
remove_punc = function(x) {
  gsub("[[:punct:]]", "", x)
}

BigramTokenizer = function(x, ngrams = 2) {
  NgramTokenizer(x, ngrams)
}

# Separate function because it's unclear how to change a parameter in the DocumentTermMatrix call.
TrigramTokenizer = function(x, ngrams = 3) {
  NgramTokenizer(x, ngrams)
}

# Change ngrams to 3 when calling to get trigrams.
NgramTokenizer = function(x, ngrams = 2) {
  # Specify NLP package because qdap also provides the ngrams function.
  unlist(lapply(NLP::ngrams(words(x), ngrams), paste, collapse = " "), use.names = FALSE)
}

load_stopwords = function(input_file = "data/common-english-words.txt", output_file = "data/stopwords.Rdata",
                          reload_file=F) {
  # Just re-use the output file if it already exists.
  # Set reload_file = T if you want to force the function to reload the input file.
  if (!reload_file && length(output_file) > 0 && file.exists(output_file)) {
    load(output_file)
  } else {
    # Load the official stopword list and make sure it's the same as the one used by tm.
    file_con = file(input_file)
    # Process it as one line separated by commas, and convert it to a vector.
    stopwords = unlist(strsplit(readLines(file_con)[1], split=c(","), fixed=T))
    close(file_con)

    if (length(output_file) > 0) {
      save(stopwords, file=output_file)
    }
  }
  return(stopwords)
}
