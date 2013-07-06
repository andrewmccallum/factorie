package cc.factorie.app.nlp.coref

import cc.factorie.app.nlp.Document
import cc.factorie.util.coref.GenericEntityMap
import cc.factorie.app.nlp.mention.{MentionList, Mention}
import cc.factorie.app.nlp.wordnet.WordNet
import java.io.File

/**
 * User: apassos
 * Date: 6/27/13
 * Time: 1:01 PM
 */


trait WithinDocCoref2TrainerOpts extends cc.factorie.util.DefaultCmdOptions {
  val trainFile = new CmdOption("train", "conll-train-clean.txt", "STRING", "File with training data")
  val testFile = new CmdOption("test", "conll-test-clean.txt", "STRING", "File with testing data")
  val numPositivePairsTrain = new CmdOption("prune-train", 2, "INT", "number of positive pairs before pruning instances in training")
  val numPositivePairsTest = new CmdOption("prune-test", 100, "INT", "number of positive pairs before pruning instances in testing")
  val portion = new CmdOption("portion", 0.1, "DOUBLE", "Portion of corpus to load.")
  val serialize = new CmdOption("serialize", "N/A", "FILE", "Filename in which to serialize classifier.")
  val deserialize = new CmdOption("deserialize", "N/A", "FILE", "Filename from which to deserialize classifier.")
  val numThreads = new CmdOption("num-threads", 4, "INT", "Number of threads to use")
  val featureComputationsPerThread = new CmdOption("feature-computations-per-thread", 2, "INT", "Number of feature computations per thread to run in parallel during training")
  val numTrainingIterations = new CmdOption("num-training-iterations", 3, "INT", "Number of passes through the training data")
  val randomSeed = new CmdOption("random-seed", 0, "INT", "Seed for the random number generator")
  val writeConllFormat = new CmdOption("write-conll-format", false, "BOOLEAN", "Write CoNLL format data.")
  val useAverageIterate = new CmdOption("use-average-iterate", true, "BOOLEAN", "Use the average iterate instead of the last iterate?")
  val useMIRA = new CmdOption("use-mira", false, "BOOLEAN", "Whether to use MIRA as an optimizer")
  val saveFrequency = new CmdOption("save-frequency", 4, "INT", "how often to save the model between epochs")
  val useExactEntTypeMatch = new CmdOption("use-exact-entity-type-match", true, "BOOLEAN", "whether to require exact alignment between NER annotation and NP annotation")
  val trainPortionForTest = new CmdOption("train-portion-for-test", 0.1, "DOUBLE", "When testing on train, what portion to use.")
  val wnDir = new CmdOption("wordnet", "wordnet", "STRING", "Path to the wordnet database.")   //now we load from a jar
  val mergeFeaturesAtAll = new CmdOption("merge-features-at-all", true, "BOOLEAN", "Whether to merge features")
  val conjunctionStyle = new CmdOption("conjunction-style", "NONE", "NONE|HASH|SLOW", "What types of conjunction features to use - options are NONE, HASH, and SLOW (use slow string-based conjunctions).")
  val entityLR = new CmdOption("entity-left-right",false,"BOOLEAN","whether to do entity-based pruning in lr search")
  val slackRescale = new CmdOption("slack-rescale",2.0,"FLOAT","recall bias for hinge loss")
  val useNonGoldBoundaries = new CmdOption("use-nongold-boundaries",true,"BOOLEAN","whether to use non-gold mention boundaries")
  val mentionAlignmentShiftWidth = new CmdOption("alignment-width",0,"INT","tolerance on boundaries when aligning detected mentions to gt mentions")
  val useEntityType = new CmdOption("use-entity-type",true,"BOOLEAN","whether to use entity type info")
  val mergeAppositions = new CmdOption("mergeAppositions",false,"BOOLEAN","whether to merge appositions as a rule")
  val usePronounRules = new CmdOption("use-pronoun-rules",false,"BOOLEAN","whether to do deterministic assigning of pronouns and not consider pronouns for training")
}

object WithinDocCoref2Trainer {

  def printConll2011Format(doc: Document, map: GenericEntityMap[Mention], out: java.io.PrintStream) {
    val mappedMentions = doc.attr[MentionList]
    val (singleTokMents, multiTokMents) = mappedMentions.partition(_.span.length == 1)
    val beginningTokMap = multiTokMents.groupBy(_.span.head)
    val endingTokMap = multiTokMents.groupBy(_.span.last)
    val singleTokMap = singleTokMents.groupBy(_.span.head)
    val fId = doc.name
    val docName = fId.substring(0, fId.length() - 4)
    val partNum = fId.takeRight(3)

    out.println("#begin document (" + docName + "); part " + partNum)
    for (s <- doc.sentences) {
      for (ti <- 0 until s.tokens.size) {
        val beginningMents = beginningTokMap.get(s(ti))
        val endingMents = endingTokMap.get(s(ti))
        val singleTokMents = singleTokMap.get(s(ti))
        assert(singleTokMents.size <= 1)
        out.print(docName + " " + partNum.toInt + " " + (ti + 1) + " " + s(ti).string + " " + s(ti).posLabel.categoryValue + " - - - - - - - ")
        var ments = List[String]()
        if (beginningMents.isDefined) ments = beginningMents.get.reverse.map(m => "(" + map.reverseMap(m)).mkString("|") :: ments
        if (singleTokMents.isDefined) ments = singleTokMents.get.map(m => "(" + map.reverseMap(m) + ")").mkString("|") :: ments
        if (endingMents.isDefined) ments = endingMents.get.reverse.map(m => map.reverseMap(m) + ")").mkString("|") :: ments
        if (ments.size > 0) out.println(ments.mkString("|"))
        else out.println("-")
      }
      out.println()
    }
    out.println("#end document")
  }


  object opts extends WithinDocCoref2TrainerOpts


  def main(args: Array[String]) {
    opts.parse(args)
    val options = new Coref2Options
    //options that get serialized with the model
    options.setConfig("useEntityType",opts.useEntityType.value)

    // options which affect only learning
    options.useAverageIterate = opts.useAverageIterate.value
    options.numTrainingIterations = opts.numTrainingIterations.value
    options.trainPortionForTest = opts.trainPortionForTest.value
    options.useEntityLR = opts.entityLR.value
    options.saveFrequency =  opts.saveFrequency.value
    options.numThreads = opts.numThreads.value
    options.featureComputationsPerThread = opts.featureComputationsPerThread.value
    options.pruneNegTrain = opts.numPositivePairsTrain.value > 0
    options.pruneNegTest = opts.numPositivePairsTest.value > 0
    options.numPositivePairsTrain = opts.numPositivePairsTrain.value
    options.numPositivePairsTest = opts.numPositivePairsTest.value
    options.useExactEntTypeMatch = opts.useExactEntTypeMatch.value
    options.slackRescale = opts.slackRescale.value
    options.mentionAlignmentShiftWidth = opts.mentionAlignmentShiftWidth.value
    options.useNonGoldBoundaries = opts.useNonGoldBoundaries.value
    options.mergeMentionWithApposition = opts.mergeAppositions.value
    options.setConfig("usePronounRules",opts.usePronounRules.value)
    // options still in flux
    options.mergeFeaturesAtAll = opts.mergeFeaturesAtAll.value
    options.conjunctionStyle = opts.conjunctionStyle.value match {
      case "NONE" => options.NO_CONJUNCTIONS
      case "HASH" => options.HASH_CONJUNCTIONS
      case "SLOW" => options.SLOW_CONJUNCTIONS
      case s => sys.error("Unknown conjunction style: " + s)
    }

    println("** Arguments")
    val ignoreOpts = Set("config", "help", "version")

    for (o <- opts.values.toSeq.sortBy(_.name); if !ignoreOpts(o.name)) println(o.name + " = " + o.value)
    println()

    //val wn =   new WordNet(new File(opts.wnDir.value))   //cc.factorie.app.nlp.wordnet.WordNet
    val wn = WordNet
    val rng = new scala.util.Random(opts.randomSeed.value)

    val (trainDocs,trainPredMaps,testDocs,testTrueMaps) =  if(opts.useNonGoldBoundaries.value )
      makeTrainTestDataNonGold(opts.trainFile.value,opts.testFile.value,options)
    else makeTrainTestData(opts.trainFile.value,opts.testFile.value)

    val mentPairClsf =
      if (opts.deserialize.wasInvoked){
        val lr = new WithinDocCoref2()
        lr.deserialize(opts.deserialize.value)
        lr.doTest(testDocs, wn, testTrueMaps.toMap, "Test")
        lr
      }
      else{
        val lr = if (options.conjunctionStyle == options.HASH_CONJUNCTIONS) new ImplicitConjunctionWithinDocCoref2 else new WithinDocCoref2
        lr.options.setConfigHash(options.getConfigHash)
        lr.train(trainDocs, testDocs, wn, rng, trainPredMaps.toMap, testTrueMaps.toMap,opts.saveFrequency.wasInvoked,opts.saveFrequency.value,opts.serialize.value)
        lr
      }


    if (opts.serialize.wasInvoked && !opts.deserialize.wasInvoked)
      mentPairClsf.serialize(opts.serialize.value + "-final")

    if (opts.writeConllFormat.value) {
      val conllFormatPrinter = new CorefScorer[CorefMention]
      val conllFormatGold = new java.io.PrintStream(new java.io.File("conll-test.filteredgold"))
      testDocs.foreach(d => conllFormatPrinter.printConll2011Format(d, testTrueMaps(d.name), conllFormatGold))
      conllFormatGold.flush()
      conllFormatGold.close()

      val conllFormatGold2 = new java.io.PrintStream(new java.io.File("conll-test.nonfilteredgold"))
      testDocs.foreach(d => printConll2011Format(d, testTrueMaps(d.name), conllFormatGold2))
      conllFormatGold2.flush()
      conllFormatGold2.close()
    }
  }

  def makeTrainTestData(trainFile: String, testFile: String): (Seq[Document],collection.mutable.Map[String,GenericEntityMap[Mention]],Seq[Document],collection.mutable.Map[String,GenericEntityMap[Mention]]) = {
    val allTrainDocs = ConllCorefLoader.loadWithParse(trainFile)
    val allTestDocs  =  ConllCorefLoader.loadWithParse(testFile)

    val trainDocs = allTrainDocs.take((allTrainDocs.length*opts.portion.value).toInt)
    val testDocs = allTestDocs.take((allTestDocs.length*opts.portion.value).toInt)
    println("Train: "+trainDocs.length+" documents, " + trainDocs.map(d => d.attr[MentionList].length).sum.toFloat / trainDocs.length + " mentions/doc")
    println("Test : "+ testDocs.length+" documents, " + testDocs.map(d => d.attr[MentionList].length).sum.toFloat / testDocs.length + " mention/doc")

    val testEntityMaps =  collection.mutable.Map(testDocs.map(d  => d.name -> (new BaseCorefModel).generateTrueMap(d.attr[MentionList])).toSeq: _*)
    val trainEntityMaps = collection.mutable.Map(trainDocs.map(d => d.name -> (new BaseCorefModel).generateTrueMap(d.attr[MentionList])).toSeq: _*)


    (trainDocs,trainEntityMaps,testDocs,testEntityMaps)
  }


  def makeTrainTestDataNonGold(trainFile: String, testFile: String, options: Coref2Options): (Seq[Document],collection.mutable.Map[String,GenericEntityMap[Mention]],Seq[Document],collection.mutable.Map[String,GenericEntityMap[Mention]]) = {
    import cc.factorie.app.nlp.Implicits._
    val (trainDocs,trainMap) = MentionAlignment.makeLabeledData(trainFile,null,opts.portion.value,options.useEntityType, options)
    val (testDocs,testMap) = MentionAlignment.makeLabeledData(testFile,null,opts.portion.value,options.useEntityType, options)
    (trainDocs,trainMap,testDocs,testMap)
  }
}
