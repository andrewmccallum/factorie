package cc.factorie.app.classify2

import cc.factorie._
import cc.factorie.la.Tensor1
import cc.factorie.optimize.MultiClassClassification
import scala.collection.mutable.{HashMap,ArrayBuffer}


// TODO Should we make a separate BinaryClassification? -akm

///** A collection of Classification results, along with methods for calculating several evaluation measures.
//    You can subclass Trial to add new evaluation measures. */
//class Trial[L<:LabeledMutableDiscreteVar[_]](val classifier:Classifier[L])
//  extends LabeledDiscreteEvaluation(classifier.labelDomain.asInstanceOf[CategoricalDomain[String]]) with IndexedSeq[Classification[L]] with ClassifierEvaluator[L] {
//  private val classifications = new ArrayBuffer[Classification[L]]
//  def length = classifications.length
//  def apply(i:Int) = classifications(i)
//  def +=(label:L): Unit = { classifications.+=(classifier.classify(label)); super.+=(label) }
//  def ++=(labels:Iterable[L]): this.type = { labels.foreach(+=(_)); this }
//  def +=(c:Classification[L]): Unit = {
//    if (c.classifier == classifier) classifications += c
//    else throw new Error("Classification.classifier does not match.")
//  }
//  override def toString: String = "OVERALL: " + overallEvalString + "\n" +  evalString
//}
