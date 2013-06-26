package cc.factorie.app.classify2

import cc.factorie._
import cc.factorie.la.Tensor1
import cc.factorie.optimize._

/** Infrastructure for independent classification, assigning values to DiscreteVars. 
    @author Andrew McCallum
    @since 0.8
  */

// TODO Should we also store the input features here? -akm
/** The result of applying a VarClassifier to a DiscreteVar */
class VarClassification[+L<:DiscreteVar](theVar:L, theScores:Tensor1) extends MultiClassClassification(theScores) with DiscreteMarginal1[L] {
  val _1: L = theVar // For DiscreteMarginal1
  def bestDiscreteValue: L#Value = _1.domain.apply(bestLabelIndex).asInstanceOf[L#Value]
}

// TODO Should we have a type of Classification that extends DiscreteMarginal1Factor2, like this:
class VarLinearClassification[L<:DiscreteVar,I<:DiscreteTensorVar](theVar:L, theScores:Tensor1, val factor:Factor2[L,I]) extends VarClassification(theVar,theScores) with DiscreteMarginal1Factor2[L,I]

/** Performs independent prediction of (iid) Labels (all of which must share the same domain).
    Has abstract method "labelDomain". */
trait VarClassifier[L<:DiscreteVar,I] extends MultiClassClassifier[I] {
  type ClassificationType <: VarClassification[L]
  def varToInput: L=>I
  /** Return a record summarizing the outcome of applying this classifier to the given label.  Afterwards the label will have the same value it had before this call. */
  def classification(label:L): ClassificationType = new VarClassification(label, score(varToInput(label))).asInstanceOf[ClassificationType]
  def classifications(labels: Iterable[L]): Seq[ClassificationType] = labels.toSeq.map(label => classification(label))
  /** Set the label to classifier-predicted value and return a Classification object summarizing the outcome. */
  def classify(label: L): ClassificationType = {
    val c = classification(label)
    c.setToMaximize(null)
    c
  }
  def classify(labels: Iterable[L]): Seq[ClassificationType] = {
    val c = classifications(labels)
    c.foreach(_.setToMaximize(null))
    c
  }
}

class ModelBasedVarClassifier[L<:DiscreteVar,M<:Model](val model:M) extends VarClassifier[L,L] {
  def varToInput = l => l
  def score(label:L): Proportions1 = label.proportions(model) // TODO Change this to just get unnormalized scores for the label instead.
}

// TODO This could be a Model.  Any reason not to make it one?
class LinearVarClassifier[L<:DiscreteVar](outputDomainSize:Int, inputDomainSize:Int, val varToInput:L=>Tensor1) extends LinearMultiClassClassifier(outputDomainSize, inputDomainSize) with VarClassifier[L,Tensor1] {
  //type ClassificationType <: VarLinearClassification[L,Tensor1]
}

// TODO Have something like:  class ClassifierTrainingExamples {}

///** A classifier that uses a Model to score alternative label values. */
//class ModelBasedClassifier[L <: MutableDiscreteVar[_], Input, M <: Model](val model: M, val labelDomain: DiscreteDomain) extends Classifier[L,Input] {
//  def classification(label: L): Classification[L,Input] = {
//    require(label.domain eq labelDomain)
//    new Classification(label, this, label.proportions(model))
//  }
//}

///** An object that can train a Classifier given a LabelList. */
//trait ClassifierTrainer {
//  def train[L <: LabeledMutableDiscreteVar[_], F <: DiscreteTensorVar](il: LabelList[L, F]): Classifier[L]
//}
//
///** An object that can gather evaluation data for a Classifier, and convert it to a printable form. */
//trait ClassifierEvaluator[L <: MutableDiscreteVar[_]] {
//  def += (c: Classification[L]): Unit
//  def toString: String
//}
//
//// TODO Consider renaming this to MultiClassParameters -akm
//trait MultiClassModel extends Parameters {
//  val evidence: Weights2
//  def predict(feats: Tensor1): Int = scores(feats).maxIndex
//  def scores(feats: Tensor1): Tensor1 = evidence.value * feats
//}
//
//// TODO Consider renaming this to BinaryClassParameters -akm
//trait BinaryModel extends Parameters {
//  val evidence: Weights1
//  def predict(feats: Tensor1): Int = if (score(feats) > 0) 1 else -1
//  def score(feats: Tensor1): Double = evidence.value dot feats
//}