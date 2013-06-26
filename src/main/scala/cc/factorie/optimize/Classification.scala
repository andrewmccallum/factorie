package cc.factorie.optimize

import cc.factorie._
import cc.factorie.la.{Tensor2, DenseTensor2, Tensor1, DenseTensor1}

/**
 * User: apassos
 * Date: 6/12/13
 * Time: 3:54 PM
 */


trait BaseClassification[+Pred] {
  def score: Pred
  def proportions: Proportions
  def bestLabelIndex: Int 
}

// TODO Does this trait need to exist?  When would you have either Binary or Multiclass, and not care which?
// TODO Consider swapping order on Pred & Input, to match order of Function1; then swap in all subclasses also
trait BaseClassifier[+Pred, -Input] {
  def score(features: Input): Pred
  def classification(features: Input): BaseClassification[Pred]
}

class BinaryClassification(val score: Double) extends BaseClassification[Double] {
  lazy val proportions = {
    val t = new DenseTensor1(2)
    t(1) = LinearObjectives.logisticLinkFunction(score)
    t(0) = 1.0-t(1)
    new DenseTensorProportions1(t)
  }
  lazy val bestValue = score > 0
  def bestLabelIndex: Int = if (score > 0) 1 else 0
}

// TODO Rename to MulticlassClassification
// TODO I wish "score" were named "scores" instead.  Does there need to be a BaseClassification in common between the Binary and Multiclass?
class MultiClassClassification(val score: Tensor1) extends BaseClassification[Tensor1] {
  lazy val proportions: Proportions1 = score match { case p:Proportions1 => p case _ => new DenseTensorProportions1(score.expNormalized.asInstanceOf[Tensor1]) }
  lazy val bestLabelIndex: Int = score.maxIndex
}

trait BaseBinaryClassifier[Features] extends BaseClassifier[Double, Features] {
  def classification(features: Features) = new BinaryClassification(score(features))
}

// TODO Rename this to MulticlassClassifier
trait MultiClassClassifier[Features] extends BaseClassifier[Tensor1, Features] {
  def classification(features: Features) = new MultiClassClassification(score(features))
}

// TODO These trainers just run to convergence.  We should make it eaiser to train a bit, then more...

// TODO Rename this to BaseLinearMulticlassTrainer, because this already insists on Tensor1 inputs
trait MultiClassTrainerBase[+C <: MultiClassClassifier[Tensor1]] {
  // TODO Rename this to just "train"?
  // TODO Use DoubleSeq instead of Seq[Double] for efficiency with lots of instances
  def simpleTrain(labelSize: Int, featureSize: Int, labels: Seq[Int], features: Seq[Tensor1], weights: Seq[Double], evaluate: C => Unit): C

  // TOOD I would like to make it easier to provide a custom "evaluate"
  // TODO Use logging instead of println, so it could be redirected at a high level and subject to log levels
  def train(labels: Seq[LabeledDiscreteVar], features: Seq[DiscreteTensorVar], weights: Seq[Double], testLabels: Seq[LabeledDiscreteVar], testFeatures: Seq[TensorVar]): C= {
    val evaluate = (c: C) => println(f"Test accuracy: ${testFeatures.map(i => c.classification(i.value.asInstanceOf[Tensor1]).bestLabelIndex)
                                                                                         .zip(testLabels).count(i => i._1 == i._2.targetIntValue).toDouble/testLabels.length}%1.4f")
    simpleTrain(labels.head.domain.size, features.head.domain.dimensionSize, labels.map(_.targetIntValue), features.map(_.value), weights, evaluate)
  }
  def train(labels: Seq[LabeledDiscreteVar], features: Seq[DiscreteTensorVar], testLabels: Seq[LabeledDiscreteVar], testFeatures: Seq[TensorVar]): C =
    train(labels, features, labels.map(i => 1.0), testLabels, testFeatures)
  def train(labels: Seq[LabeledDiscreteVar], features: Seq[DiscreteTensorVar], weights: Seq[Double]): C =
    simpleTrain(labels.head.domain.size, features.head.domain.dimensionSize, labels.map(_.targetIntValue), features.map(_.value), weights, c => ())
  def train(labels: Seq[LabeledDiscreteVar], features: Seq[DiscreteTensorVar]): C =
    simpleTrain(labels.head.domain.size, features.head.domain.dimensionSize, labels.map(_.targetIntValue), features.map(_.value), labels.map(i => 1.0), c => ())
  def train(labels: Seq[LabeledDiscreteVar], features: Seq[DiscreteTensorVar], weights: Seq[Double], evaluate: C => Unit): C =
    simpleTrain(labels.head.domain.size, features.head.domain.dimensionSize, labels.map(_.targetIntValue), features.map(_.value), weights, evaluate)
  def train(labels: Seq[LabeledDiscreteVar], features: Seq[DiscreteTensorVar], evaluate: C => Unit): C =
    simpleTrain(labels.head.domain.size, features.head.domain.dimensionSize, labels.map(_.targetIntValue), features.map(_.value), labels.map(i => 1.0), evaluate)
  def train[Label<:LabeledDiscreteVar](labels: Seq[Label], l2f: Label => DiscreteTensorVar, testLabels: Seq[Label], l2w: Label => Double = (l: Label) => 1.0): C =
    train(labels, labels.map(l2f), labels.map(l2w), testLabels, testLabels.map(l2f))
  def train[Label<:LabeledDiscreteVar](labels: Seq[Label], l2f: Label => DiscreteTensorVar, l2w: Label => Double = (l: Label) => 1.0): C =
    train(labels, labels.map(l2f), labels.map(l2w))
}

class ClassifierTemplate2[T <: DiscreteVar](val l2f: T => TensorVar, val classifier: MultiClassClassifier[Tensor1])(implicit ml: Manifest[T], mf: Manifest[TensorVar]) extends Template2[T, TensorVar] {
  def unroll1(v: T) = Factor(v, l2f(v))
  def unroll2(v: TensorVar) = Nil
  def score(v1: T#Value, v2: TensorVar#Value): Double = classifier.score(v2.asInstanceOf[Tensor1])(v1.asInstanceOf[DiscreteValue].intValue)
}

class LinearBinaryClassifier(val featureSize: Int) extends BaseBinaryClassifier[Tensor1] with Parameters {
  val weights = Weights(new DenseTensor1(featureSize))
  def score(features: Tensor1) = weights.value.dot(features)
}

// TODO Rename labelSize to ouputDomainSize and featureSize to inputDomainSize
// TODO Most of the rest of FACTORIE deals with Factors.  What would be so awkward about having a Family and Factor here?
//  With a Factor we could at least make the corresponding Classification extend DiscreteMarginal1Factor2, 
//  and thus use this Classifier to help construct a Summary
class LinearMultiClassClassifier(val labelSize: Int, val featureSize: Int) extends MultiClassClassifier[Tensor1] with Parameters {
  val weights = Weights(new DenseTensor2(labelSize, featureSize))
  def score(features: Tensor1) = weights.value * features
  def asTemplate[T <: LabeledMutableDiscreteVar[_]](l2f: T => TensorVar)(implicit ml: Manifest[T]) = new DotTemplateWithStatistics2[T,TensorVar] {
    def unroll1(v: T) = Factor(v, l2f(v))
    def unroll2(v: TensorVar) = Nil // TODO This might be dangerous; hard to know what the alternative is though. -akm
    val weights = LinearMultiClassClassifier.this.weights
  }
}

class LinearMultiClassTrainer(val optimizer: GradientOptimizer,
                        val useParallelTrainer: Boolean,
                        val useOnlineTrainer: Boolean,
                        val objective: LinearObjectives.MultiClass,
                        val maxIterations: Int,
                        val miniBatch: Int,
                        val nThreads: Int)(implicit random: scala.util.Random) extends MultiClassTrainerBase[LinearMultiClassClassifier]
{
  // TODO Make it easier to override the type of classifier,... but it hard to predict what all the necessary constructor arguments might be.
  //def newClassifier: LinearMultiClassClassifier = new LinearMultiClassClassifier(labelSize, featureSize)
  def baseTrain(classifier: LinearMultiClassClassifier, labels: Seq[Int], features: Seq[Tensor1], weights: Seq[Double], evaluate: LinearMultiClassClassifier => Unit) {
    val examples = (0 until labels.length).map(i => new LinearMultiClassExample(classifier.weights, features(i), labels(i), objective, weight=weights(i)))
    Trainer.train(parameters=classifier.parameters, examples=examples, maxIterations=maxIterations, evaluate = () => evaluate(classifier), optimizer=optimizer, useParallelTrainer=useParallelTrainer, useOnlineTrainer=useOnlineTrainer, miniBatch=miniBatch, nThreads=nThreads)
  }
  def simpleTrain(labelSize: Int, featureSize: Int, labels: Seq[Int], features: Seq[Tensor1], weights: Seq[Double], evaluate: LinearMultiClassClassifier => Unit) = {
    val classifier = new LinearMultiClassClassifier(labelSize, featureSize)
    baseTrain(classifier, labels, features, weights, evaluate)
    classifier
  }
}

class SVMMultiClassTrainer(parallel: Boolean=false)(implicit random: scala.util.Random) extends LinearMultiClassTrainer(optimizer=null, useParallelTrainer=parallel, useOnlineTrainer=false, objective=null, miniBatch= -1, maxIterations= -1, nThreads= -1) {
  override def baseTrain(classifier: LinearMultiClassClassifier, labels: Seq[Int], features: Seq[Tensor1], weights: Seq[Double], evaluate: LinearMultiClassClassifier => Unit) {
    val ll = labels.toArray
    val ff = features.toArray
    val numLabels = classifier.weights.value.dim1
    val weightTensor = {
      if (parallel) (0 until numLabels).par.map { label => (new LinearL2SVM).train(ff, ll, label) }
      else (0 until numLabels).map { label => (new LinearL2SVM).train(ff, ll, label) }
    }
    val weightsValue = classifier.weights.value
    for (f <- 0 until weightsValue.dim2; (l,t) <- (0 until numLabels).zip(weightTensor)) {
      weightsValue(l,f) = t(f)
    }
    evaluate(classifier)
  }
}

class OnlineLinearMultiClassTrainer(useParallel:Boolean = false,
                              optimizer: GradientOptimizer = new AdaGrad with ParameterAveraging,
                              objective: LinearObjectives.MultiClass = LinearObjectives.sparseLogMultiClass,
                              maxIterations: Int = 3,
                              miniBatch: Int = -1,
                              nThreads: Int = Runtime.getRuntime.availableProcessors())(implicit random: scala.util.Random)
  extends LinearMultiClassTrainer(optimizer, useParallel, useOnlineTrainer = true, objective, maxIterations, miniBatch, nThreads) {}

class BatchLinearMultiClassTrainer(useParallel:Boolean = true,
                             optimizer: GradientOptimizer = new LBFGS with L2Regularization,
                             objective: LinearObjectives.MultiClass = LinearObjectives.sparseLogMultiClass,
                             maxIterations: Int = 200,
                             nThreads: Int = Runtime.getRuntime.availableProcessors())(implicit random: scala.util.Random)
  extends LinearMultiClassTrainer(optimizer, useParallel, useOnlineTrainer = false, objective, maxIterations, -1, nThreads) {}

