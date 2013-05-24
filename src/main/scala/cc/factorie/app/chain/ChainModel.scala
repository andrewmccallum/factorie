/* Copyright (C) 2008-2010 University of Massachusetts Amherst,
   Department of Computer Science.
   This file is part of "FACTORIE" (Factor graphs, Imperative, Extensible)
   http://factorie.cs.umass.edu, http://code.google.com/p/factorie/
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

package cc.factorie.app.chain

import cc.factorie._
import cc.factorie.maths.ArrayOps
import cc.factorie.la._
import cc.factorie.optimize._
import cc.factorie.app.chain.infer._
import scala.collection.mutable.{ListBuffer,ArrayBuffer}
import java.io.File
import scala.collection.mutable
import org.junit.Assert._
import scala.collection.mutable.LinkedHashMap
import cc.factorie.util.{BinarySerializer, DoubleAccumulator}

class ChainModel[Label<:LabeledMutableDiscreteVarWithTarget[_], Features<:CategoricalDimensionTensorVar[String], Token<:Observation[Token]]
(val labelDomain:CategoricalDomain[String],
 val featuresDomain:CategoricalDimensionTensorDomain[String],
 val labelToFeatures:Label=>Features,
 val labelToToken:Label=>Token,
 val tokenToLabel:Token=>Label) 
 (implicit lm:Manifest[Label], fm:Manifest[Features], tm:Manifest[Token])
extends ModelWithContext[IndexedSeq[Label]] //with Trainer[ChainModel[Label,Features,Token]]
with Parameters
{
  val labelClass = lm.erasure
  val featureClass = fm.erasure
  val tokenClass = tm.erasure
  val bias = new DotFamilyWithStatistics1[Label] {
    factorName = "Label"
    val weights = Weights(new la.DenseTensor1(labelDomain.size))
  }
  val obs = new DotFamilyWithStatistics2[Label,Features] {
    factorName = "Label,Token"
    val weights = Weights(new la.DenseTensor2(labelDomain.size, featuresDomain.dimensionSize))
  }
  val markov = new DotFamilyWithStatistics2[Label,Label] {
    factorName = "Label,Label"
    val weights = Weights(new la.DenseTensor2(labelDomain.size, labelDomain.size))
  }
  val obsmarkov = new DotFamilyWithStatistics3[Label,Label,Features] {
    factorName = "Label,Label,Token"
    val weights = Weights(if (useObsMarkov) new la.DenseTensor3(labelDomain.size, labelDomain.size, featuresDomain.dimensionSize) else new la.DenseTensor3(1, 1, 1))
  }
  var useObsMarkov = false
  // NOTE if I don't annotate this type here I get a crazy compiler crash when it finds the upper bounds of these template types -luke
  def families: Seq[DotFamily] = if (useObsMarkov) Seq(bias, obs, markov, obsmarkov) else Seq(bias, obs, markov)

  // TODO this does not calculate statistics for obsmarkov template -luke
  def marginalStatistics(labels: Seq[Label], nodeMargs: Array[Array[Double]], edgeMargs: Array[Array[Double]]): WeightsMap = {
    val biasStats = Tensor.newDense(bias.weights.value)
    val obsStats = Tensor.newSparse(obs.weights.value)
    val obsmarkovStats = if (useObsMarkov) Tensor.newSparse(obsmarkov.weights.value) else null

    var li = 0
    while (li < labels.length) {
      val l = labels(li)
      val marg = nodeMargs(li)
      val features = labelToFeatures(l).tensor
      var mi = 0
      while (mi < marg.length) {
        val m = marg(mi)
        if (m != 0) {
          features.foreachActiveElement((fi, fv) => obsStats += (mi * features.length + fi, fv * m))
          biasStats(mi) += m
        }
        mi += 1
      }
      li += 1
    }

    val markovStats =
      if (labels.length > 1) new DenseTensor1(ArrayOps.elementwiseSum(edgeMargs))
      else new SparseTensor1(labels(0).domain.size * labels(0).domain.size)

    val result = parameters.blankSparseMap
    result(bias.weights) = biasStats
    result(obs.weights) = obsStats
    result(markov.weights) = markovStats
    if (useObsMarkov) result(obsmarkov.weights) = obsmarkovStats
    result
  }

  def assignmentStatistics(labels: Seq[Label], assignments: Seq[Int]): WeightsMap = {
    val biasStats = Tensor.newDense(bias.weights.value)
    val obsStats = Tensor.newSparse(obs.weights.value)
    val markovStats = Tensor.newSparse(markov.weights.value)
    val obsmarkovStats = if (useObsMarkov) Tensor.newSparse(obsmarkov.weights.value) else null

    for (i <- 0 until labels.length) {
      val l = labels(i)
      val features = labelToFeatures(l).tensor
      biasStats += (assignments(i), 1)
      obsStats += Tensor.outer(new SingletonBinaryTensor1(l.domain.size, assignments(i)), features)
      if (i > 0) {
        val prev = labels(i - 1)
        markovStats += (assignments(i - 1) * labelDomain.size + assignments(i), 1)
        if (useObsMarkov) obsmarkovStats += Tensor.outer(
          prev.tensor.asInstanceOf[Tensor1], l.target.asInstanceOf[Tensor1], features)
      }
    }

    val result = parameters.blankSparseMap
    result(bias.weights) = biasStats
    result(obs.weights) = obsStats
    result(markov.weights) = markovStats
    if (useObsMarkov) result(obsmarkov.weights) = obsmarkovStats
    result
  }

  def targetStatistics(labels: Seq[Label]): WeightsMap =
    assignmentStatistics(labels, labels.map(_.targetIntValue))

  def serialize(prefix: String) {
    val modelFile = new File(prefix + "-model")
    if (modelFile.getParentFile ne null)
      modelFile.getParentFile.mkdirs()
    BinarySerializer.serialize(this, modelFile)
    val labelDomainFile = new File(prefix + "-labelDomain")
    BinarySerializer.serialize(labelDomain, labelDomainFile)
    val featuresDomainFile = new File(prefix + "-featuresDomain")
    BinarySerializer.serialize(featuresDomain.dimensionDomain, featuresDomainFile)
  }

  def deSerialize(prefix: String) {
    val labelDomainFile = new File(prefix + "-labelDomain")
    assert(labelDomainFile.exists(), "Trying to load inexistent label domain file: '" + prefix + "-labelDomain'")
    BinarySerializer.deserialize(labelDomain, labelDomainFile)
    val featuresDomainFile = new File(prefix + "-featuresDomain")
    assert(featuresDomainFile.exists(), "Trying to load inexistent label domain file: '" + prefix + "-featuresDomain'")
    BinarySerializer.deserialize(featuresDomain.dimensionDomain, featuresDomainFile)
    val modelFile = new File(prefix + "-model")
    assert(modelFile.exists(), "Trying to load inexisting model file: '" + prefix + "-model'")
    assertEquals(markov.weights.value.length, labelDomain.length * labelDomain.length)
    BinarySerializer.deserialize(this, modelFile)
  }

  def factorsWithContext(labels:IndexedSeq[Label]): Iterable[Factor] = {
    val result = new ListBuffer[Factor]
    for (i <- 0 until labels.length) {
      result += bias.Factor(labels(i))
      result += obs.Factor(labels(i), labelToFeatures(labels(i)))
      if (i > 0) {
        result += markov.Factor(labels(i-1), labels(i))
        if (useObsMarkov) result += obsmarkov.Factor(labels(i-1), labels(i), labelToFeatures(labels(i)))
      }
    }
    result
  }
  override def factors(variables: Iterable[Var]): Iterable[Factor] = variables match {
    case variables: IndexedSeq[Label] if (variables.forall(v => labelClass.isAssignableFrom(v.getClass))) => factorsWithContext(variables)
    case _ => super.factors(variables)
  }
  def factors(v: Var) = v match {
    case label:Label if (label.getClass eq labelClass) => {
      val result = new ArrayBuffer[Factor](4)
      result += bias.Factor(label)
      result += obs.Factor(label, labelToFeatures(label))
      val token = labelToToken(label)
      if (token.hasPrev) {
        result += markov.Factor(tokenToLabel(token.prev), label)
        if (useObsMarkov)
          result += obsmarkov.Factor(tokenToLabel(token.prev), label, labelToFeatures(label))
      }
      if (token.hasNext) {
        result += markov.Factor(label, tokenToLabel(token.next))
        if (useObsMarkov)
          result += obsmarkov.Factor(label, tokenToLabel(token.next), labelToFeatures(tokenToLabel(token.next)))
      }
      result
    }
  }

  // Inference
  // TODO FIXME this does not work for useObsMarkov=true right now -luke
  def inferBySumProduct(labels: IndexedSeq[Label]): ChainSummary = {
    val summary = new ChainSummary {
      private val (__nodeMarginals, __edgeMarginals, _logZ) = ForwardBackward.marginalsAndLogZ(labels, obs, markov, bias, labelToFeatures)
      private val _expectations = marginalStatistics(labels, __nodeMarginals, __edgeMarginals)
      private val _nodeMarginals = new LinkedHashMap[Var, DiscreteMarginal] ++=
        labels.zip(__nodeMarginals).map({case (l, m) => l -> new DiscreteMarginal1(l, new DenseProportions1(m)) with MarginalWithoutTensorStatistics })
      private val _edgeMarginals =
        labels.zip(labels.drop(1)).zip(__edgeMarginals).map({ case ((l1, l2), m) =>
          // m is label X label
          val ds = labelDomain.length
          val t = new DenseProportions2(ds, ds)
          var d1 = 0
          while (d1 < ds) {
	          var d2 = 0
            while (d2 < ds) {
              t.masses(d1, d2) += m(d1 * ds + d2)
              d2 += 1
            }
	          d1 += 1
          }
          new DiscreteMarginal2(l1, l2, t) with MarginalWithoutTensorStatistics
        }).toArray
      def expectations = _expectations
      override def logZ = _logZ
      def marginals: Iterable[DiscreteMarginal] = _nodeMarginals.values
      def marginal(v: Var): DiscreteMarginal = _nodeMarginals(v)
      override def marginal(_f: Factor): DiscreteMarginal = {
        val f = _f.asInstanceOf[DotFamily#Factor]
        if (f.family == bias || f.family == obs)
          marginal(f.variables.head)
        else if (f.family == markov)
          _edgeMarginals(labels.indexOf(f.variables.head))
        else
          throw new Error("ChainModel marginals can only be returned for ChainModel factors")
      }
    }
    summary
  }

  // TODO learning doesn't work for this yet... -luke
  def inferByMaxProduct(labels: IndexedSeq[Label]): ChainSummary = {
    new ChainSummary {
      private val targetInts = Viterbi.search(labels, obs, markov, bias, labelToFeatures).toArray
      lazy private val variableTargetMap = labels.zip(targetInts).toMap
      private val variables = labels
      lazy private val _marginals = new LinkedHashMap[Var, DiscreteMarginal] ++=
        labels.zip(targetInts).map({case (l, t) => l -> new DiscreteMarginal1(l, new SingletonProportions1(labelDomain.size, t)) with MarginalWithoutTensorStatistics})
      def marginals: Iterable[DiscreteMarginal] = _marginals.values
      def marginal(v: Var): DiscreteMarginal = _marginals(v)
      override def setToMaximize(implicit d:DiffList): Unit = {
        var i = 0
        while (i < variables.length) {
          variables(i).set(targetInts(i))(d)
          i += 1
        }
      }
      val expectations = assignmentStatistics(labels, targetInts)
      override val logZ = expectations.toSeq.map({case (weights, stats) => weights.value dot stats}).sum
      override def marginal(_f: Factor): DiscreteMarginal = {
        val f = _f.asInstanceOf[DotFamily#Factor]
        if (f.family == bias)
          f match { case f: DotFamilyWithStatistics1[Label]#Factor => new DiscreteMarginal1(f._1, _marginals(f._1).proportions.asInstanceOf[Proportions1]) with DiscreteMarginal1Factor1[Label] { val factor = f } }
        else if (f.family == obs)
          f match { case f:DotFamilyWithStatistics2[Label,Features]#Factor => new DiscreteMarginal1(f._1, _marginals(f._1).proportions.asInstanceOf[Proportions1]) with DiscreteMarginal1Factor2[Label,Features] { val factor = f } }
        else if (f.family == markov) {
          val f2 = _f.asInstanceOf[Factor2[Label, Label]]
          val s = labelDomain.dimensionDomain.size
          new DiscreteMarginal2(f2._1, f2._2, new NormalizedTensorProportions2(new SingletonBinaryTensor2(s, s, variableTargetMap(f2._1), variableTargetMap(f2._2)))) with DiscreteMarginal2Factor2[Label,Label] { val factor = f2 }
        }
        else
          throw new Error("ChainModel marginals can only be returned for ChainModel factors")
      }
    }
  }
}

trait ChainSummary extends Summary[DiscreteMarginal] {
  // Do we actually want the marginal of arbitrary sets of variables? -brian
  def marginal(vs:Var*): DiscreteMarginal = null
  def marginal(v:Var): DiscreteMarginal
  def expectations: WeightsMap // TODO this should be 1-hot tensors for Viterbi -luke
}

object ChainModel {
  trait ChainInfer {
    def infer[L <: LabeledMutableDiscreteVarWithTarget[_]](variables: Iterable[L], model: ChainModel[L, _, _]): ChainSummary
  }
  object MarginalInference extends ChainInfer {
    def infer[L <: LabeledMutableDiscreteVarWithTarget[_]](variables: Iterable[L], model: ChainModel[L, _, _]): ChainSummary =
      model.inferBySumProduct(variables.asInstanceOf[IndexedSeq[L]])
  }
  object MAPInference extends ChainInfer {
    def infer[L <: LabeledMutableDiscreteVarWithTarget[_]](variables: Iterable[L], model: ChainModel[L, _, _]): ChainSummary =
      model.inferByMaxProduct(variables.asInstanceOf[IndexedSeq[L]])
  }
  class ChainExample[L <: LabeledMutableDiscreteVarWithTarget[_]](model: ChainModel[L, _, _], val labels: IndexedSeq[L], infer: ChainInfer = MarginalInference) extends Example {
    private var cachedTargetStats: WeightsMap = null
    def accumulateExampleInto(gradient: WeightsMapAccumulator, value: DoubleAccumulator): Unit = {
      if (labels.size == 0) return
      if (cachedTargetStats == null) cachedTargetStats = model.targetStatistics(labels)
      val summary = infer.infer(labels, model)
      if (gradient != null) {
        for ((weights, stats) <- cachedTargetStats.toSeq) gradient.accumulate(weights, stats)
        for ((weights, stats) <- summary.expectations.toSeq) gradient.accumulate(weights, stats, -1.0)
      }
      if (value != null) {
        val targetValue = cachedTargetStats.toSeq.map({case (weights, stats) => weights.value dot stats}).sum
        value.accumulate(targetValue - summary.logZ)
      }
    }
  }
  
  def createChainExample[L <: LabeledMutableDiscreteVarWithTarget[_]](model: ChainModel[L,_,_], labels: IndexedSeq[L]) = new ChainExample(model, labels)
}