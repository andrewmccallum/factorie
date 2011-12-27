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

package cc.factorie.app.nlp.parse
import cc.factorie._
import cc.factorie.app.nlp._
import java.lang.StringBuffer

// Representation for a dependency parse

object ParseLabelDomain extends CategoricalDomain[String]
class ParseLabel(val edge:ParseEdge, targetValue:String) extends LabelVariable(targetValue) { def domain = ParseLabelDomain } 

object ParseFeaturesDomain extends CategoricalVectorDomain[String]
class ParseFeatures(val token:Token) extends BinaryFeatureVectorVariable[String] { def domain = ParseFeaturesDomain }

class ParseEdge(theChild:Token, initialParent:Token, labelString:String) extends ArrowVariable[Token,Token](theChild, initialParent) {
  @inline final def child = src
  @inline final def parent = dst
  val label = new ParseLabel(this, labelString)
  val childEdges = new ParseChildEdges
  def children = childEdges.value.map(_.child)

  // Initialization
  child.attr += this // Add the edge as an attribute to the child node.
  if (initialParent ne null) initialParent.attr[ParseEdge].childEdges.add(this)(null)
  // Note that the line above requires that the parent already have a ParseEdge attribute.
  // One way to avoid this need is to create the ParseEdges with null parents, and then set them later.
  
  override def set(newParent: Token)(implicit d: DiffList): Unit =
    if (newParent ne parent) {
      // update parent's child pointers
      if (parent ne null) parent.attr[ParseEdge].childEdges.remove(this)
      if (newParent ne null) newParent.attr[ParseEdge].childEdges.add(this)
      super.set(newParent)(d)
    }
}

class ParseChildEdges extends SetVariable[ParseEdge]

// Example usages:
// token.attr[ParseEdge].parent
// token.attr[ParseEdge].childEdges.map(_.child)
// token.attr[ParseChildEdges].map(_.child)


// Proposed new style

object ParseTreeLabelDomain extends CategoricalDomain[String]
class ParseTreeLabel(val tree:ParseTree, targetValue:String = "O") extends LabelVariable(targetValue) { def domain = ParseTreeLabelDomain }

object ParseTree {
  val rootIndex = -1
  val noIndex = -2
}
class ParseTree(val sentence:Sentence) {
  private val _parents = new Array[Int](sentence.length)
  private val _labels = Array.tabulate(sentence.length)(i => new ParseTreeLabel(this))
  /** Returns the position in the sentence of the root token. */ 
  def rootChildIndex: Int = firstChild(-1)
  /** Return the token at the root of the parse tree.  The parent of this token is null.  The parentIndex of this position is -1. */
  def rootChild: Token = sentence(rootChildIndex)
  /** Make the argument the root of the tree.  This method does not prevent their being two roots. */
  def setRootChild(token:Token): Unit = setParent(token.position - sentence.start, -1)
  /** Returns the sentence position of the parent of the token at position childIndex */
  def parentIndex(childIndex:Int): Int = _parents(childIndex)
  /** Returns the parent token of the token at position childIndex */
  def parent(index:Int): Token = sentence(_parents(index))
  /** Returns the parent token of the given token */
  def parent(token:Token): Token = { require(token.sentence eq sentence); parent(token.position - sentence.start) }
  /** Set the parent of the token at position 'child' to be at position 'parentIndex'.  A parentIndex of -1 indicates the root.  */
  def setParent(childIndex:Int, parentIndex:Int): Unit = _parents(childIndex) = parentIndex
  /** Set the parent of the token 'child' to be 'parent'. */
  def setParent(child:Token, parent:Token): Unit = {
    require(child.sentence eq sentence)
    if (parent eq null) {
      _parents(child.position - sentence.start) = -1
    } else {
      require(parent.sentence eq sentence)
      _parents(child.position - sentence.start) = parent.position - sentence.start
    }
  }
  /** Return the sentence index of the first token whose parent is 'parentIndex' */
  private def firstChild(parentIndex:Int): Int = {
    var i = 0
    while ( i < _parents.length) {
      if (_parents(i) == parentIndex) return i
      i += 1
    }
    return -1
  }
  /** Return a list of tokens who are the children of the token at sentence position 'parentIndex' */
  def children(parentIndex:Int): Seq[Token] = {
    val result = new scala.collection.mutable.ArrayBuffer[Token]
    var i = 0
    while (i < _parents.length) {
      if (_parents(i) == parentIndex) result += sentence(i)
      i += 1
    }
    result
  }
  def leftChildren(parentIndex:Int): Seq[Token] = {
    val result = new scala.collection.mutable.ArrayBuffer[Token]
    var i = 0
    while (i < parentIndex) {
      if (_parents(i) == parentIndex) result += sentence(i)
      i += 1
    }
    result
  }
  def rightChildren(parentIndex:Int): Seq[Token] = {
    val result = new scala.collection.mutable.ArrayBuffer[Token]
    var i = parentIndex+1
    while (i > _parents.length) {
      if (_parents(i) == parentIndex) result += sentence(i)
      i += 1
    }
    result
  }
  /** Return a list of tokens who are the children of parentToken */
  //def children(parentToken:Token): Seq[Token] = children(parentToken.position - sentence.start)
  /** Return a list of tokens who are the children of the token at sentence position 'parentIndex' and who also have the indicated label value. */
  def childrenLabeled(index:Int, labelIntValue:Int): Seq[Token] = {
    val result = new scala.collection.mutable.ArrayBuffer[Token]
    var i = 0
    while (i < _parents.length) {
      if (_parents(i) == index && _labels(i).intValue == labelIntValue) result += sentence(i)
      i += 1
    }
    result
  }
  def leftChildrenLabeled(parentIndex:Int, labelIntValue:Int): Seq[Token] = {
    val result = new scala.collection.mutable.ArrayBuffer[Token]
    var i = 0
    while (i < parentIndex) {
      if (_parents(i) == parentIndex && _labels(i).intValue == labelIntValue) result += sentence(i)
      i += 1
    }
    result
  }
  def rightChildrenLabeled(parentIndex:Int, labelIntValue:Int): Seq[Token] = {
    val result = new scala.collection.mutable.ArrayBuffer[Token]
    var i = parentIndex+1
    while (i > _parents.length) {
      if (_parents(i) == parentIndex && _labels(i).intValue == labelIntValue) result += sentence(i)
      i += 1
    }
    result
  }
  //def childrenOfLabel(token:Token, labelIntValue:Int): Seq[Token] = childrenOfLabel(token.position - sentence.start, labelIntValue)
  //def childrenLabeled(index:Int, labelValue:DiscreteValue): Seq[Token] = childrenLabeled(index, labelValue.intValue) 
  //def childrenOfLabel(token:Token, labelValue:DiscreteValue): Seq[Token] = childrenOfLabel(token.position - sentence.start, labelValue.intValue)
  /** Return the label on the edge from the child at sentence position 'index' to its parent. */
  def label(index:Int): ParseTreeLabel = _labels(index)
  /** Return the label on the edge from 'childToken' to its parent. */
  //def label(childToken:Token): ParseTreeLabel = { require(childToken.sentence eq sentence); label(childToken.position - sentence.start) }
  override def toString(): String = {
    val tokenStrings = sentence.tokens.map(_.string)
    val labelStrings = _labels.map(_.value.toString())
    val buff = new StringBuffer()
    for (i <- 0 until sentence.size)
      buff.append(i + " " + _parents(i) + " " + tokenStrings(i) + " " + labelStrings(i) + "\n")
    buff.toString()
  }
}

// Example usages:
// token.sentence.attr[ParseTree].parent(token)
// sentence.attr[ParseTree].children(token)
// sentence.attr[ParseTree].setParent(token, parentToken)
// sentence.attr[ParseTree].label(token)
// sentence.attr[ParseTree].label(token).set("SUBJ")

// Methods also created in Token supporting:
// token.parseParent
// token.setParseParent(parentToken)
// token.parseChildren
// token.parseLabel
// token.leftChildren